//===--------- command_buffer.cpp - HIP Adapter --------------------------===//
//
// Copyright (C) 2023 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "command_buffer.hpp"

#include <hip/hip_runtime.h>

#include "common.hpp"
#include "enqueue.hpp"
#include "event.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "queue.hpp"

#include <cstring>

ur_exp_command_buffer_handle_t_::ur_exp_command_buffer_handle_t_(
    ur_context_handle_t hContext, ur_device_handle_t hDevice)
    : Context(hContext),
      Device(hDevice), HIPGraph{nullptr}, HIPGraphExec{nullptr}, RefCount{1} {
  urContextRetain(hContext);
  urDeviceRetain(hDevice);
}

/// The ur_exp_command_buffer_handle_t_ destructor releases
/// all the memory objects allocated for command_buffer managment
ur_exp_command_buffer_handle_t_::~ur_exp_command_buffer_handle_t_() {
  // Release the memory allocated to the Context stored in the command_buffer
  UR_TRACE(urContextRelease(Context));

  // Release the device
  UR_TRACE(urDeviceRelease(Device));

  // Release the memory allocated to the HIPGraph
  hipGraphDestroy(HIPGraph);

  // Release the memory allocated to the HIPGraphExec
  hipGraphExecDestroy(HIPGraphExec);
}

/// Helper function for finding the HIP Nodes associated with the
/// commands in a command-buffer, each event is pointed to by a sync-point in
/// the wait list.
///
/// @param[in] CommandBuffer to lookup the events from.
/// @param[in] NumSyncPointsInWaitList Length of \p SyncPointWaitList.
/// @param[in] SyncPointWaitList List of sync points in \p CommandBuffer
/// to find the events for.
/// @param[out] HipNodesList Return parameter for the HIP Nodes associated with
/// each sync-point in \p SyncPointWaitList.
///
/// @return UR_RESULT_SUCCESS or an error code on failure
static ur_result_t getNodesFromSyncPoints(
    const ur_exp_command_buffer_handle_t &CommandBuffer,
    size_t NumSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *SyncPointWaitList,
    std::vector<hipGraphNode_t> &HIPNodesList) {
  // Map of ur_exp_command_buffer_sync_point_t to ur_event_handle_t defining
  // the event associated with each sync-point
  auto SyncPoints = CommandBuffer->SyncPoints;

  // For each sync-point add associated HIP graph node to the return list.
  for (size_t i = 0; i < NumSyncPointsInWaitList; i++) {
    if (auto NodeHandle = SyncPoints.find(SyncPointWaitList[i]);
        NodeHandle != SyncPoints.end()) {
      HIPNodesList.push_back(*NodeHandle->second.get());
    } else {
      return UR_RESULT_ERROR_INVALID_VALUE;
    }
  }
  return UR_RESULT_SUCCESS;
}

/// Set parameter for General 1D memory copy.
/// If the source and/or destination is on the device, SrcPtr and/or DstPtr
/// must be a pointer to a device ptr
static hipMemcpy3DParms setCopyParams(const void *SrcPtr,
                                      const hipMemoryType SrcType, void *DstPtr,
                                      const hipMemoryType DstType,
                                      size_t Size) {
  hipMemcpy3DParms Params = {};
  Params.extent.depth = 1;
  Params.extent.height = 1;
  Params.extent.width = Size;
  Params.srcPtr.ptr = const_cast<void *>(SrcPtr);
  Params.srcPtr.pitch = Size;
  Params.srcPtr.xsize = Size;
  Params.srcPtr.ysize = 1;
  Params.dstPtr.ptr = DstPtr;
  Params.dstPtr.pitch = Size;
  Params.dstPtr.xsize = Size;
  Params.dstPtr.ysize = 1;
  Params.kind = (SrcType == hipMemoryTypeDevice
                     ? (DstType == hipMemoryTypeDevice ? hipMemcpyDeviceToDevice
                                                       : hipMemcpyDeviceToHost)
                     : (DstType == hipMemoryTypeDevice ? hipMemcpyHostToDevice
                                                       : hipMemcpyHostToHost));
  return Params;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferCreateExp(
    ur_context_handle_t hContext, ur_device_handle_t hDevice,
    const ur_exp_command_buffer_desc_t *pCommandBufferDesc,
    ur_exp_command_buffer_handle_t *phCommandBuffer) {
  (void)pCommandBufferDesc;

  try {
    *phCommandBuffer = new ur_exp_command_buffer_handle_t_(hContext, hDevice);
  } catch (const std::bad_alloc &) {
    return UR_RESULT_ERROR_OUT_OF_HOST_MEMORY;
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }

  try {
    UR_CHECK_ERROR(hipGraphCreate(&(*phCommandBuffer)->HIPGraph, 0));
  } catch (...) {
    return UR_RESULT_ERROR_OUT_OF_RESOURCES;
  }

  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferRetainExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  hCommandBuffer->incrementReferenceCount();
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferReleaseExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  if (hCommandBuffer->decrementReferenceCount() != 0)
    return UR_RESULT_SUCCESS;

  delete hCommandBuffer;
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL
urCommandBufferFinalizeExp(ur_exp_command_buffer_handle_t hCommandBuffer) {
  try {
    const unsigned long long flags = 0;
    UR_CHECK_ERROR(hipGraphInstantiateWithFlags(
        &hCommandBuffer->HIPGraphExec, hCommandBuffer->HIPGraph, flags));
  } catch (...) {
    return UR_RESULT_ERROR_UNKNOWN;
  }
  return UR_RESULT_SUCCESS;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendKernelLaunchExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_kernel_handle_t hKernel,
    uint32_t workDim, const size_t *pGlobalWorkOffset,
    const size_t *pGlobalWorkSize, const size_t *pLocalWorkSize,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  // Preconditions
  UR_ASSERT(hCommandBuffer->Context == hKernel->getContext(),
            UR_RESULT_ERROR_INVALID_KERNEL);
  UR_ASSERT(workDim > 0, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);
  UR_ASSERT(workDim < 4, UR_RESULT_ERROR_INVALID_WORK_DIMENSION);

  ur_result_t Result = UR_RESULT_SUCCESS;
  hipGraphNode_t GraphNode;

  std::vector<hipGraphNode_t> DepsList;

  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList),
          Result);

  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  if (*pGlobalWorkSize == 0) {
    try {
      // Create an empty node if the kernel workload size is zero
      UR_CHECK_ERROR(hipGraphAddEmptyNode(&GraphNode, hCommandBuffer->HIPGraph,
                                          DepsList.data(), DepsList.size()));

      // Get sync point and register the HipNode with it.
      *pSyncPoint = hCommandBuffer->AddSyncPoint(
          std::make_shared<hipGraphNode_t>(GraphNode));
    } catch (ur_result_t Err) {
      Result = Err;
    }
    return Result;
  }

  // Set the number of threads per block to the number of threads per warp
  // by default unless user has provided a better number
  size_t ThreadsPerBlock[3] = {64u, 1u, 1u};
  size_t BlocksPerGrid[3] = {1u, 1u, 1u};

  uint32_t LocalSize = hKernel->getLocalSize();
  hipFunction_t HIPFunc = hKernel->get();
  Result =
      setKernelParams(hCommandBuffer->Context, hCommandBuffer->Device, workDim,
                      pGlobalWorkOffset, pGlobalWorkSize, pLocalWorkSize,
                      hKernel, HIPFunc, ThreadsPerBlock, BlocksPerGrid);
  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    // Set node param structure with the kernel related data
    auto &ArgIndices = hKernel->getArgIndices();
    hipKernelNodeParams NodeParams;
    NodeParams.func = HIPFunc;
    NodeParams.gridDim.x = BlocksPerGrid[0];
    NodeParams.gridDim.y = BlocksPerGrid[1];
    NodeParams.gridDim.z = BlocksPerGrid[2];
    NodeParams.blockDim.x = ThreadsPerBlock[0];
    NodeParams.blockDim.y = ThreadsPerBlock[1];
    NodeParams.blockDim.z = ThreadsPerBlock[2];
    NodeParams.sharedMemBytes = LocalSize;
    NodeParams.kernelParams = const_cast<void **>(ArgIndices.data());
    NodeParams.extra = nullptr;

    // Create and add an new kernel node to the HIP graph
    UR_CHECK_ERROR(hipGraphAddKernelNode(&GraphNode, hCommandBuffer->HIPGraph,
                                         DepsList.data(), DepsList.size(),
                                         &NodeParams));

    if (LocalSize != 0)
      hKernel->clearLocalSize();

    // Get sync point and register the HIPNode with it.
    *pSyncPoint = hCommandBuffer->AddSyncPoint(
        std::make_shared<hipGraphNode_t>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendUSMMemcpyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, void *pDst, const void *pSrc,
    size_t size, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  hipGraphNode_t GraphNode;
  std::vector<hipGraphNode_t> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList),
          Result);

  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    hipMemcpy3DParms NodeParams =
        setCopyParams(pSrc, hipMemoryTypeHost, pDst, hipMemoryTypeHost, size);

    UR_CHECK_ERROR(hipGraphAddMemcpyNode(&GraphNode, hCommandBuffer->HIPGraph,
                                         DepsList.data(), DepsList.size(),
                                         &NodeParams));

    // Get sync point and register the HIPNode with it.
    *pSyncPoint = hCommandBuffer->AddSyncPoint(
        std::make_shared<hipGraphNode_t>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, size_t srcOffset, size_t dstOffset, size_t size,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  hipGraphNode_t GraphNode;
  std::vector<hipGraphNode_t> DepsList;

  UR_ASSERT(size + dstOffset <= std::get<BufferMem>(hDstMem->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);
  UR_ASSERT(size + srcOffset <= std::get<BufferMem>(hSrcMem->Mem).getSize(),
            UR_RESULT_ERROR_INVALID_SIZE);

  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList),
          Result);

  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    auto Src = std::get<BufferMem>(hSrcMem->Mem)
                   .getPtrWithOffset(hCommandBuffer->Device, srcOffset);
    auto Dst = std::get<BufferMem>(hDstMem->Mem)
                   .getPtrWithOffset(hCommandBuffer->Device, dstOffset);

    hipMemcpy3DParms NodeParams = setCopyParams(&Src, hipMemoryTypeDevice, &Dst,
                                                hipMemoryTypeDevice, size);

    UR_CHECK_ERROR(hipGraphAddMemcpyNode(&GraphNode, hCommandBuffer->HIPGraph,
                                         DepsList.data(), DepsList.size(),
                                         &NodeParams));

    // Get sync point and register the HIPNode with it.
    *pSyncPoint = hCommandBuffer->AddSyncPoint(
        std::make_shared<hipGraphNode_t>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferAppendMemBufferCopyRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hSrcMem,
    ur_mem_handle_t hDstMem, ur_rect_offset_t srcOrigin,
    ur_rect_offset_t dstOrigin, ur_rect_region_t region, size_t srcRowPitch,
    size_t srcSlicePitch, size_t dstRowPitch, size_t dstSlicePitch,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  hipGraphNode_t GraphNode;
  std::vector<hipGraphNode_t> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList),
          Result);

  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    auto SrcPtr =
        std::get<BufferMem>(hSrcMem->Mem).getPtr(hCommandBuffer->Device);
    auto DstPtr =
        std::get<BufferMem>(hDstMem->Mem).getPtr(hCommandBuffer->Device);
    hipMemcpy3DParms NodeParams = {};

    setCopyRectParams(region, &SrcPtr, hipMemoryTypeDevice, srcOrigin,
                      srcRowPitch, srcSlicePitch, &DstPtr, hipMemoryTypeDevice,
                      dstOrigin, dstRowPitch, dstSlicePitch, NodeParams);

    UR_CHECK_ERROR(hipGraphAddMemcpyNode(&GraphNode, hCommandBuffer->HIPGraph,
                                         DepsList.data(), DepsList.size(),
                                         &NodeParams));

    // Get sync point and register the HIPNode with it.
    *pSyncPoint = hCommandBuffer->AddSyncPoint(
        std::make_shared<hipGraphNode_t>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, const void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  hipGraphNode_t GraphNode;
  std::vector<hipGraphNode_t> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList),
          Result);

  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    auto Dst = std::get<BufferMem>(hBuffer->Mem)
                   .getPtrWithOffset(hCommandBuffer->Device, offset);

    hipMemcpy3DParms NodeParams =
        setCopyParams(pSrc, hipMemoryTypeHost, &Dst, hipMemoryTypeDevice, size);

    UR_CHECK_ERROR(hipGraphAddMemcpyNode(&GraphNode, hCommandBuffer->HIPGraph,
                                         DepsList.data(), DepsList.size(),
                                         &NodeParams));

    // Get sync point and register the HIPNode with it.
    *pSyncPoint = hCommandBuffer->AddSyncPoint(
        std::make_shared<hipGraphNode_t>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    size_t offset, size_t size, void *pDst, uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  hipGraphNode_t GraphNode;
  std::vector<hipGraphNode_t> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList),
          Result);

  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    auto Src = std::get<BufferMem>(hBuffer->Mem)
                   .getPtrWithOffset(hCommandBuffer->Device, offset);

    hipMemcpy3DParms NodeParams =
        setCopyParams(&Src, hipMemoryTypeDevice, pDst, hipMemoryTypeHost, size);

    UR_CHECK_ERROR(hipGraphAddMemcpyNode(&GraphNode, hCommandBuffer->HIPGraph,
                                         DepsList.data(), DepsList.size(),
                                         &NodeParams));

    // Get sync point and register the HIPNode with it.
    *pSyncPoint = hCommandBuffer->AddSyncPoint(
        std::make_shared<hipGraphNode_t>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferWriteRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pSrc,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  hipGraphNode_t GraphNode;
  std::vector<hipGraphNode_t> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList),
          Result);

  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    auto DstPtr =
        std::get<BufferMem>(hBuffer->Mem).getPtr(hCommandBuffer->Device);
    hipMemcpy3DParms NodeParams = {};

    setCopyRectParams(region, pSrc, hipMemoryTypeHost, hostOffset, hostRowPitch,
                      hostSlicePitch, &DstPtr, hipMemoryTypeDevice,
                      bufferOffset, bufferRowPitch, bufferSlicePitch,
                      NodeParams);

    UR_CHECK_ERROR(hipGraphAddMemcpyNode(&GraphNode, hCommandBuffer->HIPGraph,
                                         DepsList.data(), DepsList.size(),
                                         &NodeParams));

    // Get sync point and register the hipNode with it.
    *pSyncPoint = hCommandBuffer->AddSyncPoint(
        std::make_shared<hipGraphNode_t>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT
ur_result_t UR_APICALL urCommandBufferAppendMemBufferReadRectExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_mem_handle_t hBuffer,
    ur_rect_offset_t bufferOffset, ur_rect_offset_t hostOffset,
    ur_rect_region_t region, size_t bufferRowPitch, size_t bufferSlicePitch,
    size_t hostRowPitch, size_t hostSlicePitch, void *pDst,
    uint32_t numSyncPointsInWaitList,
    const ur_exp_command_buffer_sync_point_t *pSyncPointWaitList,
    ur_exp_command_buffer_sync_point_t *pSyncPoint) {
  ur_result_t Result = UR_RESULT_SUCCESS;
  hipGraphNode_t GraphNode;
  std::vector<hipGraphNode_t> DepsList;
  UR_CALL(getNodesFromSyncPoints(hCommandBuffer, numSyncPointsInWaitList,
                                 pSyncPointWaitList, DepsList),
          Result);

  if (Result != UR_RESULT_SUCCESS) {
    return Result;
  }

  try {
    auto SrcPtr =
        std::get<BufferMem>(hBuffer->Mem).getPtr(hCommandBuffer->Device);
    hipMemcpy3DParms NodeParams = {};

    setCopyRectParams(region, &SrcPtr, hipMemoryTypeDevice, bufferOffset,
                      bufferRowPitch, bufferSlicePitch, pDst, hipMemoryTypeHost,
                      hostOffset, hostRowPitch, hostSlicePitch, NodeParams);

    UR_CHECK_ERROR(hipGraphAddMemcpyNode(&GraphNode, hCommandBuffer->HIPGraph,
                                         DepsList.data(), DepsList.size(),
                                         &NodeParams));

    // Get sync point and register the hipNode with it.
    *pSyncPoint = hCommandBuffer->AddSyncPoint(
        std::make_shared<hipGraphNode_t>(GraphNode));
  } catch (ur_result_t Err) {
    Result = Err;
  }
  return Result;
}

UR_APIEXPORT ur_result_t UR_APICALL urCommandBufferEnqueueExp(
    ur_exp_command_buffer_handle_t hCommandBuffer, ur_queue_handle_t hQueue,
    uint32_t numEventsInWaitList, const ur_event_handle_t *phEventWaitList,
    ur_event_handle_t *phEvent) {
  ur_result_t Result = UR_RESULT_SUCCESS;

  try {
    std::unique_ptr<ur_event_handle_t_> RetImplEvent{nullptr};
    ScopedContext Active(hQueue->getDevice());
    uint32_t StreamToken;
    ur_stream_quard Guard;
    hipStream_t HIPStream = hQueue->getNextComputeStream(
        numEventsInWaitList, phEventWaitList, Guard, &StreamToken);

    if ((Result = enqueueEventsWait(hQueue, HIPStream, numEventsInWaitList,
                                    phEventWaitList)) != UR_RESULT_SUCCESS) {
      return Result;
    }

    if (phEvent) {
      RetImplEvent = std::unique_ptr<ur_event_handle_t_>(
          ur_event_handle_t_::makeNative(UR_COMMAND_COMMAND_BUFFER_ENQUEUE_EXP,
                                         hQueue, HIPStream, StreamToken));
      UR_CHECK_ERROR(RetImplEvent->start());
    }

    // Launch graph
    UR_CHECK_ERROR(hipGraphLaunch(hCommandBuffer->HIPGraphExec, HIPStream));

    if (phEvent) {
      UR_CHECK_ERROR(RetImplEvent->record());
      *phEvent = RetImplEvent.release();
    }
  } catch (ur_result_t Err) {
    Result = Err;
  }

  return Result;
}
