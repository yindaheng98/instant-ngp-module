/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed.cu
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/common_device.cuh>
#include <neural-graphics-primitives/json_binding.h>
#include <neural-graphics-primitives/marching_cubes.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/nerf_network.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/takikawa_encoding.cuh>
#include <neural-graphics-primitives/testbed.h>
#include <neural-graphics-primitives/tinyexr_wrapper.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>
#include <neural-graphics-primitives/triangle_bvh.cuh>
#include <neural-graphics-primitives/triangle_octree.cuh>

#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/loss.h>
#include <tiny-cuda-nn/network.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/optimizer.h>
#include <tiny-cuda-nn/trainer.h>

#include <json/json.hpp>

#include <filesystem/directory.h>
#include <filesystem/path.h>

#include <zstr.hpp>

#include <fstream>
#include <set>
#include <unordered_set>

#ifdef NGP_GUI
#	include <imgui/backends/imgui_impl_glfw.h>
#	include <imgui/backends/imgui_impl_opengl3.h>
#	include <imgui/imgui.h>
#	include <imguizmo/ImGuizmo.h>
#	ifdef _WIN32
#		include <GL/gl3w.h>
#	else
#		include <GL/glew.h>
#	endif
#	include <GLFW/glfw3.h>
#	include <GLFW/glfw3native.h>
#	include <cuda_gl_interop.h>

#endif

// Windows.h is evil
#undef min
#undef max
#undef near
#undef far


using namespace std::literals::chrono_literals;

namespace ngp {
GPUMemory<bool> last_grid_hit;

void Testbed::do_grid_hit(GPUMemory<uint32_t>* grid_hit) {
    const uint64_t K = 64;
    uint64_t* counter_gpu;
    CUDA_CHECK_THROW(cudaMalloc(&counter_gpu, sizeof(uint64_t) * K));
    CUDA_CHECK_THROW(cudaMemset(counter_gpu, 0, sizeof(uint64_t) * K));
    parallel_for_gpu(m_stream.get(), grid_hit->size(), [grid_hit=grid_hit->data(), counter_gpu=counter_gpu, K=K] __device__ (size_t i) {
        for (uint64_t k=0;k<K;k++)
        if (grid_hit[i] > k) atomicAdd(counter_gpu + k, 1);
    });
    uint64_t counter_cpu[K];
    CUDA_CHECK_THROW(cudaMemcpyAsync(counter_cpu, counter_gpu, sizeof(uint64_t) * K, cudaMemcpyDeviceToHost, m_stream.get()));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
    CUDA_CHECK_THROW(cudaFree(counter_gpu));
    // for (uint64_t k=0;k<K;k++)
    // tlog::info() << grid_hit->data() << ' ' << counter_cpu[k] << '/' << grid_hit->size();
    tlog::info() << grid_hit->data() << ' ' << counter_cpu[0] << '/' << grid_hit->size();

    CUDA_CHECK_THROW(cudaMalloc(&counter_gpu, sizeof(uint64_t)));
    CUDA_CHECK_THROW(cudaMemset(counter_gpu, 0, sizeof(uint64_t)));
    if (last_grid_hit.size() != grid_hit->size()) {
        last_grid_hit.resize(grid_hit->size());
        last_grid_hit.memset(0);
    }
    parallel_for_gpu(m_stream.get(), grid_hit->size(), [grid_hit=grid_hit->data(), last_grid_hit=last_grid_hit.data(), counter_gpu=counter_gpu] __device__ (size_t i) {
        if (grid_hit[i] > 0) {
            if (!last_grid_hit[i])
                atomicAdd(counter_gpu, 1);
                last_grid_hit[i] = true;
        }
    });
    CUDA_CHECK_THROW(cudaMemcpyAsync(counter_cpu, counter_gpu, sizeof(uint64_t), cudaMemcpyDeviceToHost, m_stream.get()));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
    CUDA_CHECK_THROW(cudaFree(counter_gpu));
    tlog::info() << grid_hit->data() << ' ' << counter_cpu[0] << " not overlap";
}

}

