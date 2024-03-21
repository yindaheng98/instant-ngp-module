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
GPUMemory<bool> accu_grid_hit;
GPUMemory<bool> last_grid_hit;
GPUMemory<int64_t> the_last_grid_frame;
GPUMemory<__half> the_params;
GPUMemory<__half> the_residuals;
int64_t the_frame = 0;
template< typename... Args >
std::string string_sprintf( const char* format, Args... args ) {
  int length = std::snprintf( nullptr, 0, format, args... );
  assert( length >= 0 );

  char* buf = new char[length + 1];
  std::snprintf( buf, length + 1, format, args... );

  std::string str( buf );
  delete[] buf;
  return str;
}

void Testbed::do_grid_hit(GPUMemory<uint32_t>* grid_hit) {
    const uint64_t K = 64;
    uint64_t* counter_gpu;
    CUDA_CHECK_THROW(cudaMalloc(&counter_gpu, sizeof(uint64_t) * K));
    CUDA_CHECK_THROW(cudaMemset(counter_gpu, 0, sizeof(uint64_t) * K));
    parallel_for_gpu(m_stream.get(), grid_hit->size(), [grid_hit=grid_hit->data(), counter_gpu, K] __device__ (size_t i) {
        for (uint64_t k=0;k<K;k++)
        if (grid_hit[i] > k) atomicAdd(counter_gpu + k, 1);
    });
    uint64_t counter_cpu[K];
    CUDA_CHECK_THROW(cudaMemcpyAsync(counter_cpu, counter_gpu, sizeof(uint64_t) * K, cudaMemcpyDeviceToHost, m_stream.get()));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
    CUDA_CHECK_THROW(cudaFree(counter_gpu));
    // for (uint64_t k=0;k<K;k++)
    // tlog::info() << grid_hit->data() << ' ' << counter_cpu[k] << '/' << grid_hit->size();
    tlog::info() << "total " << counter_cpu[0] << '/' << grid_hit->size();

    if (accu_grid_hit.size() != grid_hit->size()) {
        accu_grid_hit.resize(grid_hit->size());
        accu_grid_hit.memset(0);
    }
    if (last_grid_hit.size() != grid_hit->size()) {
        last_grid_hit.resize(grid_hit->size());
        last_grid_hit.memset(0);
    }
    CUDA_CHECK_THROW(cudaMalloc(&counter_gpu, sizeof(uint64_t) * 2));
    CUDA_CHECK_THROW(cudaMemset(counter_gpu, 0, sizeof(uint64_t) * 2));
    uint64_t* accu_counter_gpu = counter_gpu;
    uint64_t* last_counter_gpu = counter_gpu + 1;
    parallel_for_gpu(m_stream.get(), grid_hit->size(), [grid_hit=grid_hit->data(), last_grid_hit=last_grid_hit.data(), accu_grid_hit=accu_grid_hit.data(), accu_counter_gpu, last_counter_gpu] __device__ (size_t i) {
        if (grid_hit[i] > 0 && !last_grid_hit[i]) atomicAdd(last_counter_gpu, 1);
        if (grid_hit[i] > 0 && !accu_grid_hit[i]) atomicAdd(accu_counter_gpu, 1);
    });
    CUDA_CHECK_THROW(cudaMemcpyAsync(counter_cpu, counter_gpu, sizeof(uint64_t) * 2, cudaMemcpyDeviceToHost, m_stream.get()));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
    CUDA_CHECK_THROW(cudaFree(counter_gpu));
    tlog::info() << "static not overlap accu " << counter_cpu[0] << " not overlap last " << counter_cpu[1];

    if (last_grid_frame.size() != n_params() || this_grid_frame.size() != n_params() || current_residual.size() != n_params()) return;
    if (the_params.size() != n_params()) the_params.resize(n_params()); the_params.memset(0);
    if (the_residuals.size() != n_params()) the_residuals.resize(n_params()); the_residuals.memset(0);
    if (the_last_grid_frame.size() != grid_hit->size()) {
        the_last_grid_frame.resize(grid_hit->size());
        the_last_grid_frame.memset(-128);
    }
    size_t offset = n_params() - grid_hit->size();
    CUDA_CHECK_THROW(cudaMalloc(&counter_gpu, sizeof(uint64_t) * 3));
    CUDA_CHECK_THROW(cudaMemset(counter_gpu, 0, sizeof(uint64_t) * 3));
    uint64_t* inter_counter_gpu = counter_gpu;
    uint64_t* intra_counter_gpu = counter_gpu + 1;
	uint64_t* equal_counter_gpu = counter_gpu + 2;
    parallel_for_gpu(m_stream.get(), grid_hit->size(),
    [
        grid_hit=grid_hit->data(),
        last_grid_frame=last_grid_frame.data() + offset,
        this_grid_frame=this_grid_frame.data() + offset,
        current_residual=current_residual.data() + offset,
        the_last_grid_frame=the_last_grid_frame.data(),
        the_params=the_params.data() + offset,
        the_residuals=the_residuals.data() + offset,
        inter_counter_gpu, intra_counter_gpu, equal_counter_gpu
    ] __device__ (size_t i) {
        if (grid_hit[i] <= 0) return;
        if (the_last_grid_frame[i] == this_grid_frame[i]) {
            atomicAdd(equal_counter_gpu, 1);
        }
        else if (the_last_grid_frame[i] == last_grid_frame[i]) {
            atomicAdd(inter_counter_gpu, 1);
            the_residuals[i] = current_residual[i];
        }
		else {
            atomicAdd(intra_counter_gpu, 1);
            the_params[i] = this_grid_frame[i];
        }
        the_last_grid_frame[i] = this_grid_frame[i];
    });
    uint64_t int_counter_cpu[3];
    CUDA_CHECK_THROW(cudaMemcpyAsync(int_counter_cpu, counter_gpu, sizeof(uint64_t) * 3, cudaMemcpyDeviceToHost, m_stream.get()));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
    CUDA_CHECK_THROW(cudaFree(counter_gpu));
    tlog::info() << "dynamic inter " << int_counter_cpu[0] << " intra " << int_counter_cpu[1] << " equal " << int_counter_cpu[2];

    parallel_for_gpu(m_stream.get(), grid_hit->size(), [grid_hit=grid_hit->data(), last_grid_hit=last_grid_hit.data(), accu_grid_hit=accu_grid_hit.data()] __device__ (size_t i) {
        last_grid_hit[i] = grid_hit[i] > 0;
        accu_grid_hit[i] = grid_hit[i] > 0 || accu_grid_hit[i];
    });

    CUDA_CHECK_THROW(cudaMalloc(&counter_gpu, sizeof(uint64_t) * 2));
    CUDA_CHECK_THROW(cudaMemset(counter_gpu, 0, sizeof(uint64_t) * 2));
    inter_counter_gpu = counter_gpu;
    intra_counter_gpu = counter_gpu + 1;
    parallel_for_gpu(m_stream.get(), grid_hit->size(), 
    [
        grid_hit=grid_hit->data(),
        residual=current_residual.data() + offset,
        the_params=the_params.data() + offset,
        the_residuals=the_residuals.data() + offset,
        inter_counter_gpu, intra_counter_gpu, K
    ] __device__ (size_t i) {
        if ((float)the_params[i] != 0) atomicAdd(intra_counter_gpu, 1);
        if ((float)the_residuals[i] != 0) atomicAdd(inter_counter_gpu, 1);
    });
    CUDA_CHECK_THROW(cudaMemcpyAsync(int_counter_cpu, counter_gpu, sizeof(uint64_t) * 2, cudaMemcpyDeviceToHost, m_stream.get()));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
    CUDA_CHECK_THROW(cudaFree(counter_gpu));
    tlog::info() << "nonzero inter " << int_counter_cpu[0] << " intra " << int_counter_cpu[1];

    json data;
    data["intra"] = the_params;
    data["inter"] = the_residuals;
    fs::path path = native_string(string_sprintf(grid_hit_path.c_str(), the_frame));
    fs::create_directories(path.parent_path());
    std::ofstream f{path.str(), std::ios::out | std::ios::binary};
    zstr::ostream zf{f, zstr::default_buff_size, Z_BEST_COMPRESSION};
    json::to_bson(data, zf);
    the_frame++;
}

}

