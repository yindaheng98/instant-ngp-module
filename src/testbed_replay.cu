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

#include <thrust/partition.h>

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
#define MIN_RESIDUAL 0.01

namespace ngp {
GPUMemory<bool> accu_grid_hit;
GPUMemory<bool> last_grid_hit;
GPUMemory<network_precision_t> last_params;
GPUMemory<network_precision_t> inter_params;
GPUMemory<network_precision_t> intra_params;
GPUMemory<network_precision_t> residual_topk_i;
GPUMemory<network_precision_t> residual_topk_o;
unsigned int M = 10000;
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
network_precision_t topk(network_precision_t* input, int length, int k) {
    thrust::sort(thrust::device, input, input+length, thrust::greater<network_precision_t>());
    network_precision_t top;
    CUDA_CHECK_THROW(cudaMemcpy(&top, input+k, sizeof(network_precision_t), cudaMemcpyDeviceToHost));
    return top;
}

void Testbed::do_grid_hit(GPUMemory<uint32_t>* grid_hit) {
    uint64_t* counter_gpu;
    uint64_t counter_cpu[32];

    // 统计：被调用超过k次的参数数量
    CUDA_CHECK_THROW(cudaMalloc(&counter_gpu, sizeof(uint64_t)));
    CUDA_CHECK_THROW(cudaMemset(counter_gpu, 0, sizeof(uint64_t)));
    parallel_for_gpu(m_stream.get(), grid_hit->size(), [grid_hit=grid_hit->data(), counter_gpu] __device__ (size_t i) {
        if (grid_hit[i] > 0) atomicAdd(counter_gpu, 1);
    });
    CUDA_CHECK_THROW(cudaMemcpyAsync(counter_cpu, counter_gpu, sizeof(uint64_t), cudaMemcpyDeviceToHost, m_stream.get()));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
    CUDA_CHECK_THROW(cudaFree(counter_gpu));
    tlog::info() << "total " << counter_cpu[0] << '/' << grid_hit->size(); // 输出counter_cpu[0]是被调用过至少一次的参数数量

    if (accu_grid_hit.size() != grid_hit->size()) {
        accu_grid_hit.resize(grid_hit->size());
        accu_grid_hit.memset(0);
    }
    if (last_grid_hit.size() != grid_hit->size()) {
        last_grid_hit.resize(grid_hit->size());
        last_grid_hit.memset(0);
    }
    // 统计：当前视角和上一个视角有多少参数相交；当前视角和之前所有视角有多少参数相交
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

    if (last_params.size() != n_params()) { last_params.resize(n_params()); last_params.memset(0); }
    if (inter_params.size() != n_params()) inter_params.resize(n_params()); inter_params.memset(0);
    if (intra_params.size() != n_params()) intra_params.resize(n_params()); intra_params.memset(0);
    if (residual_topk_i.size() != grid_hit->size()) residual_topk_i.resize(grid_hit->size()); residual_topk_i.memset(0);
    if (residual_topk_o.size() != grid_hit->size()) residual_topk_o.resize(grid_hit->size()); residual_topk_o.memset(0);
    // 核心过程：过滤掉小残差
    // 统计：需要传完整参数的参数数量，过滤掉小残差后的残差数量和不变的参数数量
    size_t offset = n_params() - grid_hit->size();
    CUDA_CHECK_THROW(cudaMemcpy(last_params.data(), m_network->params(), sizeof(network_precision_t) * offset, cudaMemcpyDeviceToDevice)); // MLP参数不会变
    CUDA_CHECK_THROW(cudaMalloc(&counter_gpu, sizeof(uint64_t) * 3));
    CUDA_CHECK_THROW(cudaMemset(counter_gpu, 0, sizeof(uint64_t) * 3));
    uint64_t* inter_counter_gpu = counter_gpu;
    uint64_t* intra_counter_gpu = counter_gpu + 1;
	uint64_t* equal_counter_gpu = counter_gpu + 2;
    parallel_for_gpu(m_stream.get(), grid_hit->size(),
    [
        grid_hit=grid_hit->data(),
        accu_grid_hit=accu_grid_hit.data(),
        params=m_network->params() + offset,
        last_params=last_params.data() + offset,
        inter_params=inter_params.data() + offset,
        intra_params=intra_params.data() + offset,
        residual_topk_i=residual_topk_i.data(),
        inter_counter_gpu, intra_counter_gpu, equal_counter_gpu
    ] __device__ (size_t i) {
        if (grid_hit[i] <= 0) return;
        if (!accu_grid_hit[i]) {
            atomicAdd(intra_counter_gpu, 1);
            intra_params[i] = params[i];
            return;
        }
        network_precision_t residual = params[i] - last_params[i];
        if (residual > (network_precision_t)MIN_RESIDUAL || residual < -(network_precision_t)MIN_RESIDUAL) {
            residual_topk_i[atomicAdd(inter_counter_gpu, 1)] = residual;
            inter_params[i] = residual;
        }
		else {
            atomicAdd(equal_counter_gpu, 1);
        }
    });
    uint64_t int_counter_cpu[3];
    CUDA_CHECK_THROW(cudaMemcpyAsync(int_counter_cpu, counter_gpu, sizeof(uint64_t) * 3, cudaMemcpyDeviceToHost, m_stream.get()));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
    tlog::info() << "dynamic inter " << int_counter_cpu[0] << " intra " << int_counter_cpu[1] << " equal " << int_counter_cpu[2];

    // 核心过程：top k
    uint64_t inter_counter_cpu = int_counter_cpu[0];
    parallel_for_gpu(m_stream.get(), inter_counter_cpu, [input=residual_topk_i.data(), output=residual_topk_o.data()] __device__ (size_t i) {
        output[i] = (input[i]>=(network_precision_t)0)?input[i]:-input[i];
    });
    network_precision_t top = topk(residual_topk_o.data(), inter_counter_cpu, fminf(M, int_counter_cpu[0]));
    tlog::info() << "top " << fminf(M, int_counter_cpu[0]) << " = " << (float)top;

    if (inter_params.size() != n_params()) inter_params.resize(n_params()); inter_params.memset(0);
    CUDA_CHECK_THROW(cudaMemset(counter_gpu, 0, sizeof(uint64_t) * 3));
    // 核心过程：k th 残差过滤
    // 统计：需要传完整参数的参数数量，过滤掉残差后的残差数量和不变的参数数量
    parallel_for_gpu(m_stream.get(), grid_hit->size(),
    [
        grid_hit=grid_hit->data(),
        accu_grid_hit=accu_grid_hit.data(),
        params=m_network->params() + offset,
        last_params=last_params.data() + offset,
        inter_params=inter_params.data() + offset,
        intra_params=intra_params.data() + offset,
        top, inter_counter_gpu, intra_counter_gpu, equal_counter_gpu
    ] __device__ (size_t i) {
        if (grid_hit[i] <= 0) return;
        if (!accu_grid_hit[i]) {
            atomicAdd(intra_counter_gpu, 1);
            return;
        }
        network_precision_t residual = params[i] - last_params[i];
        if (residual > top || residual < -top) {
            inter_params[i] = residual;
            atomicAdd(inter_counter_gpu, 1);
        }
		else {
            atomicAdd(equal_counter_gpu, 1);
        }
    });
    CUDA_CHECK_THROW(cudaMemcpyAsync(int_counter_cpu, counter_gpu, sizeof(uint64_t) * 3, cudaMemcpyDeviceToHost, m_stream.get()));
    CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
    CUDA_CHECK_THROW(cudaFree(counter_gpu));
    tlog::info() << "filterd inter " << int_counter_cpu[0] << " intra " << int_counter_cpu[1] << " equal " << int_counter_cpu[2];

    // 核心过程：更新hit grid
    parallel_for_gpu(m_stream.get(), grid_hit->size(), [grid_hit=grid_hit->data(), last_grid_hit=last_grid_hit.data(), accu_grid_hit=accu_grid_hit.data()] __device__ (size_t i) {
        last_grid_hit[i] = grid_hit[i] > 0;
        accu_grid_hit[i] = grid_hit[i] > 0 || accu_grid_hit[i];
    });

    // 核心过程：模拟残差加
    parallel_for_gpu(m_stream.get(), grid_hit->size(),
    [
        params=m_network->params() + offset,
        last_params=last_params.data() + offset,
        inter_params=inter_params.data() + offset,
        intra_params=intra_params.data() + offset
    ] __device__ (size_t i) {
        if (intra_params[i] != (network_precision_t)0) last_params[i] = intra_params[i];
        else if (inter_params[i] != (network_precision_t)0) last_params[i] += inter_params[i];
        params[i] = last_params[i];
    });

    auto& snapshot = grid_hit_json;
    snapshot["params"] = last_params.size();
    snapshot["params_size"] = last_params;
    snapshot["density_grid_bitfield"] = m_nerf.density_grid_bitfield;
    snapshot["density_grid_bitfield_size"] = m_nerf.density_grid_bitfield.size();
    snapshot["density_grid_size"] = NERF_GRIDSIZE();
    GPUMemory<__half> density_grid_fp16(m_nerf.density_grid.size());
    parallel_for_gpu(density_grid_fp16.size(), [density_grid=m_nerf.density_grid.data(), density_grid_fp16=density_grid_fp16.data()] __device__ (size_t i) {
        density_grid_fp16[i] = (__half)density_grid[i];
    });

    fs::path save_path = native_string(string_sprintf(grid_hit_path.c_str(), the_frame));
    fs::create_directories(save_path.parent_path());
    save_grid_hit(save_path);
    the_frame++;
}

}

