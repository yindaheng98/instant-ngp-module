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

// Windows.h is evil
#undef min
#undef max
#undef near
#undef far


using namespace std::literals::chrono_literals;

namespace ngp {

void Testbed::enable_residual_regulization(network_precision_t l2_reg) {
	if (m_testbed_mode == ETestbedMode::Nerf) {
		m_nerf_network->enable_residual_regulization(m_stream.get(), (network_precision_t)l2_reg);
	}
}

void Testbed::set_mode(ETestbedMode mode) {
	if (mode == m_testbed_mode) {
		return;
	}

	// Reset mode-specific members
	m_image = {};
	m_mesh = {};
	m_nerf = {};
	m_volume = {};

	// Kill training-related things
	m_encoding = {};
	m_loss = {};
	m_network = {};
	m_nerf_network = {};
	m_optimizer = {};
	m_trainer = {};
	m_envmap = {};
	m_distortion = {};
	m_training_data_available = false;

	// Clear device-owned data that might be mode-specific
	for (auto&& device : m_devices) {
		device.clear();
	}

	// Reset paths that might be attached to the chosen mode
	m_data_path = {};

	m_testbed_mode = mode;

}

void Testbed::train_and_render(bool skip_rendering) {
	train(m_training_batch_size);

	// If we don't have a trainer, as can happen when having loaded training data or changed modes without having
	// explicitly loaded a new neural network.
	if (m_testbed_mode != ETestbedMode::None && !m_network) {
		reload_network_from_file();
		if (!m_network) {
			throw std::runtime_error{"Unable to reload neural network."};
		}
	}

	if (m_mesh.optimize_mesh) {
		optimise_mesh_step(1);
	}
}

bool Testbed::frame() {
	train_and_render(true);

	return true;
}

void Testbed::set_max_level(float maxlevel) {
	if (!m_network) return;
	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_max_level(maxlevel);
	}
}

Testbed::Testbed(ETestbedMode mode) {
	tcnn::set_log_callback([](LogSeverity severity, const std::string& msg) {
		tlog::ESeverity s = tlog::ESeverity::Info;
		switch (severity) {
			case LogSeverity::Info: s = tlog::ESeverity::Info; break;
			case LogSeverity::Debug: s = tlog::ESeverity::Debug; break;
			case LogSeverity::Warning: s = tlog::ESeverity::Warning; break;
			case LogSeverity::Error: s = tlog::ESeverity::Error; break;
			case LogSeverity::Success: s = tlog::ESeverity::Success; break;
			default: break;
		}
		tlog::log(s) << msg;
	});

	if (!(__CUDACC_VER_MAJOR__ > 10 || (__CUDACC_VER_MAJOR__ == 10 && __CUDACC_VER_MINOR__ >= 2))) {
		throw std::runtime_error{"Testbed requires CUDA 10.2 or later."};
	}

	// Reset our stream, which was allocated on the originally active device,
	// to make sure it corresponds to the now active device.
	m_stream = {};

	int active_device = cuda_device();
	int active_compute_capability = cuda_compute_capability();
	tlog::success() << fmt::format(
		"Initialized CUDA {}. Active GPU is #{}: {} [{}]",
		cuda_runtime_version_string(),
		active_device,
		cuda_device_name(),
		active_compute_capability
	);

	if (active_compute_capability < MIN_GPU_ARCH) {
		tlog::warning() << "Insufficient compute capability " << active_compute_capability << " detected.";
		tlog::warning() << "This program was compiled for >=" << MIN_GPU_ARCH << " and may thus behave unexpectedly.";
	}

	m_devices.emplace_back(active_device, true);

	// Multi-GPU is only supported in NeRF mode for now
	int n_devices = cuda_device_count();
	for (int i = 0; i < n_devices; ++i) {
		if (i == active_device) {
			continue;
		}

		if (cuda_compute_capability(i) >= MIN_GPU_ARCH) {
			m_devices.emplace_back(i, false);
		}
	}

	if (m_devices.size() > 1) {
		tlog::success() << "Detected auxiliary GPUs:";
		for (size_t i = 1; i < m_devices.size(); ++i) {
			const auto& device = m_devices[i];
			tlog::success() << "  #" << device.id() << ": " << device.name() << " [" << device.compute_capability() << "]";
		}
	}

	m_network_config = {
		{"loss", {
			{"otype", "L2"}
		}},
		{"optimizer", {
			{"otype", "Adam"},
			{"learning_rate", 1e-3},
			{"beta1", 0.9f},
			{"beta2", 0.99f},
			{"epsilon", 1e-15f},
			{"l2_reg", 1e-6f},
		}},
		{"encoding", {
			{"otype", "HashGrid"},
			{"n_levels", 16},
			{"n_features_per_level", 2},
			{"log2_hashmap_size", 19},
			{"base_resolution", 16},
		}},
		{"network", {
			{"otype", "FullyFusedMLP"},
			{"n_neurons", 64},
			{"n_layers", 2},
			{"activation", "ReLU"},
			{"output_activation", "None"},
		}},
	};

	set_mode(mode);
	set_max_level(1.f);
}

Testbed::~Testbed() {

	// If any temporary file was created, make sure it's deleted
	clear_tmp_dir();
}

void Testbed::train(uint32_t batch_size) {
	if (!m_training_data_available) {
		tlog::warning() << "No dataset. Aborting training.";
		return;
	}

	if (m_testbed_mode == ETestbedMode::None) {
		throw std::runtime_error{"Cannot train without a mode."};
	}

	set_all_devices_dirty();

	// If we don't have a trainer, as can happen when having loaded training data or changed modes without having
	// explicitly loaded a new neural network.
	if (!m_trainer) {
		reload_network_from_file();
		if (!m_trainer) {
			throw std::runtime_error{"Unable to create a neural network trainer."};
		}
	}

	if (m_testbed_mode == ETestbedMode::Nerf) {
		if (m_nerf.training.optimize_extra_dims) {
			if (m_nerf.training.dataset.n_extra_learnable_dims == 0) {
				m_nerf.training.dataset.n_extra_learnable_dims = 16;
				reset_network();
			}
		}
	}

	uint32_t n_prep_to_skip = m_testbed_mode == ETestbedMode::Nerf ? clamp(m_training_step / 16u, 1u, 16u) : 1u;
	if (m_training_step % n_prep_to_skip == 0) {
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_prep_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count() / n_prep_to_skip);
		}};

		switch (m_testbed_mode) {
			case ETestbedMode::Nerf: training_prep_nerf(batch_size, m_stream.get()); break;
			default: throw std::runtime_error{"Invalid training mode."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}

	// Find leaf optimizer and update its settings
	json* leaf_optimizer_config = &m_network_config["optimizer"];
	while (leaf_optimizer_config->contains("nested")) {
		leaf_optimizer_config = &(*leaf_optimizer_config)["nested"];
	}
	(*leaf_optimizer_config)["optimize_matrix_params"] = m_train_network;
	(*leaf_optimizer_config)["optimize_non_matrix_params"] = m_train_encoding;
	m_optimizer->update_hyperparams(m_network_config["optimizer"]);

	bool get_loss_scalar = m_training_step % 16 == 0;

	{
		auto start = std::chrono::steady_clock::now();
		ScopeGuard timing_guard{[&]() {
			m_training_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count());
		}};

		switch (m_testbed_mode) {
			case ETestbedMode::Nerf: train_nerf(batch_size, get_loss_scalar, m_stream.get()); break;
			default: throw std::runtime_error{"Invalid training mode."};
		}

		CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
	}
}

}

