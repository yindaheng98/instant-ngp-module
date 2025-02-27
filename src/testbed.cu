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

extern std::atomic<size_t> g_total_n_bytes_allocated;

void Testbed::set_mode() {

	// Reset mode-specific members
	m_mesh = {};
	m_nerf = {};

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

	// Set various defaults depending on mode
	if (m_devices.size() > 1) {
		m_use_aux_devices = true;
	}

	if (m_dlss_provider && m_aperture_size == 0.0f) {
		m_dlss = true;
	}

	reset_camera();

#ifdef NGP_GUI
	update_vr_performance_settings();
#endif
}

void Testbed::reset_accumulation(bool due_to_camera_movement, bool immediate_redraw) {
	if (immediate_redraw) {
		redraw_next_frame();
	}

	if (!due_to_camera_movement || !reprojection_available()) {
		m_windowless_render_surface.reset_accumulation();
		for (auto& view : m_views) {
			view.render_buffer->reset_accumulation();
		}
	}
}

void Testbed::set_visualized_dim(int dim) {
	m_visualized_dimension = dim;
	reset_accumulation();
}

void Testbed::translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down) {
	vec3 movement = rot * rel;
	if (!allow_up_down) {
		movement -= dot(movement, m_up_dir) * m_up_dir;
	}

	m_camera[3] += movement;
	reset_accumulation(true);
}

void Testbed::set_nerf_camera_matrix(const mat4x3& cam) {
	m_camera = m_nerf.training.dataset.nerf_matrix_to_ngp(cam);
}

vec3 Testbed::look_at() const {
	return view_pos() + view_dir() * m_scale;
}

void Testbed::set_look_at(const vec3& pos) {
	m_camera[3] += pos - look_at();
}

void Testbed::set_scale(float scale) {
	auto prev_look_at = look_at();
	m_camera[3] = (view_pos() - prev_look_at) * (scale / m_scale) + prev_look_at;
	m_scale = scale;
}

void Testbed::set_view_dir(const vec3& dir) {
	auto old_look_at = look_at();
	m_camera[0] = normalize(cross(dir, m_up_dir));
	m_camera[1] = normalize(cross(dir, m_camera[0]));
	m_camera[2] = normalize(dir);
	set_look_at(old_look_at);
}

void Testbed::first_training_view() {
	m_nerf.training.view = 0;
	set_camera_to_training_view(m_nerf.training.view);
}

void Testbed::last_training_view() {
	m_nerf.training.view = m_nerf.training.dataset.n_images-1;
	set_camera_to_training_view(m_nerf.training.view);
}

void Testbed::previous_training_view() {
	if (m_nerf.training.view != 0) {
		m_nerf.training.view -= 1;
	}

	set_camera_to_training_view(m_nerf.training.view);
}

void Testbed::next_training_view() {
	if (m_nerf.training.view != m_nerf.training.dataset.n_images-1) {
		m_nerf.training.view += 1;
	}

	set_camera_to_training_view(m_nerf.training.view);
}

void Testbed::set_camera_to_training_view(int trainview) {
	auto old_look_at = look_at();
	m_camera = m_smoothed_camera = get_xform_given_rolling_shutter(m_nerf.training.transforms[trainview], m_nerf.training.dataset.metadata[trainview].rolling_shutter, vec2{0.5f, 0.5f}, 0.0f);
	m_relative_focal_length = m_nerf.training.dataset.metadata[trainview].focal_length / (float)m_nerf.training.dataset.metadata[trainview].resolution[m_fov_axis];
	m_scale = std::max(dot(old_look_at - view_pos(), view_dir()), 0.1f);
	m_nerf.render_with_lens_distortion = true;
	m_nerf.render_lens = m_nerf.training.dataset.metadata[trainview].lens;
	if (!supports_dlss(m_nerf.render_lens.mode)) {
		m_dlss = false;
	}

	m_screen_center = vec2(1.0f) - m_nerf.training.dataset.metadata[trainview].principal_point;
	m_nerf.training.view = trainview;

	reset_accumulation(true);
}

void Testbed::reset_camera() {
	m_fov_axis = 1;
	m_zoom = 1.0f;
	m_screen_center = vec2(0.5f);

	set_fov(50.625f);
	m_scale = 1.5f;

	m_camera = transpose(mat3x4{
		1.0f, 0.0f, 0.0f, 0.5f,
		0.0f, -1.0f, 0.0f, 0.5f,
		0.0f, 0.0f, -1.0f, 0.5f
	});

	m_camera[3] -= m_scale * view_dir();

	m_smoothed_camera = m_camera;
	m_sun_dir = normalize(vec3(1.0f));

	reset_accumulation();
}

void Testbed::set_train(bool mtrain) {
	if (m_train && !mtrain && m_max_level_rand_training) {
		set_max_level(1.f);
	}
	m_train = mtrain;
}

void Testbed::compute_and_save_marching_cubes_mesh(const fs::path& filename, ivec3 res3d , BoundingBox aabb, float thresh, bool unwrap_it) {
	mat3 render_aabb_to_local = mat3::identity();
	if (aabb.is_empty()) {
		aabb = m_render_aabb;
		render_aabb_to_local = m_render_aabb_to_local;
	}
	marching_cubes(res3d, aabb, render_aabb_to_local, thresh);
	save_mesh(m_mesh.verts, m_mesh.vert_normals, m_mesh.vert_colors, m_mesh.indices, filename, unwrap_it, m_nerf.training.dataset.scale, m_nerf.training.dataset.offset);
}

ivec3 Testbed::compute_and_save_png_slices(const fs::path& filename, int res, BoundingBox aabb, float thresh, float density_range, bool flip_y_and_z_axes) {
	mat3 render_aabb_to_local = mat3::identity();
	if (aabb.is_empty()) {
		aabb = m_render_aabb;
		render_aabb_to_local = m_render_aabb_to_local;
	}
	if (thresh == std::numeric_limits<float>::max()) {
		thresh = m_mesh.thresh;
	}
	float range = density_range;
	auto res3d = get_marching_cubes_res(res, aabb);

	std::string fname = fmt::format(".density_slices_{}x{}x{}.png", res3d.x, res3d.y, res3d.z);
	GPUMemory<float> density = get_density_on_grid(res3d, aabb, render_aabb_to_local);
	save_density_grid_to_png(density, filename.str() + fname, res3d, thresh, flip_y_and_z_axes, range);
	return res3d;
}

inline float linear_to_db(float x) {
	return -10.f*logf(x)/logf(10.f);
}

mat4x3 Testbed::crop_box(bool nerf_space) const {
	vec3 cen = transpose(m_render_aabb_to_local) * m_render_aabb.center();
	vec3 radius = m_render_aabb.diag() * 0.5f;
	vec3 x = row(m_render_aabb_to_local, 0) * radius.x;
	vec3 y = row(m_render_aabb_to_local, 1) * radius.y;
	vec3 z = row(m_render_aabb_to_local, 2) * radius.z;
	mat4x3 rv;
	rv[0] = x;
	rv[1] = y;
	rv[2] = z;
	rv[3] = cen;
	if (nerf_space) {
		rv = m_nerf.training.dataset.ngp_matrix_to_nerf(rv, true);
	}
	return rv;
}

void Testbed::set_crop_box(mat4x3 m, bool nerf_space) {
	if (nerf_space) {
		m = m_nerf.training.dataset.nerf_matrix_to_ngp(m, true);
	}

	vec3 radius{length(m[0]), length(m[1]), length(m[2])};
	vec3 cen(m[3]);

	m_render_aabb_to_local = row(m_render_aabb_to_local, 0, m[0] / radius.x);
	m_render_aabb_to_local = row(m_render_aabb_to_local, 1, m[1] / radius.y);
	m_render_aabb_to_local = row(m_render_aabb_to_local, 2, m[2] / radius.z);
	cen = m_render_aabb_to_local * cen;
	m_render_aabb.min = cen - radius;
	m_render_aabb.max = cen + radius;
}

std::vector<vec3> Testbed::crop_box_corners(bool nerf_space) const {
	mat4x3 m = crop_box(nerf_space);
	std::vector<vec3> rv(8);
	for (int i = 0; i < 8; ++i) {
		rv[i] = m * vec4{(i & 1) ? 1.f : -1.f, (i & 2) ? 1.f : -1.f, (i & 4) ? 1.f : -1.f, 1.f};
		/* debug print out corners to check math is all lined up */
		if (0) {
			tlog::info() << rv[i].x << "," << rv[i].y << "," << rv[i].z << " [" << i << "]";
			vec3 mn = m_render_aabb.min;
			vec3 mx = m_render_aabb.max;
			mat3 m = transpose(m_render_aabb_to_local);
			vec3 a;

			a.x = (i&1) ? mx.x : mn.x;
			a.y = (i&2) ? mx.y : mn.y;
			a.z = (i&4) ? mx.z : mn.z;
			a = m * a;
			if (nerf_space) {
				a = m_nerf.training.dataset.ngp_position_to_nerf(a);
			}
			tlog::info() << a.x << "," << a.y << "," << a.z << " [" << i << "]";
		}
	}
	return rv;
}

#ifdef NGP_GUI
bool imgui_colored_button(const char *name, float hue) {
	ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(hue, 0.6f, 0.6f));
	ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(hue, 0.7f, 0.7f));
	ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(hue, 0.8f, 0.8f));
	bool rv = ImGui::Button(name);
	ImGui::PopStyleColor(3);
	return rv;
}

void Testbed::imgui() {
	// If a GUI interaction causes an error, write that error to the following string and call
	//   ImGui::OpenPopup("Error");
	static std::string imgui_error_string = "";

	m_picture_in_picture_res = 0;

	bool train_extra_dims = m_nerf.training.dataset.n_extra_learnable_dims > 0;
	if (train_extra_dims && m_nerf.training.n_images_for_training > 0) {
		if (ImGui::Begin("Latent space 2D embedding")) {
			ImVec2 size = ImGui::GetContentRegionAvail();
			if (size.x<100.f) size.x = 100.f;
			if (size.y<100.f) size.y = 100.f;
			ImGui::InvisibleButton("##empty", size);

			static std::vector<float> X;
			static std::vector<float> Y;
			uint32_t n_extra_dims = m_nerf.training.dataset.n_extra_dims();
			std::vector<float> mean(n_extra_dims, 0.0f);
			uint32_t n = m_nerf.training.n_images_for_training;
			float norm = 1.0f / n;
			for (uint32_t i = 0; i < n; ++i) {
				for (uint32_t j = 0; j < n_extra_dims; ++j) {
					mean[j] += m_nerf.training.extra_dims_opt[i].variable()[j] * norm;
				}
			}

			std::vector<float> cov(n_extra_dims * n_extra_dims, 0.0f);
			float scale = 0.001f;	// compute scale
			for (uint32_t i = 0; i < n; ++i) {
				std::vector<float> v = m_nerf.training.extra_dims_opt[i].variable();
				for (uint32_t j = 0; j < n_extra_dims; ++j) {
					v[j] -= mean[j];
				}

				for (uint32_t m = 0; m < n_extra_dims; ++m) {
					for (uint32_t n = 0; n < n_extra_dims; ++n) {
						cov[m + n * n_extra_dims] += v[m] * v[n];
					}
				}
			}

			scale = 3.0f; // fixed scale
			if (X.size() != mean.size()) { X = std::vector<float>(mean.size(), 0.0f); }
			if (Y.size() != mean.size()) { Y = std::vector<float>(mean.size(), 0.0f); }

			// power iteration to get X and Y. TODO: modified gauss siedel to orthonormalize X and Y jointly?
			// X = (X * cov); if (X.norm() == 0.f) { X.setZero(); X.x() = 1.f; } else X.normalize();
			// Y = (Y * cov); Y -= Y.dot(X) * X; if (Y.norm() == 0.f) { Y.setZero(); Y.y() = 1.f; } else Y.normalize();

			std::vector<float> tmp(mean.size(), 0.0f);
			norm = 0.0f;
			for (uint32_t m = 0; m < n_extra_dims; ++m) {
				tmp[m] = 0.0f;
				for (uint32_t n = 0; n < n_extra_dims; ++n) {
					tmp[m] += X[n] * cov[m + n * n_extra_dims];
				}
				norm += tmp[m] * tmp[m];
			}
			norm = std::sqrt(norm);
			for (uint32_t m = 0; m < n_extra_dims; ++m) {
				if (norm == 0.0f) {
					X[m] = m == 0 ? 1.0f : 0.0f;
					continue;
				}
				X[m] = tmp[m] / norm;
			}

			float y_dot_x = 0.0f;
			for (uint32_t m = 0; m < n_extra_dims; ++m) {
				tmp[m] = 0.0f;
				for (uint32_t n = 0; n < n_extra_dims; ++n) {
					tmp[m] += Y[n] * cov[m + n * n_extra_dims];
				}
				y_dot_x += tmp[m] * X[m];
			}

			norm = 0.0f;
			for (uint32_t m = 0; m < n_extra_dims; ++m) {
				Y[m] = tmp[m] - y_dot_x * X[m];
				norm += Y[m] * Y[m];
			}
			norm = std::sqrt(norm);
			for (uint32_t m = 0; m < n_extra_dims; ++m) {
				if (norm == 0.0f) {
					Y[m] = m == 1 ? 1.0f : 0.0f;
					continue;
				}
				Y[m] = Y[m] / norm;
			}

			const ImVec2 p0 = ImGui::GetItemRectMin();
			const ImVec2 p1 = ImGui::GetItemRectMax();
			ImDrawList* draw_list = ImGui::GetWindowDrawList();
			draw_list->AddRectFilled(p0, p1, IM_COL32(0, 0, 0, 255));
			draw_list->AddRect(p0, p1, IM_COL32(255, 255, 255, 128));
			ImGui::PushClipRect(p0, p1, true);
			vec2 mouse = {ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y};
			for (uint32_t i = 0; i < n; ++i) {
				vec2 p = vec2(0.0f);

				std::vector<float> v = m_nerf.training.extra_dims_opt[i].variable();
				for (uint32_t j = 0; j < n_extra_dims; ++j) {
					p.x += (v[j] - mean[j]) * X[j] / scale;
					p.y += (v[j] - mean[j]) * Y[j] / scale;
				}

				p = ((p * vec2{p1.x - p0.x - 20.f, p1.y - p0.y - 20.f}) + vec2{p0.x + p1.x, p0.y + p1.y}) * 0.5f;
				if (distance(p, mouse) < 10.0f) {
					ImGui::SetTooltip("%d", i);
				}

				float theta = i * PI() * 2.0f / n;
				ImColor col(sinf(theta) * 0.4f + 0.5f, sinf(theta + PI() * 2.0f / 3.0f) * 0.4f + 0.5f, sinf(theta + PI() * 4.0f / 3.0f) * 0.4f + 0.5f);
				draw_list->AddCircleFilled(ImVec2{p.x, p.y}, 10.f, col);
				draw_list->AddCircle(ImVec2{p.x, p.y}, 10.f, IM_COL32(255, 255, 255, 64));
			}

			ImGui::PopClipRect();
		}

		ImGui::End();
	}

	ImGui::Begin("instant-ngp v" NGP_VERSION);

	size_t n_bytes = tcnn::total_n_bytes_allocated() + g_total_n_bytes_allocated;
	if (m_dlss_provider) {
		n_bytes += m_dlss_provider->allocated_bytes();
	}

	ImGui::Text("Frame: %.2f ms (%.1f FPS); Mem: %s", m_frame_ms.ema_val(), 1000.0f / m_frame_ms.ema_val(), bytes_to_string(n_bytes).c_str());
	bool accum_reset = false;

	if (!m_training_data_available) { ImGui::BeginDisabled(); }

	if (ImGui::CollapsingHeader("Training", m_training_data_available ? ImGuiTreeNodeFlags_DefaultOpen : 0)) {
		if (imgui_colored_button(m_train ? "Stop training" : "Start training", 0.4)) {
			set_train(!m_train);
		}


		ImGui::SameLine();
		if (imgui_colored_button("Reset training", 0.f)) {
			reload_network_from_file();
		}

		ImGui::SameLine();
		ImGui::Checkbox("encoding", &m_train_encoding);
		ImGui::SameLine();
		ImGui::Checkbox("network", &m_train_network);
		ImGui::SameLine();
		ImGui::Checkbox("rand levels", &m_max_level_rand_training);
		ImGui::Checkbox("envmap", &m_nerf.training.train_envmap);
		ImGui::SameLine();
		ImGui::Checkbox("extrinsics", &m_nerf.training.optimize_extrinsics);
		ImGui::SameLine();
		ImGui::Checkbox("distortion", &m_nerf.training.optimize_distortion);
		ImGui::SameLine();
		ImGui::Checkbox("per-image latents", &m_nerf.training.optimize_extra_dims);


		static bool export_extrinsics_in_quat_format = true;
		static bool extrinsics_have_been_optimized = false;

		if (m_nerf.training.optimize_extrinsics) {
			extrinsics_have_been_optimized = true;
		}

		if (extrinsics_have_been_optimized) {
			if (imgui_colored_button("Export extrinsics", 0.4f)) {
				m_nerf.training.export_camera_extrinsics(m_imgui.extrinsics_path, export_extrinsics_in_quat_format);
			}

			ImGui::SameLine();
			ImGui::Checkbox("as quaternions", &export_extrinsics_in_quat_format);
			ImGui::InputText("File##Extrinsics file path", m_imgui.extrinsics_path, sizeof(m_imgui.extrinsics_path));
		}

		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
		ImGui::SliderInt("Batch size", (int*)&m_training_batch_size, 1 << 12, 1 << 22, "%d", ImGuiSliderFlags_Logarithmic);
		ImGui::SameLine();
		ImGui::DragInt("Seed", (int*)&m_seed, 1.0f, 0, std::numeric_limits<int>::max());
		ImGui::PopItemWidth();

		m_training_batch_size = next_multiple(m_training_batch_size, BATCH_SIZE_GRANULARITY);

		if (m_train) {
			std::vector<std::string> timings;
			timings.emplace_back(fmt::format("Grid: {:.01f}ms", m_training_prep_ms.ema_val()));

			timings.emplace_back(fmt::format("Training: {:.01f}ms", m_training_ms.ema_val()));
			ImGui::Text("%s", join(timings, ", ").c_str());
		} else {
			ImGui::Text("Training paused");
		}

		ImGui::Text("Rays/batch: %d, Samples/ray: %.2f, Batch size: %d/%d", m_nerf.training.counters_rgb.rays_per_batch, (float)m_nerf.training.counters_rgb.measured_batch_size / (float)m_nerf.training.counters_rgb.rays_per_batch, m_nerf.training.counters_rgb.measured_batch_size, m_nerf.training.counters_rgb.measured_batch_size_before_compaction);

		float elapsed_training = std::chrono::duration<float>(std::chrono::steady_clock::now() - m_training_start_time_point).count();
		ImGui::Text("Steps: %d, Loss: %0.6f (%0.2f dB), Elapsed: %.1fs", m_training_step, m_loss_scalar.ema_val(), linear_to_db(m_loss_scalar.ema_val()), elapsed_training);
		ImGui::PlotLines("loss graph", m_loss_graph.data(), std::min(m_loss_graph_samples, m_loss_graph.size()), (m_loss_graph_samples < m_loss_graph.size()) ? 0 : (m_loss_graph_samples % m_loss_graph.size()), 0, FLT_MAX, FLT_MAX, ImVec2(0, 50.f));

		if (ImGui::TreeNode("NeRF training options")) {
			ImGui::Checkbox("Random bg color", &m_nerf.training.random_bg_color);
			ImGui::SameLine();
			ImGui::Checkbox("Snap to pixel centers", &m_nerf.training.snap_to_pixel_centers);
			ImGui::SliderFloat("Near distance", &m_nerf.training.near_distance, 0.0f, 1.0f);
			accum_reset |= ImGui::Checkbox("Linear colors", &m_nerf.training.linear_colors);
			ImGui::Combo("Loss", (int*)&m_nerf.training.loss_type, LossTypeStr);
			ImGui::Combo("Depth Loss", (int*)&m_nerf.training.depth_loss_type, LossTypeStr);
			ImGui::Combo("RGB activation", (int*)&m_nerf.rgb_activation, NerfActivationStr);
			ImGui::Combo("Density activation", (int*)&m_nerf.density_activation, NerfActivationStr);
			ImGui::SliderFloat("Cone angle", &m_nerf.cone_angle_constant, 0.0f, 1.0f/128.0f);
			ImGui::SliderFloat("Depth supervision strength", &m_nerf.training.depth_supervision_lambda, 0.f, 1.f);

			// Importance sampling options, but still related to training
			ImGui::Checkbox("Sample focal plane ~error", &m_nerf.training.sample_focal_plane_proportional_to_error);
			ImGui::SameLine();
			ImGui::Checkbox("Sample focal plane ~sharpness", &m_nerf.training.include_sharpness_in_error);
			ImGui::Checkbox("Sample image ~error", &m_nerf.training.sample_image_proportional_to_error);
			ImGui::Text("%dx%d error res w/ %d steps between updates", m_nerf.training.error_map.resolution.x, m_nerf.training.error_map.resolution.y, m_nerf.training.n_steps_between_error_map_updates);
			ImGui::Checkbox("Display error overlay", &m_nerf.training.render_error_overlay);
			if (m_nerf.training.render_error_overlay) {
				ImGui::SliderFloat("Error overlay brightness", &m_nerf.training.error_overlay_brightness, 0.f, 1.f);
			}
			ImGui::SliderFloat("Density grid decay", &m_nerf.training.density_grid_decay, 0.f, 1.f,"%.4f");
			ImGui::SliderFloat("Extrinsic L2 reg.", &m_nerf.training.extrinsic_l2_reg, 1e-8f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			ImGui::SliderFloat("Intrinsic L2 reg.", &m_nerf.training.intrinsic_l2_reg, 1e-8f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			ImGui::SliderFloat("Exposure L2 reg.", &m_nerf.training.exposure_l2_reg, 1e-8f, 0.1f, "%.6f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			ImGui::TreePop();
		}
	}

	if (!m_training_data_available) { ImGui::EndDisabled(); }

	if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
		if (!m_hmd) {
			if (ImGui::Button("Connect to VR/AR headset")) {
				try {
					init_vr();
				} catch (const std::runtime_error& e) {
					imgui_error_string = e.what();
					ImGui::OpenPopup("Error");
				}
			}
		} else {
			if (ImGui::Button("Disconnect from VR/AR headset")) {
				m_hmd.reset();
				update_vr_performance_settings();
			} else if (ImGui::TreeNodeEx("VR/AR settings", ImGuiTreeNodeFlags_DefaultOpen)) {
				static int blend_mode_idx = 0;
				const auto& supported_blend_modes = m_hmd->supported_environment_blend_modes();
				if (supported_blend_modes.size() > 1) {
					if (ImGui::Combo("Environment", &blend_mode_idx, m_hmd->supported_environment_blend_modes_imgui_string())) {
						auto b = m_hmd->supported_environment_blend_modes().at(blend_mode_idx);
						m_hmd->set_environment_blend_mode(b);
						update_vr_performance_settings();
					}
				}

				if (m_devices.size() > 1) {
					ImGui::Checkbox("Multi-GPU rendering (one per eye)", &m_use_aux_devices);
				}

				accum_reset |= ImGui::Checkbox("Depth-based reprojection", &m_vr_use_depth_reproject);
				if (ImGui::Checkbox("Mask hidden display areas", &m_vr_use_hidden_area_mask)) {
					accum_reset = true;
					set_all_devices_dirty();
				}
				accum_reset |= ImGui::Checkbox("Foveated rendering", &m_foveated_rendering) && !m_dlss;
				if (m_foveated_rendering) {
					ImGui::SameLine();
					ImGui::Text(": %.01fx", m_foveated_rendering_scaling);

					if (ImGui::TreeNodeEx("Foveated rendering settings")) {
						accum_reset |= ImGui::Checkbox("Dynamic", &m_dynamic_foveated_rendering) && !m_dlss;
						ImGui::SameLine();
						accum_reset |= ImGui::Checkbox("Visualize", &m_foveated_rendering_visualize) && !m_dlss;

						if (m_dynamic_foveated_rendering) {
							accum_reset |= ImGui::SliderFloat("Maximum scaling", &m_foveated_rendering_max_scaling, 1.0f, 16.0f, "%.01f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat) && !m_dlss;
						} else {
							accum_reset |= ImGui::SliderFloat("Scaling", &m_foveated_rendering_scaling, 1.0f, 16.0f, "%.01f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat) && !m_dlss;
						}

						accum_reset |= ImGui::SliderFloat("Fovea diameter", &m_foveated_rendering_full_res_diameter, 0.1f, 0.9f) && !m_dlss;
					}
				}

				ImGui::TreePop();
			}
		}

		ImGui::Checkbox("Render", &m_render);
		ImGui::SameLine();

		const auto& render_buffer = m_views.front().render_buffer;
		std::string spp_string = m_dlss ? std::string{""} : fmt::format("({} spp)", std::max(render_buffer->spp(), 1u));
		ImGui::Text(": %.01fms for %dx%d %s", m_render_ms.ema_val(), render_buffer->in_resolution().x, render_buffer->in_resolution().y, spp_string.c_str());

		ImGui::SameLine();
		if (ImGui::Checkbox("VSync", &m_vsync)) {
			glfwSwapInterval(m_vsync ? 1 : 0);
		}


		if (!m_dlss_provider) { ImGui::BeginDisabled(); }
		accum_reset |= ImGui::Checkbox("DLSS", &m_dlss);

		if (render_buffer->dlss()) {
			ImGui::SameLine();
			ImGui::Text("(%s)", DlssQualityStrArray[(int)render_buffer->dlss()->quality()]);
			ImGui::SameLine();
			ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
			ImGui::SliderFloat("Sharpening", &m_dlss_sharpening, 0.0f, 1.0f, "%.02f");
			ImGui::PopItemWidth();
		}

		if (!m_dlss_provider) {
			ImGui::SameLine();
#ifdef NGP_VULKAN
			ImGui::Text("(unsupported on this system)");
#else
			ImGui::Text("(Vulkan was missing at compilation time)");
#endif
			ImGui::EndDisabled();
		}

		ImGui::Checkbox("Dynamic resolution", &m_dynamic_res);
		ImGui::SameLine();
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
		if (m_dynamic_res) {
			ImGui::SliderFloat("Target FPS", &m_dynamic_res_target_fps, 2.0f, 144.0f, "%.01f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
		} else {
			ImGui::SliderInt("Resolution factor", &m_fixed_res_factor, 8, 64);
		}
		ImGui::PopItemWidth();

		accum_reset |= ImGui::Combo("Render mode", (int*)&m_render_mode, RenderModeStr);
		accum_reset |= ImGui::Combo("Tonemap curve", (int*)&m_tonemap_curve, TonemapCurveStr);
		accum_reset |= ImGui::ColorEdit4("Background", &m_background_color[0]);

		if (ImGui::SliderFloat("Exposure", &m_exposure, -5.f, 5.f)) {
			set_exposure(m_exposure);
		}

		float max_diam = max(m_aabb.max - m_aabb.min);
		float render_diam = max(m_render_aabb.max - m_render_aabb.min);

		std::string transform_section_name = "World transform";
		transform_section_name += " & Crop box";

		m_edit_render_aabb = false;

		if (ImGui::TreeNode("Advanced rendering options")) {
			ImGui::SliderInt("Max spp", &m_max_spp, 0, 1024, "%d", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			accum_reset |= ImGui::Checkbox("Render transparency as checkerboard", &m_render_transparency_as_checkerboard);
			accum_reset |= ImGui::Combo("Color space", (int*)&m_color_space, ColorSpaceStr);
			accum_reset |= ImGui::Checkbox("Snap to pixel centers", &m_snap_to_pixel_centers);

			ImGui::TreePop();
		}

		if (ImGui::TreeNode("NeRF rendering options")) {
			if (m_nerf.training.dataset.has_light_dirs) {
				vec3 light_dir = normalize(m_nerf.light_dir);
				if (ImGui::TreeNodeEx("Light Dir (Polar)", ImGuiTreeNodeFlags_DefaultOpen)) {
					float phi = atan2f(m_nerf.light_dir.x, m_nerf.light_dir.z);
					float theta = asinf(m_nerf.light_dir.y);
					bool spin = ImGui::SliderFloat("Light Dir Theta", &theta, -PI() / 2.0f, PI() / 2.0f);
					spin |= ImGui::SliderFloat("Light Dir Phi", &phi, -PI(), PI());
					if (spin) {
						float sin_phi, cos_phi;
						sincosf(phi, &sin_phi, &cos_phi);
						float cos_theta=cosf(theta);
						m_nerf.light_dir = {sin_phi * cos_theta,sinf(theta),cos_phi * cos_theta};
						accum_reset = true;
					}
					ImGui::TreePop();
				}

				if (ImGui::TreeNode("Light Dir (Cartesian)")) {
					accum_reset |= ImGui::SliderFloat("Light Dir X", ((float*)(&m_nerf.light_dir)) + 0, -1.0f, 1.0f);
					accum_reset |= ImGui::SliderFloat("Light Dir Y", ((float*)(&m_nerf.light_dir)) + 1, -1.0f, 1.0f);
					accum_reset |= ImGui::SliderFloat("Light Dir Z", ((float*)(&m_nerf.light_dir)) + 2, -1.0f, 1.0f);
					ImGui::TreePop();
				}
			}

			if (m_nerf.training.dataset.n_extra_learnable_dims) {
				accum_reset |= ImGui::SliderInt("Rendering extra dims from training view", (int*)&m_nerf.rendering_extra_dims_from_training_view, -1, m_nerf.training.dataset.n_images-1);
			}

			accum_reset |= ImGui::Checkbox("Gbuffer hard edges", &m_nerf.render_gbuffer_hard_edges);

			accum_reset |= ImGui::Combo("Groundtruth render mode", (int*)&m_ground_truth_render_mode, GroundTruthRenderModeStr);
			accum_reset |= ImGui::SliderFloat("Groundtruth alpha", &m_ground_truth_alpha, 0.0f, 1.0f, "%.02f", ImGuiSliderFlags_AlwaysClamp);

			bool lens_changed = ImGui::Checkbox("Apply lens distortion", &m_nerf.render_with_lens_distortion);
			if (m_nerf.render_with_lens_distortion) {
				lens_changed |= ImGui::Combo("Lens mode", (int*)&m_nerf.render_lens.mode, LensModeStr);
				if (m_nerf.render_lens.mode == ELensMode::OpenCV) {
					accum_reset |= ImGui::InputFloat("k1", &m_nerf.render_lens.params[0], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k2", &m_nerf.render_lens.params[1], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("p1", &m_nerf.render_lens.params[2], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("p2", &m_nerf.render_lens.params[3], 0.f, 0.f, "%.5f");
				} else if (m_nerf.render_lens.mode == ELensMode::OpenCVFisheye) {
					accum_reset |= ImGui::InputFloat("k1", &m_nerf.render_lens.params[0], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k2", &m_nerf.render_lens.params[1], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k3", &m_nerf.render_lens.params[2], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("k4", &m_nerf.render_lens.params[3], 0.f, 0.f, "%.5f");
				} else if (m_nerf.render_lens.mode == ELensMode::FTheta) {
					accum_reset |= ImGui::InputFloat("width", &m_nerf.render_lens.params[5], 0.f, 0.f, "%.0f");
					accum_reset |= ImGui::InputFloat("height", &m_nerf.render_lens.params[6], 0.f, 0.f, "%.0f");
					accum_reset |= ImGui::InputFloat("f_theta p0", &m_nerf.render_lens.params[0], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p1", &m_nerf.render_lens.params[1], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p2", &m_nerf.render_lens.params[2], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p3", &m_nerf.render_lens.params[3], 0.f, 0.f, "%.5f");
					accum_reset |= ImGui::InputFloat("f_theta p4", &m_nerf.render_lens.params[4], 0.f, 0.f, "%.5f");
				}

				if (lens_changed && !supports_dlss(m_nerf.render_lens.mode)) {
					m_dlss = false;
				}
			}

			accum_reset |= lens_changed;

			accum_reset |= ImGui::SliderFloat("Min transmittance", &m_nerf.render_min_transmittance, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			ImGui::TreePop();
		}

		if (ImGui::TreeNode("Debug visualization")) {
			ImGui::Checkbox("Visualize unit cube", &m_visualize_unit_cube);
			ImGui::SameLine();
			ImGui::Checkbox("Visualize cameras", &m_nerf.visualize_cameras);
			accum_reset |= ImGui::SliderInt("Show acceleration", &m_nerf.show_accel, -1, 7);

			if (!m_single_view) { ImGui::BeginDisabled(); }
			if (ImGui::SliderInt("Visualized dimension", &m_visualized_dimension, -1, (int)network_width(m_visualized_layer)-1)) {
				set_visualized_dim(m_visualized_dimension);
			}

			if (!m_single_view) { ImGui::EndDisabled(); }

			if (ImGui::SliderInt("Visualized layer", &m_visualized_layer, 0, std::max(0, (int)network_num_forward_activations()-1))) {
				set_visualized_layer(m_visualized_layer);
			}

			if (ImGui::Checkbox("Single view", &m_single_view)) {
				set_visualized_dim(-1);
				accum_reset = true;
			}

			if (ImGui::Button("First")) {
				first_training_view();
			}
			ImGui::SameLine();
			if (ImGui::Button("Previous")) {
				previous_training_view();
			}
			ImGui::SameLine();
			if (ImGui::Button("Next")) {
				next_training_view();
			}
			ImGui::SameLine();
			if (ImGui::Button("Last")) {
				last_training_view();
			}
			ImGui::SameLine();
			ImGui::Text("%s", m_nerf.training.dataset.paths.at(m_nerf.training.view).c_str());

			if (ImGui::SliderInt("Training view", &m_nerf.training.view, 0, (int)m_nerf.training.dataset.n_images-1)) {
				set_camera_to_training_view(m_nerf.training.view);
				accum_reset = true;
			}
			ImGui::PlotLines("Training view error", m_nerf.training.error_map.pmf_img_cpu.data(), m_nerf.training.error_map.pmf_img_cpu.size(), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0, 60.f));

			if (m_nerf.training.optimize_exposure) {
				std::vector<float> exposures(m_nerf.training.dataset.n_images);
				for (uint32_t i = 0; i < m_nerf.training.dataset.n_images; ++i) {
					exposures[i] = m_nerf.training.cam_exposure[i].variable().x;
				}

				ImGui::PlotLines("Training view exposures", exposures.data(), exposures.size(), 0, nullptr, FLT_MAX, FLT_MAX, ImVec2(0, 60.f));
			}

			if (ImGui::SliderInt("Glow mode", &m_nerf.glow_mode, 0, 16)) {
				accum_reset = true;
			}

			if (m_nerf.glow_mode && ImGui::SliderFloat("Glow height", &m_nerf.glow_y_cutoff, -2.f, 3.f)) {
				accum_reset = true;
			}

			ImGui::TreePop();
		}
	}

	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Checkbox("First person controls", &m_fps_camera);
		ImGui::SameLine();
		ImGui::Checkbox("Smooth motion", &m_camera_smoothing);
		ImGui::SameLine();
		ImGui::PushItemWidth(ImGui::GetWindowWidth() * 0.3f);
		if (ImGui::SliderFloat("Aperture size", &m_aperture_size, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat)) {
			m_dlss = false;
			accum_reset = true;
		}
		ImGui::SameLine();
		accum_reset |= ImGui::SliderFloat("Focus depth", &m_slice_plane_z, -m_bounding_radius, m_bounding_radius);

		float local_fov = fov();
		if (ImGui::SliderFloat("Field of view", &local_fov, 0.0f, 120.0f)) {
			set_fov(local_fov);
			accum_reset = true;
		}
		ImGui::SameLine();
		accum_reset |= ImGui::SliderFloat("Zoom", &m_zoom, 1.f, 10.f);
		ImGui::PopItemWidth();



		if (ImGui::TreeNode("Advanced camera settings")) {
			accum_reset |= ImGui::SliderFloat2("Screen center", &m_screen_center.x, 0.f, 1.f);
			accum_reset |= ImGui::SliderFloat2("Parallax shift", &m_parallax_shift.x, -1.f, 1.f);
			accum_reset |= ImGui::SliderFloat("Slice / focus depth", &m_slice_plane_z, -m_bounding_radius, m_bounding_radius);
			accum_reset |= ImGui::SliderFloat("Render near distance", &m_render_near_distance, 0.0f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_NoRoundToFormat);
			char buf[2048];
			vec3 v = view_dir();
			vec3 p = look_at();
			vec3 s = m_sun_dir;
			vec3 u = m_up_dir;
			vec4 b = m_background_color;
			snprintf(buf, sizeof(buf),
				"testbed.background_color = [%0.3f, %0.3f, %0.3f, %0.3f]\n"
				"testbed.exposure = %0.3f\n"
				"testbed.sun_dir = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.up_dir = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.view_dir = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.look_at = [%0.3f,%0.3f,%0.3f]\n"
				"testbed.scale = %0.3f\n"
				"testbed.fov,testbed.aperture_size,testbed.slice_plane_z = %0.3f,%0.3f,%0.3f\n"
				, b.r, b.g, b.b, b.a
				, m_exposure
				, s.x, s.y, s.z
				, u.x, u.y, u.z
				, v.x, v.y, v.z
				, p.x, p.y, p.z
				, scale()
				, fov(), m_aperture_size, m_slice_plane_z
			);

			ImGui::InputTextMultiline("Params", buf, sizeof(buf));
			ImGui::TreePop();
		}
	}

	if (ImGui::CollapsingHeader("Snapshot", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::Text("Snapshot");
		ImGui::SameLine();
		if (ImGui::Button("Save")) {
			try {
				save_snapshot(m_imgui.snapshot_path, m_include_optimizer_state_in_snapshot, m_compress_snapshot);
			} catch (const std::exception& e) {
				imgui_error_string = fmt::format("Failed to save snapshot: {}", e.what());
				ImGui::OpenPopup("Error");
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Load")) {
			try {
				load_snapshot(static_cast<fs::path>(m_imgui.snapshot_path));
			} catch (const std::exception& e) {
				imgui_error_string = fmt::format("Failed to load snapshot: {}", e.what());
				ImGui::OpenPopup("Error");
			}
		}
		ImGui::SameLine();
		if (ImGui::Button("Dump parameters as images")) {
			dump_parameters_as_images(m_trainer->params(), "params");
		}

		ImGui::SameLine();
		ImGui::Checkbox("w/ optimizer state", &m_include_optimizer_state_in_snapshot);
		ImGui::InputText("File##Snapshot file path", m_imgui.snapshot_path, sizeof(m_imgui.snapshot_path));
		ImGui::SameLine();

		bool can_compress = ends_with_case_insensitive(m_imgui.snapshot_path, ".ingp");

		if (!can_compress) {
			ImGui::BeginDisabled();
			m_compress_snapshot = false;
		}
		ImGui::Checkbox("Compress", &m_compress_snapshot);
		if (!can_compress) ImGui::EndDisabled();
	}

	if (ImGui::CollapsingHeader("Export mesh / volume / slices")) {
		static bool flip_y_and_z_axes = false;
		static float density_range = 4.f;
		BoundingBox aabb = m_render_aabb;

		auto res3d = get_marching_cubes_res(m_mesh.res, aabb);

		// If we use an octree to fit the SDF only close to the surface, then marching cubes will not work (SDF not defined everywhere)

		if (imgui_colored_button("Mesh it!", 0.4f)) {
			marching_cubes(res3d, aabb, m_render_aabb_to_local, m_mesh.thresh);
			m_nerf.render_with_lens_distortion = false;
		}
		if (m_mesh.indices.size()>0) {
			ImGui::SameLine();
			if (imgui_colored_button("Clear Mesh", 0.f)) {
				m_mesh.clear();
			}
		}

		ImGui::SameLine();

		if (imgui_colored_button("Save density PNG", -0.7f)) {
			compute_and_save_png_slices(m_data_path, m_mesh.res, {}, m_mesh.thresh, density_range, flip_y_and_z_axes);
		}

		ImGui::SameLine();
		if (imgui_colored_button("Save RGBA PNG sequence", 0.2f)) {
			auto effective_view_dir = flip_y_and_z_axes ? vec3{0.0f, 1.0f, 0.0f} : vec3{0.0f, 0.0f, 1.0f};
			// Depth of 0.01f is arbitrarily chosen to produce a visually interpretable range of alpha values.
			// Alternatively, if the true transparency of a given voxel is desired, one could use the voxel size,
			// the voxel diagonal, or some form of expected ray length through the voxel, given random directions.
			GPUMemory<vec4> rgba = get_rgba_on_grid(res3d, effective_view_dir, true, 0.01f);
			auto dir = m_data_path.is_directory() || m_data_path.empty() ? (m_data_path / "rgba_slices") : (m_data_path.parent_path() / fmt::format("{}_rgba_slices", m_data_path.filename()));
			if (!dir.exists()) {
				fs::create_directory(dir);
			}

			save_rgba_grid_to_png_sequence(rgba, dir, res3d, flip_y_and_z_axes);
		}
		if (imgui_colored_button("Save raw volumes", 0.4f)) {
			auto effective_view_dir = flip_y_and_z_axes ? vec3{0.0f, 1.0f, 0.0f} : vec3{0.0f, 0.0f, 1.0f};
			auto old_local = m_render_aabb_to_local;
			auto old_aabb = m_render_aabb;
			m_render_aabb_to_local = mat3::identity();
			auto dir = m_data_path.is_directory() || m_data_path.empty() ? (m_data_path / "volume_raw") : (m_data_path.parent_path() / fmt::format("{}_volume_raw", m_data_path.filename()));
			if (!dir.exists()) {
				fs::create_directory(dir);
			}

			for (int cascade = 0; (1<<cascade)<= m_aabb.diag().x+0.5f; ++cascade) {
				float radius = (1<<cascade) * 0.5f;
				m_render_aabb = BoundingBox(vec3(0.5f-radius), vec3(0.5f+radius));
				// Dump raw density values that the user can then convert to alpha as they please.
				GPUMemory<vec4> rgba = get_rgba_on_grid(res3d, effective_view_dir, true, 0.0f, true);
				save_rgba_grid_to_raw_file(rgba, dir, res3d, flip_y_and_z_axes, cascade);
			}
			m_render_aabb_to_local = old_local;
			m_render_aabb = old_aabb;
		}

		ImGui::SameLine();
		ImGui::Checkbox("Swap Y&Z", &flip_y_and_z_axes);
		ImGui::SliderFloat("PNG Density Range", &density_range, 0.001f, 8.f);

		ImGui::SliderInt("Res:", &m_mesh.res, 16, 2048, "%d", ImGuiSliderFlags_Logarithmic);
		ImGui::SameLine();

		ImGui::Text("%dx%dx%d", res3d.x, res3d.y, res3d.z);
		float thresh_range = 10.f;
		ImGui::SliderFloat("MC density threshold",&m_mesh.thresh, -thresh_range, thresh_range);
		ImGui::Combo("Mesh render mode", (int*)&m_mesh_render_mode, "Off\0Vertex Colors\0Vertex Normals\0\0");
		ImGui::Checkbox("Unwrap mesh", &m_mesh.unwrap);
		if (uint32_t tricount = m_mesh.indices.size()/3) {
			ImGui::InputText("##OBJFile", m_imgui.mesh_path, sizeof(m_imgui.mesh_path));
			if (ImGui::Button("Save it!")) {
				save_mesh(m_mesh.verts, m_mesh.vert_normals, m_mesh.vert_colors, m_mesh.indices, m_imgui.mesh_path, m_mesh.unwrap, m_nerf.training.dataset.scale, m_nerf.training.dataset.offset);
			}
			ImGui::SameLine();
			ImGui::Text("Mesh has %d triangles\n", tricount);
			ImGui::Checkbox("Optimize mesh", &m_mesh.optimize_mesh);
			ImGui::SliderFloat("Laplacian smoothing", &m_mesh.smooth_amount, 0.f, 2048.f);
			ImGui::SliderFloat("Density push", &m_mesh.density_amount, 0.f, 128.f);
			ImGui::SliderFloat("Inflate", &m_mesh.inflate_amount, 0.f, 128.f);
		}
	}

	if (ImGui::BeginPopupModal("Error", NULL, ImGuiWindowFlags_AlwaysAutoResize)) {
		ImGui::Text("%s", imgui_error_string.c_str());
		if (ImGui::Button("OK", ImVec2(120, 0))) {
			ImGui::CloseCurrentPopup();
		}
		ImGui::EndPopup();
	}

	if (accum_reset) {
		reset_accumulation();
	}

	if (ImGui::Button("Go to python REPL")) {
		m_want_repl = true;
	}

	ImGui::End();
}

void Testbed::visualize_nerf_cameras(ImDrawList* list, const mat4& world2proj) {
	for (int i = 0; i < m_nerf.training.n_images_for_training; ++i) {
		auto res = m_nerf.training.dataset.metadata[i].resolution;
		float aspect = float(res.x)/float(res.y);
		auto current_xform = get_xform_given_rolling_shutter(m_nerf.training.transforms[i], m_nerf.training.dataset.metadata[i].rolling_shutter, vec2{0.5f, 0.5f}, 0.0f);
		visualize_nerf_camera(list, world2proj, m_nerf.training.dataset.xforms[i].start, aspect, 0x40ffff40);
		visualize_nerf_camera(list, world2proj, m_nerf.training.dataset.xforms[i].end, aspect, 0x40ffff40);
		visualize_nerf_camera(list, world2proj, current_xform, aspect, 0x80ffffff);

		// Visualize near distance
		add_debug_line(list, world2proj, current_xform[3], current_xform[3] + current_xform[2] * m_nerf.training.near_distance, 0x20ffffff);
	}

}

void Testbed::draw_visualizations(ImDrawList* list, const mat4x3& camera_matrix) {
	mat4 view2world = camera_matrix;
	mat4 world2view = inverse(view2world);

	auto focal = calc_focal_length(ivec2(1), m_relative_focal_length, m_fov_axis, m_zoom);
	float zscale = 1.0f / focal[m_fov_axis];

	float xyscale = (float)m_window_res[m_fov_axis];
	vec2 screen_center = render_screen_center(m_screen_center);
	mat4 view2proj = transpose(mat4{
		xyscale, 0.0f,    (float)m_window_res.x*screen_center.x * zscale, 0.0f,
		0.0f,    xyscale, (float)m_window_res.y*screen_center.y * zscale, 0.0f,
		0.0f,    0.0f,    1.0f,                                           0.0f,
		0.0f,    0.0f,    zscale,                                         0.0f,
	});

	mat4 world2proj = view2proj * world2view;
	float aspect = (float)m_window_res.x / (float)m_window_res.y;

	// Visualize NeRF training poses
	if (m_nerf.visualize_cameras) {
		visualize_nerf_cameras(list, world2proj);
	}

	if (m_visualize_unit_cube) {
		visualize_cube(list, world2proj, vec3(0.f), vec3(1.f), mat3::identity());
	}

	if (m_edit_render_aabb) {
		ImGuiIO& io = ImGui::GetIO();
		// float flx = focal.x;
		float fly = focal.y;
		float zfar = m_ndc_zfar;
		float znear = m_ndc_znear;
		mat4 view2proj_guizmo = transpose(mat4{
			fly * 2.0f / aspect, 0.0f,       0.0f,                            0.0f,
			0.0f,                -fly * 2.f, 0.0f,                            0.0f,
			0.0f,                0.0f,       (zfar + znear) / (zfar - znear), -(2.0f * zfar * znear) / (zfar - znear),
			0.0f,                0.0f,       1.0f,                            0.0f,
		});

		ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

		static mat4 matrix = mat4::identity();
		static mat4 world2view_guizmo = mat4::identity();

		vec3 cen = transpose(m_render_aabb_to_local) * m_render_aabb.center();
		if (!ImGuizmo::IsUsing()) {
			// The the guizmo is being used, it handles updating its matrix on its own.
			// Outside interference can only lead to trouble.
			auto rot = transpose(m_render_aabb_to_local);
			matrix = mat4(mat4x3(rot[0], rot[1], rot[2], cen));

			// Additionally, the world2view transform must stay fixed, else the guizmo will incorrectly
			// interpret the state from past frames. Special handling is necessary here, because below
			// we emulate world translation and rotation through (inverse) camera movement.
			world2view_guizmo = world2view;
		}
	}
}

void glfw_error_callback(int error, const char* description) {
	tlog::error() << "GLFW error #" << error << ": " << description;
}

bool Testbed::keyboard_event() {
	if (ImGui::GetIO().WantCaptureKeyboard) {
		return false;
	}

	if (m_keyboard_event_callback && m_keyboard_event_callback()) {
		return false;
	}

	for (int idx = 0; idx < std::min((int)ERenderMode::NumRenderModes, 10); ++idx) {
		char c[] = { "1234567890" };
		if (ImGui::IsKeyPressed(c[idx])) {
			m_render_mode = (ERenderMode)idx;
			reset_accumulation();
		}
	}

	bool ctrl = ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Ctrl;
	bool shift = ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Shift;

	if (ImGui::IsKeyPressed('E')) {
		set_exposure(m_exposure + (shift ? -0.5f : 0.5f));
		redraw_next_frame();
	}

	if (ImGui::IsKeyPressed('R')) {
		if (shift) {
			reset_camera();
		} else {
			if (ctrl) {
				reload_training_data();
				// After reloading the training data, also reset the NN.
				// Presumably, there is no use case where the user would
				// like to hot-reload the same training data set other than
				// to slightly tweak its parameters. And to observe that
				// effect meaningfully, the NN should be trained from scratch.
			}

			reload_network_from_file();
		}
	}

	if (m_training_data_available) {
		if (ImGui::IsKeyPressed('O')) {
			m_nerf.training.render_error_overlay = !m_nerf.training.render_error_overlay;
		}

		if (ImGui::IsKeyPressed('G')) {
			m_render_ground_truth = !m_render_ground_truth;
			reset_accumulation();
		}

		if (ImGui::IsKeyPressed('T')) {
			set_train(!m_train);
		}
	}

	if (ImGui::IsKeyPressed('.')) {
		if (m_single_view) {
			if (m_visualized_dimension == network_width(m_visualized_layer)-1 && m_visualized_layer < network_num_forward_activations()-1) {
				set_visualized_layer(std::max(0, std::min((int)network_num_forward_activations()-1, m_visualized_layer+1)));
				set_visualized_dim(0);
			} else {
				set_visualized_dim(std::max(-1, std::min((int)network_width(m_visualized_layer)-1, m_visualized_dimension+1)));
			}
		} else {
			set_visualized_layer(std::max(0, std::min((int)network_num_forward_activations()-1, m_visualized_layer+1)));
		}
	}

	if (ImGui::IsKeyPressed(',')) {
		if (m_single_view) {
			if (m_visualized_dimension == 0 && m_visualized_layer > 0) {
				set_visualized_layer(std::max(0, std::min((int)network_num_forward_activations()-1, m_visualized_layer-1)));
				set_visualized_dim(network_width(m_visualized_layer)-1);
			} else {
				set_visualized_dim(std::max(-1, std::min((int)network_width(m_visualized_layer)-1, m_visualized_dimension-1)));
			}
		} else {
			set_visualized_layer(std::max(0, std::min((int)network_num_forward_activations()-1, m_visualized_layer-1)));
		}
	}

	if (ImGui::IsKeyPressed('M')) {
		m_single_view = !m_single_view;
		set_visualized_dim(-1);
		reset_accumulation();
	}

	if (ImGui::IsKeyPressed('[')) {
		if (shift) {
			first_training_view();
		} else {
			previous_training_view();
		}
	}

	if (ImGui::IsKeyPressed(']')) {
		if (shift) {
			last_training_view();
		} else {
			next_training_view();
		}
	}

	if (ImGui::IsKeyPressed('=') || ImGui::IsKeyPressed('+')) {
		if (m_fps_camera) {
			m_camera_velocity *= 1.5f;
		} else {
			set_scale(m_scale * 1.1f);
		}
	}

	if (ImGui::IsKeyPressed('-') || ImGui::IsKeyPressed('_')) {
		if (m_fps_camera) {
			m_camera_velocity /= 1.5f;
		} else {
			set_scale(m_scale / 1.1f);
		}
	}

	// WASD camera movement
	vec3 translate_vec = vec3(0.0f);
	if (ImGui::IsKeyDown('W')) {
		translate_vec.z += 1.0f;
	}

	if (ImGui::IsKeyDown('A')) {
		translate_vec.x += -1.0f;
	}

	if (ImGui::IsKeyDown('S')) {
		translate_vec.z += -1.0f;
	}

	if (ImGui::IsKeyDown('D')) {
		translate_vec.x += 1.0f;
	}

	if (ImGui::IsKeyDown(' ')) {
		translate_vec.y += -1.0f;
	}

	if (ImGui::IsKeyDown('C')) {
		translate_vec.y += 1.0f;
	}

	translate_vec *= m_camera_velocity * m_frame_ms.val() / 1000.0f;
	if (shift) {
		translate_vec *= 5.0f;
	}

	if (translate_vec != vec3(0.0f)) {
		m_fps_camera = true;

		// If VR is active, movement that isn't aligned with the current view
		// direction is _very_ jarring to the user, so make keyboard-based
		// movement aligned with the VR view, even though it is not an intended
		// movement mechanism. (Users should use controllers.)
		translate_camera(translate_vec, m_hmd && m_hmd->is_visible() ? mat3(m_views.front().camera0) : mat3(m_camera));
	}

	return false;
}

void Testbed::mouse_wheel() {
	float delta = ImGui::GetIO().MouseWheel;
	if (delta == 0) {
		return;
	}

	float scale_factor = pow(1.1f, -delta);
	set_scale(m_scale * scale_factor);

	reset_accumulation(true);
}

mat3 Testbed::rotation_from_angles(const vec2& angles) const {
	vec3 up = m_up_dir;
	vec3 side = m_camera[0];
	return rotmat(angles.x, up) * rotmat(angles.y, side);
}

void Testbed::mouse_drag() {
	vec2 rel = vec2{ImGui::GetIO().MouseDelta.x, ImGui::GetIO().MouseDelta.y} / (float)m_window_res[m_fov_axis];
	vec2 mouse = {ImGui::GetMousePos().x, ImGui::GetMousePos().y};

	vec3 side = m_camera[0];

	bool shift = ImGui::GetIO().KeyMods & ImGuiKeyModFlags_Shift;

	// Left held
	if (ImGui::GetIO().MouseDown[0]) {
		if (shift) {
			reset_accumulation();
		} else {
			float rot_sensitivity = m_fps_camera ? 0.35f : 1.0f;
			mat3 rot = rotation_from_angles(-rel * 2.0f * PI() * rot_sensitivity);

			if (m_fps_camera) {
				rot *= mat3(m_camera);
				m_camera = mat4x3(rot[0], rot[1], rot[2], m_camera[3]);
			} else {
				// Turntable
				auto old_look_at = look_at();
				set_look_at({0.0f, 0.0f, 0.0f});
				m_camera = rot * m_camera;
				set_look_at(old_look_at);
			}

			reset_accumulation(true);
		}
	}

	// Right held
	if (ImGui::GetIO().MouseDown[1]) {
		mat3 rot = rotation_from_angles(-rel * 2.0f * PI());
		if (m_render_mode == ERenderMode::Shade) {
			m_sun_dir = transpose(rot) * m_sun_dir;
		}

		m_slice_plane_z += -rel.y * m_bounding_radius;
		reset_accumulation();
	}

	// Middle pressed
	if (ImGui::GetIO().MouseClicked[2]) {
		m_drag_depth = get_depth_from_renderbuffer(*m_views.front().render_buffer, mouse / vec2(m_window_res));
	}

	// Middle held
	if (ImGui::GetIO().MouseDown[2]) {
		vec3 translation = vec3{-rel.x, -rel.y, 0.0f} / m_zoom;

		// If we have a valid depth value, scale the scene translation by it such that the
		// hovered point in 3D space stays under the cursor.
		if (m_drag_depth < 256.0f) {
			translation *= m_drag_depth / m_relative_focal_length[m_fov_axis];
		}

		translate_camera(translation, mat3(m_camera));
	}
}

bool Testbed::begin_frame() {
	if (glfwWindowShouldClose(m_glfw_window) || ImGui::IsKeyPressed(GLFW_KEY_ESCAPE) || ImGui::IsKeyPressed(GLFW_KEY_Q)) {
		destroy_window();
		return false;
	}

	{
		auto now = std::chrono::steady_clock::now();
		auto elapsed = now - m_last_frame_time_point;
		m_last_frame_time_point = now;
		m_frame_ms.update(std::chrono::duration<float, std::milli>(elapsed).count());
	}

	glfwPollEvents();
	glfwGetFramebufferSize(m_glfw_window, &m_window_res.x, &m_window_res.y);

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGlfw_NewFrame();
	ImGui::NewFrame();
	ImGuizmo::BeginFrame();

	return true;
}

void Testbed::handle_user_input() {
	if (ImGui::IsKeyPressed(GLFW_KEY_TAB) || ImGui::IsKeyPressed(GLFW_KEY_GRAVE_ACCENT)) {
		m_imgui.enabled = !m_imgui.enabled;
	}

	// Only respond to mouse inputs when not interacting with ImGui
	if (!ImGui::IsAnyItemActive() && !ImGuizmo::IsUsing() && !ImGui::GetIO().WantCaptureMouse) {
		mouse_wheel();
		mouse_drag();
	}

	if (m_render_ground_truth || m_nerf.training.render_error_overlay) {
		// find nearest training view to current camera, and set it
		int bestimage = m_nerf.find_closest_training_view(m_camera);
		m_nerf.training.view = bestimage;
		if (ImGui::GetIO().MouseReleased[0]) { // snap camera to ground truth view on mouse up
			set_camera_to_training_view(m_nerf.training.view);
			if (m_nerf.training.dataset.n_extra_dims()) {
				m_nerf.set_rendering_extra_dims_from_training_view(m_nerf.training.view);
			}
		}
	}

	keyboard_event();

	if (m_imgui.enabled) {
		imgui();
	}
}

vec3 Testbed::vr_to_world(const vec3& pos) const {
	return mat3(m_camera) * pos * m_scale + m_camera[3];
}

void Testbed::begin_vr_frame_and_handle_vr_input() {
	if (!m_hmd) {
		m_vr_frame_info = nullptr;
		return;
	}

	m_hmd->poll_events();
	if (!m_hmd->must_run_frame_loop()) {
		m_vr_frame_info = nullptr;
		return;
	}

	m_vr_frame_info = m_hmd->begin_frame();

	const auto& views = m_vr_frame_info->views;
	size_t n_views = views.size();
	size_t n_devices = m_devices.size();
	if (n_views > 0) {
		set_n_views(n_views);

		ivec2 total_size = 0;
		for (size_t i = 0; i < n_views; ++i) {
			ivec2 view_resolution = {views[i].view.subImage.imageRect.extent.width, views[i].view.subImage.imageRect.extent.height};
			total_size += view_resolution;

			m_views[i].full_resolution = view_resolution;

			// Apply the VR pose relative to the world camera transform.
			m_views[i].camera0 = mat3(m_camera) * views[i].pose;
			m_views[i].camera0[3] = vr_to_world(views[i].pose[3]);
			m_views[i].camera1 = m_views[i].camera0;

			m_views[i].visualized_dimension = m_visualized_dimension;

			const auto& xr_fov = views[i].view.fov;

			// Compute the distance on the image plane (1 unit away from the camera) that an angle of the respective FOV spans
			vec2 rel_focal_length_left_down = 0.5f * fov_to_focal_length(ivec2(1), vec2{360.0f * xr_fov.angleLeft / PI(), 360.0f * xr_fov.angleDown / PI()});
			vec2 rel_focal_length_right_up = 0.5f * fov_to_focal_length(ivec2(1), vec2{360.0f * xr_fov.angleRight / PI(), 360.0f * xr_fov.angleUp / PI()});

			// Compute total distance (for X and Y) that is spanned on the image plane.
			m_views[i].relative_focal_length = rel_focal_length_right_up - rel_focal_length_left_down;

			// Compute fraction of that distance that is spanned by the right-up part and set screen center accordingly.
			vec2 ratio = rel_focal_length_right_up / m_views[i].relative_focal_length;
			m_views[i].screen_center = { 1.0f - ratio.x, ratio.y };

			// Fix up weirdness in the rendering pipeline
			m_views[i].relative_focal_length[(m_fov_axis+1)%2] *= (float)view_resolution[(m_fov_axis+1)%2] / (float)view_resolution[m_fov_axis];
			m_views[i].render_buffer->set_hidden_area_mask(m_vr_use_hidden_area_mask ? views[i].hidden_area_mask : nullptr);

			// Render each view on a different GPU (if available)
			m_views[i].device = m_use_aux_devices ? &m_devices.at(i % m_devices.size()) : &primary_device();
		}

		// Put all the views next to each other, but at half size
		glfwSetWindowSize(m_glfw_window, total_size.x / 2, (total_size.y / 2) / n_views);

		// VR controller input
		const auto& hands = m_vr_frame_info->hands;
		m_fps_camera = true;

		// TRANSLATE BY STICK (if not pressing the stick)
		if (!hands[0].pressing) {
			vec3 translate_vec = vec3{hands[0].thumbstick.x, 0.0f, hands[0].thumbstick.y} * m_camera_velocity * m_frame_ms.val() / 1000.0f;
			if (translate_vec != vec3(0.0f)) {
				translate_camera(translate_vec, mat3(m_views.front().camera0), false);
			}
		}

		// TURN BY STICK (if not pressing the stick)
		if (!hands[1].pressing) {
			auto prev_camera = m_camera;

			// Turn around the up vector (equivalent to x-axis mouse drag) with right joystick left/right
			float sensitivity = 0.35f;
			auto rot = rotation_from_angles({-2.0f * PI() * sensitivity * hands[1].thumbstick.x * m_frame_ms.val() / 1000.0f, 0.0f}) * mat3(m_camera);
			m_camera = mat4x3(rot[0], rot[1], rot[2], m_camera[3]);

			// Translate camera such that center of rotation was about the current view
			m_camera[3] += mat3(prev_camera) * views[0].pose[3] * m_scale - mat3(m_camera) * views[0].pose[3] * m_scale;
		}

		// TRANSLATE, SCALE, AND ROTATE BY GRAB
		{
			bool both_grabbing = hands[0].grabbing && hands[1].grabbing;
			float drag_factor = both_grabbing ? 0.5f : 1.0f;

			if (both_grabbing) {
				drag_factor = 0.5f;

				vec3 prev_diff = hands[0].prev_grab_pos - hands[1].prev_grab_pos;
				vec3 diff = hands[0].grab_pos - hands[1].grab_pos;
				vec3 center = 0.5f * (hands[0].grab_pos + hands[1].grab_pos);

				vec3 center_world = vr_to_world(0.5f * (hands[0].grab_pos + hands[1].grab_pos));

				// Scale around center position of the two dragging hands. Makes the scaling feel similar to phone pinch-to-zoom
				float scale = m_scale * length(prev_diff) / length(diff);
				m_camera[3] = (view_pos() - center_world) * (scale / m_scale) + center_world;
				m_scale = scale;

				// Take rotational component and project it to the nearest rotation about the up vector.
				// We don't want to rotate the scene about any other axis.
				vec3 rot = cross(normalize(prev_diff), normalize(diff));
				float rot_radians = std::asin(dot(m_up_dir, rot));

				auto prev_camera = m_camera;
				auto rotcam = rotmat(rot_radians, m_up_dir) * mat3(m_camera);
				m_camera = mat4x3(rotcam[0], rotcam[1], rotcam[2], m_camera[3]);
				m_camera[3] += mat3(prev_camera) * center * m_scale - mat3(m_camera) * center * m_scale;
			}

			for (const auto& hand : hands) {
				if (hand.grabbing) {
					m_camera[3] -= drag_factor * mat3(m_camera) * hand.drag() * m_scale;
				}
			}
		}

		// ERASE OCCUPANCY WHEN PRESSING STICK/TRACKPAD
		for (const auto& hand : hands) {
			if (hand.pressing) {
				mark_density_grid_in_sphere_empty(vr_to_world(hand.pose[3]), m_scale * 0.05f, m_stream.get());
			}
		}
	}
}

void Testbed::SecondWindow::draw(GLuint texture) {
	if (!window)
		return;
	int display_w, display_h;
	GLFWwindow *old_context = glfwGetCurrentContext();
	glfwMakeContextCurrent(window);
	glfwGetFramebufferSize(window, &display_w, &display_h);
	glViewport(0, 0, display_w, display_h);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindVertexArray(vao);
	if (program)
		glUseProgram(program);
	glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindVertexArray(0);
	glUseProgram(0);
	glfwSwapBuffers(window);
	glfwMakeContextCurrent(old_context);
}

void Testbed::init_opengl_shaders() {
	static const char* shader_vert = R"glsl(#version 140
		out vec2 UVs;
		void main() {
			UVs = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
			gl_Position = vec4(UVs * 2.0 - 1.0, 0.0, 1.0);
		})glsl";

	static const char* shader_frag = R"glsl(#version 140
		in vec2 UVs;
		out vec4 frag_color;
		uniform sampler2D rgba_texture;
		uniform sampler2D depth_texture;

		struct FoveationWarp {
			float al, bl, cl;
			float am, bm;
			float ar, br, cr;
			float switch_left, switch_right;
			float inv_switch_left, inv_switch_right;
		};

		uniform FoveationWarp warp_x;
		uniform FoveationWarp warp_y;

		float unwarp(in FoveationWarp warp, float y) {
			y = clamp(y, 0.0, 1.0);
			if (y < warp.inv_switch_left) {
				return (sqrt(-4.0 * warp.al * warp.cl + 4.0 * warp.al * y + warp.bl * warp.bl) - warp.bl) / (2.0 * warp.al);
			} else if (y > warp.inv_switch_right) {
				return (sqrt(-4.0 * warp.ar * warp.cr + 4.0 * warp.ar * y + warp.br * warp.br) - warp.br) / (2.0 * warp.ar);
			} else {
				return (y - warp.bm) / warp.am;
			}
		}

		vec2 unwarp(in vec2 pos) {
			return vec2(unwarp(warp_x, pos.x), unwarp(warp_y, pos.y));
		}

		void main() {
			vec2 tex_coords = UVs;
			tex_coords.y = 1.0 - tex_coords.y;
			tex_coords = unwarp(tex_coords);
			frag_color = texture(rgba_texture, tex_coords.xy);
			//Uncomment the following line of code to visualize debug the depth buffer for debugging.
			// frag_color = vec4(vec3(texture(depth_texture, tex_coords.xy).r), 1.0);
			gl_FragDepth = texture(depth_texture, tex_coords.xy).r;
		})glsl";

	GLuint vert = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vert, 1, &shader_vert, NULL);
	glCompileShader(vert);
	check_shader(vert, "Blit vertex shader", false);

	GLuint frag = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(frag, 1, &shader_frag, NULL);
	glCompileShader(frag);
	check_shader(frag, "Blit fragment shader", false);

	m_blit_program = glCreateProgram();
	glAttachShader(m_blit_program, vert);
	glAttachShader(m_blit_program, frag);
	glLinkProgram(m_blit_program);
	check_shader(m_blit_program, "Blit shader program", true);

	glDeleteShader(vert);
	glDeleteShader(frag);

	glGenVertexArrays(1, &m_blit_vao);
}

void Testbed::blit_texture(const Foveation& foveation, GLint rgba_texture, GLint rgba_filter_mode, GLint depth_texture, GLint framebuffer, const ivec2& offset, const ivec2& resolution) {
	if (m_blit_program == 0) {
		return;
	}

	// Blit image to OpenXR swapchain.
	// Note that the OpenXR swapchain is 8bit while the rendering is in a float texture.
	// As some XR runtimes do not support float swapchains, we can't render into it directly.

	bool tex = glIsEnabled(GL_TEXTURE_2D);
	bool depth = glIsEnabled(GL_DEPTH_TEST);
	bool cull = glIsEnabled(GL_CULL_FACE);

	if (!tex) glEnable(GL_TEXTURE_2D);
	if (!depth) glEnable(GL_DEPTH_TEST);
	if (cull) glDisable(GL_CULL_FACE);

	glDepthFunc(GL_ALWAYS);
	glDepthMask(GL_TRUE);

	glBindVertexArray(m_blit_vao);
	glUseProgram(m_blit_program);
	glUniform1i(glGetUniformLocation(m_blit_program, "rgba_texture"), 0);
	glUniform1i(glGetUniformLocation(m_blit_program, "depth_texture"), 1);

	auto bind_warp = [&](const FoveationPiecewiseQuadratic& warp, const std::string& uniform_name) {
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".al").c_str()), warp.al);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".bl").c_str()), warp.bl);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".cl").c_str()), warp.cl);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".am").c_str()), warp.am);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".bm").c_str()), warp.bm);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".ar").c_str()), warp.ar);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".br").c_str()), warp.br);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".cr").c_str()), warp.cr);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".switch_left").c_str()), warp.switch_left);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".switch_right").c_str()), warp.switch_right);

		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".inv_switch_left").c_str()), warp.inv_switch_left);
		glUniform1f(glGetUniformLocation(m_blit_program, (uniform_name + ".inv_switch_right").c_str()), warp.inv_switch_right);
	};

	bind_warp(foveation.warp_x, "warp_x");
	bind_warp(foveation.warp_y, "warp_y");

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, depth_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, rgba_texture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, rgba_filter_mode);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, rgba_filter_mode);

	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
	glViewport(offset.x, offset.y, resolution.x, resolution.y);

	glDrawArrays(GL_TRIANGLES, 0, 3);

	glBindVertexArray(0);
	glUseProgram(0);

	glDepthFunc(GL_LESS);

	// restore old state
	if (!tex) glDisable(GL_TEXTURE_2D);
	if (!depth) glDisable(GL_DEPTH_TEST);
	if (cull) glEnable(GL_CULL_FACE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Testbed::draw_gui() {
	// Make sure all the cuda code finished its business here
	CUDA_CHECK_THROW(cudaDeviceSynchronize());

	if (!m_rgba_render_textures.empty()) {
		m_second_window.draw((GLuint)m_rgba_render_textures.front()->texture());
	}

	glfwMakeContextCurrent(m_glfw_window);
	int display_w, display_h;
	glfwGetFramebufferSize(m_glfw_window, &display_w, &display_h);
	glViewport(0, 0, display_w, display_h);
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_BLEND);
	glBlendEquationSeparate(GL_FUNC_ADD, GL_FUNC_ADD);
	glBlendFuncSeparate(GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	ivec2 extent = {(int)((float)display_w / m_n_views.x), (int)((float)display_h / m_n_views.y)};

	int i = 0;
	for (int y = 0; y < m_n_views.y; ++y) {
		for (int x = 0; x < m_n_views.x; ++x) {
			if (i >= m_views.size()) {
				break;
			}

			auto& view = m_views[i];
			ivec2 top_left{x * extent.x, display_h - (y + 1) * extent.y};
			blit_texture(m_foveated_rendering_visualize ? Foveation{} : view.foveation, m_rgba_render_textures.at(i)->texture(), m_foveated_rendering ? GL_LINEAR : GL_NEAREST, m_depth_render_textures.at(i)->texture(), 0, top_left, extent);

			++i;
		}
	}
	glFinish();
	glViewport(0, 0, display_w, display_h);


	ImDrawList* list = ImGui::GetBackgroundDrawList();
	list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);

	auto draw_mesh = [&]() {
		glClear(GL_DEPTH_BUFFER_BIT);
		ivec2 res = {display_w, display_h};
		vec2 focal_length = calc_focal_length(res, m_relative_focal_length, m_fov_axis, m_zoom);
		draw_mesh_gl(m_mesh.verts, m_mesh.vert_normals, m_mesh.vert_colors, m_mesh.indices, res, focal_length, m_smoothed_camera, render_screen_center(m_screen_center), (int)m_mesh_render_mode);
	};

	// Visualizations are only meaningful when rendering a single view
	if (m_views.size() == 1) {
		if (m_mesh.verts.size() != 0 && m_mesh.indices.size() != 0 && m_mesh_render_mode != EMeshRenderMode::Off) {
			list->AddCallback([](const ImDrawList*, const ImDrawCmd* cmd) {
				(*(decltype(draw_mesh)*)cmd->UserCallbackData)();
			}, &draw_mesh);
			list->AddCallback(ImDrawCallback_ResetRenderState, nullptr);
		}

		draw_visualizations(list, m_smoothed_camera);
	}

	if (m_render_ground_truth) {
		list->AddText(ImVec2(4.f, 4.f), 0xffffffff, "Ground Truth");
	}

	ImGui::Render();
	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glfwSwapBuffers(m_glfw_window);

	// Make sure all the OGL code finished its business here.
	// Any code outside of this function needs to be able to freely write to
	// textures without being worried about interfering with rendering.
	glFinish();
}
#endif //NGP_GUI

__global__ void to_8bit_color_kernel(
	ivec2 resolution,
	EColorSpace output_color_space,
	cudaSurfaceObject_t surface,
	uint8_t* result
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	vec4 color;
	surf2Dread((float4*)&color, surface, x * sizeof(float4), y);

	if (output_color_space == EColorSpace::Linear) {
		color.rgb() = linear_to_srgb(color.rgb());
	}

	for (uint32_t i = 0; i < 3; ++i) {
		result[(x + resolution.x * y) * 3 + i] = (uint8_t)(clamp(color[i], 0.0f, 1.0f) * 255.0f + 0.5f);
	}
}

void Testbed::train_and_render(bool skip_rendering) {
	if (m_train) {
		train(m_training_batch_size);
	}

	// If we don't have a trainer, as can happen when having loaded training data or changed modes without having
	// explicitly loaded a new neural network.
	if (!m_network) {
		reload_network_from_file();
		if (!m_network) {
			throw std::runtime_error{"Unable to reload neural network."};
		}
	}

	if (m_mesh.optimize_mesh) {
		optimise_mesh_step(1);
	}

	// Don't do any smoothing here if a camera path is being rendered. It'll take care
	// of the smoothing on its own.
	float frame_ms = m_frame_ms.val();
	apply_camera_smoothing(frame_ms);

	if (!m_render_window || !m_render || skip_rendering) {
		return;
	}

	auto start = std::chrono::steady_clock::now();
	ScopeGuard timing_guard{[&]() {
		m_render_ms.update(std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now()-start).count());
	}};

	if (frobenius_norm(m_smoothed_camera - m_camera) < 0.001f) {
		m_smoothed_camera = m_camera;
	}

#ifdef NGP_GUI
	if (m_hmd && m_hmd->is_visible()) {
		for (auto& view : m_views) {
			view.visualized_dimension = m_visualized_dimension;
		}

		m_n_views = {(int)m_views.size(), 1};

		m_nerf.render_with_lens_distortion = false;
		reset_accumulation(true);
	} else if (m_single_view) {
		set_n_views(1);
		m_n_views = {1, 1};

		auto& view = m_views.front();

		view.full_resolution = m_window_res;

		view.camera0 = m_smoothed_camera;

		// Motion blur over the fraction of time that the shutter is open. Interpolate in log-space to preserve rotations.
		view.camera1 = view.camera0;

		view.visualized_dimension = m_visualized_dimension;
		view.relative_focal_length = m_relative_focal_length;
		view.screen_center = m_screen_center;
		view.render_buffer->set_hidden_area_mask(nullptr);
		view.foveation = {};
		view.device = &primary_device();
	} else {
		int n_views = n_dimensions_to_visualize()+1;

		float d = std::sqrt((float)m_window_res.x * (float)m_window_res.y / (float)n_views);

		int nx = (int)std::ceil((float)m_window_res.x / d);
		int ny = (int)std::ceil((float)n_views / (float)nx);

		m_n_views = {nx, ny};
		ivec2 view_size = {m_window_res.x / nx, m_window_res.y / ny};

		set_n_views(n_views);

		int i = 0;
		for (int y = 0; y < ny; ++y) {
			for (int x = 0; x < nx; ++x) {
				if (i >= n_views) {
					break;
				}

				m_views[i].full_resolution = view_size;

				m_views[i].camera0 = m_views[i].camera1 = m_smoothed_camera;
				m_views[i].visualized_dimension = i-1;
				m_views[i].relative_focal_length = m_relative_focal_length;
				m_views[i].screen_center = m_screen_center;
				m_views[i].render_buffer->set_hidden_area_mask(nullptr);
				m_views[i].foveation = {};
				m_views[i].device = &primary_device();
				++i;
			}
		}
	}

	if (m_dlss) {
		m_aperture_size = 0.0f;
		if (!supports_dlss(m_nerf.render_lens.mode)) {
			m_nerf.render_with_lens_distortion = false;
		}
	}

	// Update dynamic res and DLSS
	{
		// Don't count the time being spent allocating buffers and resetting DLSS as part of the frame time.
		// Otherwise the dynamic resolution calculations for following frames will be thrown out of whack
		// and may even start oscillating.
		auto skip_start = std::chrono::steady_clock::now();
		ScopeGuard skip_timing_guard{[&]() {
			start += std::chrono::steady_clock::now() - skip_start;
		}};

		size_t n_pixels = 0, n_pixels_full_res = 0;
		for (const auto& view : m_views) {
			n_pixels += product(view.render_buffer->in_resolution());
			n_pixels_full_res += product(view.full_resolution);
		}

		float pixel_ratio = (n_pixels == 0 || (m_train && m_training_step == 0)) ? (1.0f / 256.0f) : ((float)n_pixels / (float)n_pixels_full_res);

		float last_factor = std::sqrt(pixel_ratio);
		float factor = std::sqrt(pixel_ratio / m_render_ms.val() * 1000.0f / m_dynamic_res_target_fps);
		if (!m_dynamic_res) {
			factor = 8.f / (float)m_fixed_res_factor;
		}

		factor = clamp(factor, 1.0f / 16.0f, 1.0f);

		for (auto&& view : m_views) {
			if (m_dlss) {
				view.render_buffer->enable_dlss(*m_dlss_provider, view.full_resolution);
			} else {
				view.render_buffer->disable_dlss();
			}

			ivec2 render_res = view.render_buffer->in_resolution();
			ivec2 new_render_res = clamp(ivec2(vec2(view.full_resolution) * factor), view.full_resolution / 16, view.full_resolution);

			float ratio = std::sqrt((float)product(render_res) / (float)product(new_render_res));
			if (ratio > 1.2f || ratio < 0.8f || factor == 1.0f || !m_dynamic_res) {
				render_res = new_render_res;
			}

			if (view.render_buffer->dlss()) {
				render_res = view.render_buffer->dlss()->clamp_resolution(render_res);
				view.render_buffer->dlss()->update_feature(render_res, view.render_buffer->dlss()->is_hdr(), view.render_buffer->dlss()->sharpen());
			}

			view.render_buffer->resize(render_res);

			if (m_foveated_rendering) {
				if (m_dynamic_foveated_rendering) {
					vec2 resolution_scale = vec2(render_res) / vec2(view.full_resolution);

					// Only start foveation when DLSS if off or if DLSS is asked to do more than 1.5x upscaling.
					// The reason for the 1.5x threshold is that DLSS can do up to 3x upscaling, at which point a foveation
					// factor of 2x = 3.0x/1.5x corresponds exactly to bilinear super sampling, which is helpful in
					// suppressing DLSS's artifacts.
					float foveation_begin_factor = m_dlss ? 1.5f : 1.0f;

					resolution_scale = clamp(resolution_scale * foveation_begin_factor, vec2(1.0f / m_foveated_rendering_max_scaling), vec2(1.0f));
					view.foveation = {resolution_scale, vec2(1.0f) - view.screen_center, vec2(m_foveated_rendering_full_res_diameter * 0.5f)};

					m_foveated_rendering_scaling = 2.0f / sum(resolution_scale);
				} else {
					view.foveation = {vec2(1.0f / m_foveated_rendering_scaling), vec2(1.0f) - view.screen_center, vec2(m_foveated_rendering_full_res_diameter * 0.5f)};
				}
			} else {
				view.foveation = {};
			}
		}
	}

	// Make sure all in-use auxiliary GPUs have the latest model and bitfield
	std::unordered_set<CudaDevice*> devices_in_use;
	for (auto& view : m_views) {
		if (!view.device || devices_in_use.count(view.device) != 0) {
			continue;
		}

		devices_in_use.insert(view.device);
		sync_device(*view.render_buffer, *view.device);
	}

	{
		SyncedMultiStream synced_streams{m_stream.get(), m_views.size()};

		std::vector<std::future<void>> futures(m_views.size());
		for (size_t i = 0; i < m_views.size(); ++i) {
			auto& view = m_views[i];
			futures[i] = view.device->enqueue_task([this, &view, stream=synced_streams.get(i)]() {
				auto device_guard = use_device(stream, *view.render_buffer, *view.device);
				render_frame_main(*view.device, view.camera0, view.camera1, view.screen_center, view.relative_focal_length, {0.0f, 0.0f, 0.0f, 1.0f}, view.foveation, view.visualized_dimension);
			});
		}

		for (size_t i = 0; i < m_views.size(); ++i) {
			auto& view = m_views[i];

			if (futures[i].valid()) {
				futures[i].get();
			}

			render_frame_epilogue(synced_streams.get(i), view.camera0, view.prev_camera, view.screen_center, view.relative_focal_length, view.foveation, view.prev_foveation, *view.render_buffer, true);
			view.prev_camera = view.camera0;
			view.prev_foveation = view.foveation;
		}
	}

	for (size_t i = 0; i < m_views.size(); ++i) {
		m_rgba_render_textures.at(i)->blit_from_cuda_mapping();
		m_depth_render_textures.at(i)->blit_from_cuda_mapping();
	}

	if (m_picture_in_picture_res > 0) {
		ivec2 res{(int)m_picture_in_picture_res, (int)(m_picture_in_picture_res * 9.0f / 16.0f)};
		m_pip_render_buffer->resize(res);
		if (m_pip_render_buffer->spp() < 8) {
			m_pip_render_texture->blit_from_cuda_mapping();
		}
	}
	if (should_save_image) save_image(*m_views[0].render_buffer);
#endif

	CUDA_CHECK_THROW(cudaStreamSynchronize(m_stream.get()));
}


#ifdef NGP_GUI
void Testbed::create_second_window() {
	if (m_second_window.window) {
		return;
	}
	bool frameless = false;
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, !frameless);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_CENTER_CURSOR, false);
	glfwWindowHint(GLFW_DECORATED, !frameless);
	glfwWindowHint(GLFW_SCALE_TO_MONITOR, frameless);
	glfwWindowHint(GLFW_TRANSPARENT_FRAMEBUFFER, true);
	// get the window size / coordinates
	int win_w=0,win_h=0,win_x=0,win_y=0;
	GLuint ps=0,vs=0;
	{
		win_w = 1920;
		win_h = 1080;
		win_x = 0x40000000;
		win_y = 0x40000000;
		static const char* copy_shader_vert = "\
			in vec2 vertPos_data;\n\
			out vec2 texCoords;\n\
			void main(){\n\
				gl_Position = vec4(vertPos_data.xy, 0.0, 1.0);\n\
				texCoords = (vertPos_data.xy + 1.0) * 0.5; texCoords.y=1.0-texCoords.y;\n\
			}";
		static const char* copy_shader_frag = "\
			in vec2 texCoords;\n\
			out vec4 fragColor;\n\
			uniform sampler2D screenTex;\n\
			void main(){\n\
				fragColor = texture(screenTex, texCoords.xy);\n\
			}";
		vs = compile_shader(false, copy_shader_vert);
		ps = compile_shader(true, copy_shader_frag);
	}
	m_second_window.window = glfwCreateWindow(win_w, win_h, "Fullscreen Output", NULL, m_glfw_window);
	if (win_x!=0x40000000) glfwSetWindowPos(m_second_window.window, win_x, win_y);
	glfwMakeContextCurrent(m_second_window.window);
	m_second_window.program = glCreateProgram();
	glAttachShader(m_second_window.program, vs);
	glAttachShader(m_second_window.program, ps);
	glLinkProgram(m_second_window.program);
	if (!check_shader(m_second_window.program, "shader program", true)) {
		glDeleteProgram(m_second_window.program);
		m_second_window.program = 0;
	}
	// vbo and vao
	glGenVertexArrays(1, &m_second_window.vao);
	glGenBuffers(1, &m_second_window.vbo);
	glBindVertexArray(m_second_window.vao);
	const float fsquadVerts[] = {
		-1.0f, -1.0f,
		-1.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, -1.0f,
		-1.0f, -1.0f
	};
	glBindBuffer(GL_ARRAY_BUFFER, m_second_window.vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(fsquadVerts), fsquadVerts, GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void Testbed::set_n_views(size_t n_views) {
	while (m_views.size() > n_views) {
		m_views.pop_back();
	}

	m_rgba_render_textures.resize(n_views);
	m_depth_render_textures.resize(n_views);
	while (m_views.size() < n_views) {
		size_t idx = m_views.size();
		m_rgba_render_textures[idx] = std::make_shared<GLTexture>();
		m_depth_render_textures[idx] = std::make_shared<GLTexture>();
		m_views.emplace_back(View{std::make_shared<CudaRenderBuffer>(m_rgba_render_textures[idx], m_depth_render_textures[idx])});
	}
};
#endif //NGP_GUI

void Testbed::init_window(int resw, int resh, bool hidden, bool second_window) {
#ifndef NGP_GUI
	throw std::runtime_error{"init_window failed: NGP was built without GUI support"};
#else
	m_window_res = {resw, resh};

	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit()) {
		throw std::runtime_error{"GLFW could not be initialized."};
	}

#ifdef NGP_VULKAN
	// Only try to initialize DLSS (Vulkan+NGX) if the
	// GPU is sufficiently new. Older GPUs don't support
	// DLSS, so it is preferable to not make a futile
	// attempt and emit a warning that confuses users.
	if (primary_device().compute_capability() >= 70) {
		try {
			m_dlss_provider = init_vulkan_and_ngx();
			if (m_aperture_size == 0.0f) {
				m_dlss = true;
			}
		} catch (const std::runtime_error& e) {
			tlog::warning() << "Could not initialize Vulkan and NGX. DLSS not supported. (" << e.what() << ")";
		}
	}
#endif

	glfwWindowHint(GLFW_VISIBLE, hidden ? GLFW_FALSE : GLFW_TRUE);
	std::string title = "Instant Neural Graphics Primitives";
	m_glfw_window = glfwCreateWindow(m_window_res.x, m_window_res.y, title.c_str(), NULL, NULL);
	if (m_glfw_window == NULL) {
		throw std::runtime_error{"GLFW window could not be created."};
	}
	glfwMakeContextCurrent(m_glfw_window);
#ifdef _WIN32
	if (gl3wInit()) {
		throw std::runtime_error{"GL3W could not be initialized."};
	}
#else
	glewExperimental = 1;
	if (glewInit()) {
		throw std::runtime_error{"GLEW could not be initialized."};
	}
#endif
	glfwSwapInterval(0); // Disable vsync

	GLint gl_version_minor, gl_version_major;
	glGetIntegerv(GL_MINOR_VERSION, &gl_version_minor);
	glGetIntegerv(GL_MAJOR_VERSION, &gl_version_major);

	if (gl_version_major < 3 || (gl_version_major == 3 && gl_version_minor < 1)) {
		throw std::runtime_error{fmt::format("Unsupported OpenGL version {}.{}. instant-ngp requires at least OpenGL 3.1", gl_version_major, gl_version_minor)};
	}

	tlog::success() << "Initialized OpenGL version " << glGetString(GL_VERSION);

	glfwSetWindowUserPointer(m_glfw_window, this);
	glfwSetDropCallback(m_glfw_window, [](GLFWwindow* window, int count, const char** paths) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (!testbed) {
			return;
		}

		for (int i = 0; i < count; i++) {
			testbed->load_file(paths[i]);
		}
	});

	glfwSetKeyCallback(m_glfw_window, [](GLFWwindow* window, int key, int scancode, int action, int mods) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetCursorPosCallback(m_glfw_window, [](GLFWwindow* window, double xpos, double ypos) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (
			testbed &&
			(ImGui::IsAnyItemActive() || ImGui::GetIO().WantCaptureMouse || ImGuizmo::IsUsing()) &&
			(ImGui::GetIO().MouseDown[0] || ImGui::GetIO().MouseDown[1] || ImGui::GetIO().MouseDown[2])
		) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetMouseButtonCallback(m_glfw_window, [](GLFWwindow* window, int button, int action, int mods) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetScrollCallback(m_glfw_window, [](GLFWwindow* window, double xoffset, double yoffset) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_gui_next_frame();
		}
	});

	glfwSetWindowSizeCallback(m_glfw_window, [](GLFWwindow* window, int width, int height) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_next_frame();
		}
	});

	glfwSetFramebufferSizeCallback(m_glfw_window, [](GLFWwindow* window, int width, int height) {
		Testbed* testbed = (Testbed*)glfwGetWindowUserPointer(window);
		if (testbed) {
			testbed->redraw_next_frame();
		}
	});

	float xscale, yscale;
	glfwGetWindowContentScale(m_glfw_window, &xscale, &yscale);

	// IMGUI init
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	// By default, imgui places its configuration (state of the GUI -- size of windows,
	// which regions are expanded, etc.) in ./imgui.ini relative to the working directory.
	// Instead, we would like to place imgui.ini in the directory that instant-ngp project
	// resides in.
	static std::string ini_filename;
	ini_filename = (root_dir()/"imgui.ini").str();
	io.IniFilename = ini_filename.c_str();

	// New ImGui event handling seems to make camera controls laggy if input trickling is true.
	// So disable input trickling.
	io.ConfigInputTrickleEventQueue = false;
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(m_glfw_window, true);
	ImGui_ImplOpenGL3_Init("#version 140");

	ImGui::GetStyle().ScaleAllSizes(xscale);
	ImFontConfig font_cfg;
	font_cfg.SizePixels = 13.0f * xscale;
	io.Fonts->AddFontDefault(&font_cfg);

	init_opengl_shaders();

	// Make sure there's at least one usable render texture
	m_rgba_render_textures = { std::make_shared<GLTexture>() };
	m_depth_render_textures = { std::make_shared<GLTexture>() };

	m_views.clear();
	m_views.emplace_back(View{std::make_shared<CudaRenderBuffer>(m_rgba_render_textures.front(), m_depth_render_textures.front())});
	m_views.front().full_resolution = m_window_res;
	m_views.front().render_buffer->resize(m_views.front().full_resolution);

	m_pip_render_texture = std::make_shared<GLTexture>();
	m_pip_render_buffer = std::make_unique<CudaRenderBuffer>(m_pip_render_texture);

	m_render_window = true;

	if (m_second_window.window == nullptr && second_window) {
		create_second_window();
	}
#endif // NGP_GUI
}

void Testbed::destroy_window() {
#ifndef NGP_GUI
	throw std::runtime_error{"destroy_window failed: NGP was built without GUI support"};
#else
	if (!m_render_window) {
		throw std::runtime_error{"Window must be initialized to be destroyed."};
	}

	m_hmd.reset();

	m_views.clear();
	m_rgba_render_textures.clear();
	m_depth_render_textures.clear();

	m_pip_render_buffer.reset();
	m_pip_render_texture.reset();

	m_dlss = false;
	m_dlss_provider.reset();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(m_glfw_window);
	glfwTerminate();

	m_blit_program = 0;
	m_blit_vao = 0;

	m_glfw_window = nullptr;
	m_render_window = false;
#endif //NGP_GUI
}

void Testbed::init_vr() {
#ifndef NGP_GUI
	throw std::runtime_error{"init_vr failed: NGP was built without GUI support"};
#else
	try {
		if (!m_glfw_window) {
			throw std::runtime_error{"`init_window` must be called before `init_vr`"};
		}

#if defined(XR_USE_PLATFORM_WIN32)
		m_hmd = std::make_unique<OpenXRHMD>(wglGetCurrentDC(), glfwGetWGLContext(m_glfw_window));
#elif defined(XR_USE_PLATFORM_XLIB)
		Display* xDisplay = glfwGetX11Display();
		GLXContext glxContext = glfwGetGLXContext(m_glfw_window);

		int glxFBConfigXID = 0;
		glXQueryContext(xDisplay, glxContext, GLX_FBCONFIG_ID, &glxFBConfigXID);
		int attributes[3] = { GLX_FBCONFIG_ID, glxFBConfigXID, 0 };
		int nelements = 1;
		GLXFBConfig* pglxFBConfig = glXChooseFBConfig(xDisplay, 0, attributes, &nelements);
		if (nelements != 1 || !pglxFBConfig) {
			throw std::runtime_error{"init_vr(): Couldn't obtain GLXFBConfig"};
		}

		GLXFBConfig glxFBConfig = *pglxFBConfig;

		XVisualInfo* visualInfo = glXGetVisualFromFBConfig(xDisplay, glxFBConfig);
		if (!visualInfo) {
			throw std::runtime_error{"init_vr(): Couldn't obtain XVisualInfo"};
		}

		m_hmd = std::make_unique<OpenXRHMD>(xDisplay, visualInfo->visualid, glxFBConfig, glXGetCurrentDrawable(), glxContext);
#elif defined(XR_USE_PLATFORM_WAYLAND)
		m_hmd = std::make_unique<OpenXRHMD>(glfwGetWaylandDisplay());
#endif

		// Enable aggressive optimizations to make the VR experience smooth.
		update_vr_performance_settings();

		// If multiple GPUs are available, shoot for 60 fps in VR.
		// Otherwise, it wouldn't be realistic to expect more than 30.
		m_dynamic_res_target_fps = m_devices.size() > 1 ? 60 : 30;
		m_background_color = {0.0f, 0.0f, 0.0f, 0.0f};
	} catch (const std::runtime_error& e) {
		if (std::string{e.what()}.find("XR_ERROR_FORM_FACTOR_UNAVAILABLE") != std::string::npos) {
			throw std::runtime_error{"Could not initialize VR. Ensure that SteamVR, OculusVR, or any other OpenXR-compatible runtime is running. Also set it as the active OpenXR runtime."};
		} else {
			throw std::runtime_error{fmt::format("Could not initialize VR: {}", e.what())};
		}
	}
#endif //NGP_GUI
}

void Testbed::update_vr_performance_settings() {
#ifdef NGP_GUI
	if (m_hmd) {
		auto blend_mode = m_hmd->environment_blend_mode();

		// DLSS is instrumental in getting VR to look good. Enable if possible.
		// If the environment is blended in (such as in XR/AR applications),
		// DLSS causes jittering at object sillhouettes (doesn't deal well with alpha),
		// and hence stays disabled.
		m_dlss = (blend_mode == EEnvironmentBlendMode::Opaque) && m_dlss_provider;

		// Foveated rendering is similarly vital in getting high performance without losing
		// resolution in the middle of the view.
		m_foveated_rendering = true;

		// Large minimum transmittance results in another 20-30% performance increase
		// at the detriment of some transparent edges. Not super noticeable, though.
		m_nerf.render_min_transmittance = 0.2f;

		// Many VR runtimes perform optical flow for automatic reprojection / motion smoothing.
		// This breaks down for solid-color background, sometimes leading to artifacts. Hence:
		// set background color to transparent and, in spherical_checkerboard_kernel(...),
		// blend a checkerboard. If the user desires a solid background nonetheless, they can
		// set the background color to have an alpha value of 1.0 manually via the GUI or via Python.
		m_render_transparency_as_checkerboard = (blend_mode == EEnvironmentBlendMode::Opaque);
	} else {
		m_dlss = m_dlss_provider != nullptr;
		m_foveated_rendering = false;
		m_nerf.render_min_transmittance = 0.01f;
		m_render_transparency_as_checkerboard = false;
	}
#endif //NGP_GUI
}

bool Testbed::frame() {
#ifdef NGP_GUI
	if (m_render_window) {
		if (!begin_frame()) {
			return false;
		}

		handle_user_input();
		begin_vr_frame_and_handle_vr_input();
	}
#endif

	// Render against the trained neural network. If we're training and already close to convergence,
	// we can skip rendering if the scene camera doesn't change
	uint32_t n_to_skip = m_train ? clamp(m_training_step / 16u, 15u, 255u) : 0;
	if (m_render_skip_due_to_lack_of_camera_movement_counter > n_to_skip) {
		m_render_skip_due_to_lack_of_camera_movement_counter = 0;
	}
	bool skip_rendering = m_render_skip_due_to_lack_of_camera_movement_counter++ != 0;

	if (!m_dlss && m_max_spp > 0 && !m_views.empty() && m_views.front().render_buffer->spp() >= m_max_spp) {
		skip_rendering = true;
		if (!m_train) {
			std::this_thread::sleep_for(1ms);
		}
	}

#ifdef NGP_GUI
	if (m_hmd && m_hmd->is_visible()) {
		skip_rendering = false;
	}
#endif

	if (!skip_rendering || std::chrono::steady_clock::now() - m_last_gui_draw_time_point > 50ms) {
		redraw_gui_next_frame();
	}

	try {
		while (true) {
			(*m_task_queue.tryPop())();
		}
	} catch (const SharedQueueEmptyException&) {}


	train_and_render(skip_rendering);

#ifdef NGP_GUI
	if (m_render_window) {
		if (m_gui_redraw) {
			draw_gui();
			m_gui_redraw = false;

			m_last_gui_draw_time_point = std::chrono::steady_clock::now();
		}

		ImGui::EndFrame();
	}

	if (m_hmd && m_vr_frame_info) {
		// If HMD is visible to the user, splat rendered images to the HMD
		if (m_hmd->is_visible()) {
			size_t n_views = std::min(m_views.size(), m_vr_frame_info->views.size());

			// Blit textures to the OpenXR-owned framebuffers (each corresponding to one eye)
			for (size_t i = 0; i < n_views; ++i) {
				const auto& vr_view = m_vr_frame_info->views.at(i);

				ivec2 resolution = {
					vr_view.view.subImage.imageRect.extent.width,
					vr_view.view.subImage.imageRect.extent.height,
				};

				blit_texture(m_views.at(i).foveation, m_rgba_render_textures.at(i)->texture(), GL_LINEAR, m_depth_render_textures.at(i)->texture(), vr_view.framebuffer, ivec2(0), resolution);
			}

			glFinish();
		}

		// Far and near planes are intentionally reversed, because we map depth inversely
		// to z. I.e. a window-space depth of 1 refers to the near plane and a depth of 0
		// to the far plane. This results in much better numeric precision.
		m_hmd->end_frame(m_vr_frame_info, m_ndc_zfar / m_scale, m_ndc_znear / m_scale, m_vr_use_depth_reproject);
	}
#endif

	return true;
}

bool Testbed::want_repl() {
	bool b = m_want_repl;
	m_want_repl = false;
	return b;
}

void Testbed::apply_camera_smoothing(float elapsed_ms) {
	if (m_camera_smoothing) {
		float decay = std::pow(0.02f, elapsed_ms/1000.0f);
		m_smoothed_camera = camera_log_lerp(m_smoothed_camera, m_camera, 1.0f - decay);
	} else {
		m_smoothed_camera = m_camera;
	}
}

void Testbed::update_loss_graph() {
	m_loss_graph[m_loss_graph_samples++ % m_loss_graph.size()] = std::log(m_loss_scalar.val());
}

uint32_t Testbed::n_dimensions_to_visualize() const {
	return m_network ? m_network->width(m_visualized_layer) : 0;
}

float Testbed::fov() const {
	return focal_length_to_fov(1.0f, m_relative_focal_length[m_fov_axis]);
}

void Testbed::set_fov(float val) {
	m_relative_focal_length = vec2(fov_to_focal_length(1, val));
}

vec2 Testbed::fov_xy() const {
	return focal_length_to_fov(ivec2(1), m_relative_focal_length);
}

void Testbed::set_fov_xy(const vec2& val) {
	m_relative_focal_length = fov_to_focal_length(ivec2(1), val);
}

void Testbed::set_max_level(float maxlevel) {
	if (!m_network) return;
	auto hg_enc = dynamic_cast<GridEncoding<network_precision_t>*>(m_encoding.get());
	if (hg_enc) {
		hg_enc->set_max_level(maxlevel);
	}

	reset_accumulation();
}

void Testbed::set_visualized_layer(int layer) {
	m_visualized_layer = layer;
	m_visualized_dimension = std::max(-1, std::min(m_visualized_dimension, (int)network_width(layer) - 1));
	reset_accumulation();
}

Testbed::Testbed() {
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

#ifdef NGP_GUI
	// Ensure we're running on the GPU that'll host our GUI. To do so, try creating a dummy
	// OpenGL context, figure out the GPU it's running on, and then kill that context again.
	if (!is_wsl() && glfwInit()) {
		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		GLFWwindow* offscreen_context = glfwCreateWindow(640, 480, "", NULL, NULL);

		if (offscreen_context) {
			glfwMakeContextCurrent(offscreen_context);

			int gl_device = -1;
			unsigned int device_count = 0;
			if (cudaGLGetDevices(&device_count, &gl_device, 1, cudaGLDeviceListAll) == cudaSuccess) {
				if (device_count > 0 && gl_device >= 0) {
					set_cuda_device(gl_device);
				}
			}

			glfwDestroyWindow(offscreen_context);
		}

		glfwTerminate();
	}
#endif

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

	set_mode();
	set_exposure(0);
	set_max_level(1.f);

	reset_camera();
}

Testbed::~Testbed() {

	// If any temporary file was created, make sure it's deleted
	clear_tmp_dir();

	if (m_render_window) {
		destroy_window();
	}
}

vec2 Testbed::calc_focal_length(const ivec2& resolution, const vec2& relative_focal_length, int fov_axis, float zoom) const {
	return relative_focal_length * (float)resolution[fov_axis] * zoom;
}

vec2 Testbed::render_screen_center(const vec2& screen_center) const {
	// see pixel_to_ray for how screen center is used; 0.5, 0.5 is 'normal'. we flip so that it becomes the point in the original image we want to center on.
	return (0.5f - screen_center) * m_zoom + 0.5f;
}

__global__ void dlss_prep_kernel(
	ivec2 resolution,
	uint32_t sample_index,
	vec2 focal_length,
	vec2 screen_center,
	vec3 parallax_shift,
	bool snap_to_pixel_centers,
	float* depth_buffer,
	const float znear,
	const float zfar,
	mat4x3 camera,
	mat4x3 prev_camera,
	cudaSurfaceObject_t depth_surface,
	cudaSurfaceObject_t mvec_surface,
	cudaSurfaceObject_t exposure_surface,
	Foveation foveation,
	Foveation prev_foveation,
	Lens lens
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	uint32_t idx = x + resolution.x * y;

	uint32_t x_orig = x;
	uint32_t y_orig = y;

	const float depth = depth_buffer[idx];
	vec2 mvec = motion_vector(
		sample_index,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera,
		prev_camera,
		screen_center,
		parallax_shift,
		snap_to_pixel_centers,
		depth,
		foveation,
		prev_foveation,
		lens
	);

	surf2Dwrite(make_float2(mvec.x, mvec.y), mvec_surface, x_orig * sizeof(float2), y_orig);

	// DLSS was trained on games, which presumably used standard normalized device coordinates (ndc)
	// depth buffers. So: convert depth to NDC with reasonable near- and far planes.
	surf2Dwrite(to_ndc_depth(depth, znear, zfar), depth_surface, x_orig * sizeof(float), y_orig);

	// First thread write an exposure factor of 1. Since DLSS will run on tonemapped data,
	// exposure is assumed to already have been applied to DLSS' inputs.
	if (x_orig == 0 && y_orig == 0) {
		surf2Dwrite(1.0f, exposure_surface, 0, 0);
	}
}

__global__ void spherical_checkerboard_kernel(
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera,
	vec2 screen_center,
	vec3 parallax_shift,
	Foveation foveation,
	Lens lens,
	vec4 background_color,
	vec4* frame_buffer
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	Ray ray = pixel_to_ray(
		0,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		false,
		0.0f,
		1.0f,
		0.0f,
		foveation,
		{}, // No need for hidden area mask
		lens
	);

	// Blend with checkerboard to break up reprojection weirdness in some VR runtimes
	host_device_swap(ray.d.z, ray.d.y);
	vec2 spherical = dir_to_spherical(normalize(ray.d)) * 32.0f / PI();
	const vec4 dark_gray = {0.5f, 0.5f, 0.5f, 1.0f};
	const vec4 light_gray = {0.55f, 0.55f, 0.55f, 1.0f};
	vec4 checker = fabsf(fmodf(floorf(spherical.x) + floorf(spherical.y), 2.0f)) < 0.5f ? dark_gray : light_gray;

	// Blend background color on top of checkerboard first (checkerboard is meant to be "behind" the background,
	// representing transparency), and then blend the result behind the frame buffer.
	background_color.rgb() = srgb_to_linear(background_color.rgb());
	background_color += (1.0f - background_color.a) * checker;

	uint32_t idx = x + resolution.x * y;
	frame_buffer[idx] += (1.0f - frame_buffer[idx].a) * background_color;
}

__global__ void vr_overlay_hands_kernel(
	ivec2 resolution,
	vec2 focal_length,
	mat4x3 camera,
	vec2 screen_center,
	vec3 parallax_shift,
	Foveation foveation,
	Lens lens,
	vec3 left_hand_pos,
	float left_grab_strength,
	vec4 left_hand_color,
	vec3 right_hand_pos,
	float right_grab_strength,
	vec4 right_hand_color,
	float hand_radius,
	EColorSpace output_color_space,
	cudaSurfaceObject_t surface
	// TODO: overwrite depth buffer
) {
	uint32_t x = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t y = threadIdx.y + blockDim.y * blockIdx.y;

	if (x >= resolution.x || y >= resolution.y) {
		return;
	}

	Ray ray = pixel_to_ray(
		0,
		{(int)x, (int)y},
		resolution,
		focal_length,
		camera,
		screen_center,
		parallax_shift,
		false,
		0.0f,
		1.0f,
		0.0f,
		foveation,
		{}, // No need for hidden area mask
		lens
	);

	vec4 color = vec4(0.0f);
	auto composit_hand = [&](vec3 hand_pos, float grab_strength, vec4 hand_color) {
		// Don't render the hand indicator if it's behind the ray origin.
		if (dot(ray.d, hand_pos - ray.o) < 0.0f) {
			return;
		}

		float distance = ray.distance_to(hand_pos);

		vec4 base_color = vec4(0.0f);
		const vec4 border_color = {0.4f, 0.4f, 0.4f, 0.4f};

		// Divide hand radius into an inner part (4/5ths) and a border (1/5th).
		float radius = hand_radius * 0.8f;
		float border_width = hand_radius * 0.2f;

		// When grabbing, shrink the inner part as a visual indicator.
		radius *= 0.5f + 0.5f * (1.0f - grab_strength);

		if (distance < radius) {
			base_color = hand_color;
		} else if (distance < radius + border_width) {
			base_color = border_color;
		} else {
			return;
		}

		// Make hand color opaque when grabbing.
		base_color.a = grab_strength + (1.0f - grab_strength) * base_color.a;
		color += base_color * (1.0f - color.a);
	};

	if (dot(ray.d, left_hand_pos - ray.o) < dot(ray.d, right_hand_pos - ray.o)) {
		composit_hand(left_hand_pos, left_grab_strength, left_hand_color);
		composit_hand(right_hand_pos, right_grab_strength, right_hand_color);
	} else {
		composit_hand(right_hand_pos, right_grab_strength, right_hand_color);
		composit_hand(left_hand_pos, left_grab_strength, left_hand_color);
	}

	// Blend with existing color of pixel
	vec4 prev_color;
	surf2Dread((float4*)&prev_color, surface, x * sizeof(float4), y);
	if (output_color_space == EColorSpace::SRGB) {
		prev_color.rgb() = srgb_to_linear(prev_color.rgb());
	}

	color += (1.0f - color.a) * prev_color;

	if (output_color_space == EColorSpace::SRGB) {
		color.rgb() = linear_to_srgb(color.rgb());
	}

	surf2Dwrite(to_float4(color), surface, x * sizeof(float4), y);
}

void Testbed::render_frame(
	cudaStream_t stream,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const mat4x3& prev_camera_matrix,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const vec4& nerf_rolling_shutter,
	const Foveation& foveation,
	const Foveation& prev_foveation,
	int visualized_dimension,
	CudaRenderBuffer& render_buffer,
	bool to_srgb,
	CudaDevice* device
) {
	if (!device) {
		device = &primary_device();
	}

	sync_device(render_buffer, *device);

	{
		auto device_guard = use_device(stream, render_buffer, *device);
		render_frame_main(*device, camera_matrix0, camera_matrix1, orig_screen_center, relative_focal_length, nerf_rolling_shutter, foveation, visualized_dimension);
	}

	render_frame_epilogue(stream, camera_matrix0, prev_camera_matrix, orig_screen_center, relative_focal_length, foveation, prev_foveation, render_buffer, to_srgb);
}

void Testbed::render_frame_main(
	CudaDevice& device,
	const mat4x3& camera_matrix0,
	const mat4x3& camera_matrix1,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const vec4& nerf_rolling_shutter,
	const Foveation& foveation,
	int visualized_dimension
) {
	device.render_buffer_view().clear(device.stream());

	if (!m_network) {
		return;
	}

	vec2 focal_length = calc_focal_length(device.render_buffer_view().resolution, relative_focal_length, m_fov_axis, m_zoom);
	vec2 screen_center = render_screen_center(orig_screen_center);

	if (!m_render_ground_truth || m_ground_truth_alpha < 1.0f) {
		render_nerf(device.stream(), device, device.render_buffer_view(), device.nerf_network(), device.data().density_grid_bitfield_ptr, focal_length, camera_matrix0, camera_matrix1, nerf_rolling_shutter, screen_center, foveation, visualized_dimension);
	}
}

void Testbed::render_frame_epilogue(
	cudaStream_t stream,
	const mat4x3& camera_matrix0,
	const mat4x3& prev_camera_matrix,
	const vec2& orig_screen_center,
	const vec2& relative_focal_length,
	const Foveation& foveation,
	const Foveation& prev_foveation,
	CudaRenderBuffer& render_buffer,
	bool to_srgb
) {
	vec2 focal_length = calc_focal_length(render_buffer.in_resolution(), relative_focal_length, m_fov_axis, m_zoom);
	vec2 screen_center = render_screen_center(orig_screen_center);

	render_buffer.set_color_space(m_color_space);
	render_buffer.set_tonemap_curve(m_tonemap_curve);

	Lens lens = m_nerf.render_with_lens_distortion ? m_nerf.render_lens : Lens{};

	// Prepare DLSS data: motion vectors, scaled depth, exposure
	if (render_buffer.dlss()) {
		auto res = render_buffer.in_resolution();

		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1 };

		dlss_prep_kernel<<<blocks, threads, 0, stream>>>(
			res,
			render_buffer.spp(),
			focal_length,
			screen_center,
			m_parallax_shift,
			m_snap_to_pixel_centers,
			render_buffer.depth_buffer(),
			m_ndc_znear,
			m_ndc_zfar,
			camera_matrix0,
			prev_camera_matrix,
			render_buffer.dlss()->depth(),
			render_buffer.dlss()->mvec(),
			render_buffer.dlss()->exposure(),
			foveation,
			prev_foveation,
			lens
		);

		render_buffer.set_dlss_sharpening(m_dlss_sharpening);
	}

	EColorSpace output_color_space = to_srgb ? EColorSpace::SRGB : EColorSpace::Linear;

	if (m_render_transparency_as_checkerboard) {
		mat4x3 checkerboard_transform = mat4x3::identity();

#ifdef NGP_GUI
		if (m_hmd && m_vr_frame_info && !m_vr_frame_info->views.empty()) {
			checkerboard_transform = m_vr_frame_info->views[0].pose;
		}
#endif

		auto res = render_buffer.in_resolution();
		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1 };
		spherical_checkerboard_kernel<<<blocks, threads, 0, stream>>>(
			res,
			focal_length,
			checkerboard_transform,
			screen_center,
			m_parallax_shift,
			foveation,
			lens,
			m_background_color,
			render_buffer.frame_buffer()
		);
	}

	render_buffer.accumulate(m_exposure, stream);
	render_buffer.tonemap(m_exposure, m_background_color, output_color_space, m_ndc_znear, m_ndc_zfar, m_snap_to_pixel_centers, stream);

	// Overlay the ground truth image if requested
	if (m_render_ground_truth) {
		auto const& metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
		if (m_ground_truth_render_mode == EGroundTruthRenderMode::Shade) {
			render_buffer.overlay_image(
				m_ground_truth_alpha,
				vec3(m_exposure) + m_nerf.training.cam_exposure[m_nerf.training.view].variable(),
				m_background_color,
				output_color_space,
				metadata.pixels,
				metadata.image_data_type,
				metadata.resolution,
				m_fov_axis,
				m_zoom,
				vec2(0.5f),
				stream
			);
		} else if (m_ground_truth_render_mode == EGroundTruthRenderMode::Depth && metadata.depth) {
			render_buffer.overlay_depth(
				m_ground_truth_alpha,
				metadata.depth,
				1.0f/m_nerf.training.dataset.scale,
				metadata.resolution,
				m_fov_axis,
				m_zoom,
				vec2(0.5f),
				stream
			);
		}
	}

	// Visualize the accumulated error map if requested
	if (m_nerf.training.render_error_overlay) {
		const float* err_data = m_nerf.training.error_map.data.data();
		ivec2 error_map_res = m_nerf.training.error_map.resolution;
		if (m_render_ground_truth) {
			err_data = m_nerf.training.dataset.sharpness_data.data();
			error_map_res = m_nerf.training.dataset.sharpness_resolution;
		}
		size_t emap_size = error_map_res.x * error_map_res.y;
		err_data += emap_size * m_nerf.training.view;

		GPUMemory<float> average_error;
		average_error.enlarge(1);
		average_error.memset(0);
		const float* aligned_err_data_s = (const float*)(((size_t)err_data)&~15);
		const float* aligned_err_data_e = (const float*)(((size_t)(err_data+emap_size))&~15);
		size_t reduce_size = aligned_err_data_e - aligned_err_data_s;
		reduce_sum(aligned_err_data_s, [reduce_size] __device__ (float val) { return max(val,0.f) / (reduce_size); }, average_error.data(), reduce_size, stream);
		auto const &metadata = m_nerf.training.dataset.metadata[m_nerf.training.view];
		render_buffer.overlay_false_color(metadata.resolution, to_srgb, m_fov_axis, stream, err_data, error_map_res, average_error.data(), m_nerf.training.error_overlay_brightness, m_render_ground_truth);
	}

#ifdef NGP_GUI
	// If in VR, indicate the hand position and render transparent background
	if (m_hmd && m_vr_frame_info) {
		auto& hands = m_vr_frame_info->hands;

		auto res = render_buffer.out_resolution();
		const dim3 threads = { 16, 8, 1 };
		const dim3 blocks = { div_round_up((uint32_t)res.x, threads.x), div_round_up((uint32_t)res.y, threads.y), 1 };
		vr_overlay_hands_kernel<<<blocks, threads, 0, stream>>>(
			res,
			focal_length * vec2(render_buffer.out_resolution()) / vec2(render_buffer.in_resolution()),
			camera_matrix0,
			screen_center,
			m_parallax_shift,
			foveation,
			lens,
			vr_to_world(hands[0].pose[3]),
			hands[0].grab_strength,
			{hands[0].pressing ? 0.8f : 0.0f, 0.0f, 0.0f, 0.8f},
			vr_to_world(hands[1].pose[3]),
			hands[1].grab_strength,
			{hands[1].pressing ? 0.8f : 0.0f, 0.0f, 0.0f, 0.8f},
			0.05f * m_scale, // Hand radius
			output_color_space,
			render_buffer.surface()
		);
	}
#endif
}

float Testbed::get_depth_from_renderbuffer(const CudaRenderBuffer& render_buffer, const vec2& uv) {
	if (!render_buffer.depth_buffer()) {
		return m_scale;
	}

	float depth;
	auto res = render_buffer.in_resolution();
	ivec2 depth_pixel = clamp(ivec2(uv * vec2(res)), 0, res - 1);

	CUDA_CHECK_THROW(cudaMemcpy(&depth, render_buffer.depth_buffer() + depth_pixel.x + depth_pixel.y * res.x, sizeof(float), cudaMemcpyDeviceToHost));
	return depth;
}

bool Testbed::frame_data_enqueue(const fs::path& path, std::queue<QueueObj>& queue, bool read_compression) { // yin: for ngp flow
	if (read_frame_thread_counter >= max_read_frame_thread_n) return false;
	std::thread current_thread(
		[&queue, read_compression](const fs::path path, std::thread last_thread, std::atomic<int64_t>& counter, cudaStream_t stream){
			counter++;
			std::ifstream f{native_string(path), std::ios::in | std::ios::binary};
			zstr::istream zf{f};
			json data;
			if (read_compression) {
				data = json::from_bson(zf);
			} else {
				data = json::from_bson(f);
			}
			QueueObj qobj = QueueObj{};
			if (data.contains("params_size") && data.contains("params")) {
				size_t params_size = data["params_size"];
				auto params_raw = data["params"].get_binary();
				if (params_raw.size() != params_size*sizeof(__half))
					throw std::runtime_error{"size of params and params_size not match."};
				__half* params = nullptr;
				CUDA_CHECK_THROW(cudaMalloc(&params, params_size*sizeof(__half)));
				CUDA_CHECK_THROW(cudaMemcpyAsync(params, params_raw.data(), params_size*sizeof(__half), cudaMemcpyHostToDevice, stream));
				qobj.params = params;
				qobj.params_index = nullptr;
				qobj.params_size = params_size;
			}
			if (data.contains("density_grid_bitfield_size") && data.contains("density_grid_bitfield")) {
				uint32_t density_grid_size = data["density_grid_bitfield_size"];
				auto density_grid_raw = data["density_grid_bitfield"].get_binary();
				if (density_grid_raw.size() != density_grid_size*sizeof(uint8_t))
					throw std::runtime_error{"size of density_grid and density_grid_size not match."};
				uint8_t* density_grid = nullptr;
				CUDA_CHECK_THROW(cudaMalloc(&density_grid, density_grid_size*sizeof(uint8_t)));
				CUDA_CHECK_THROW(cudaMemcpyAsync(density_grid, density_grid_raw.data(), density_grid_size*sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
				qobj.density_grid_bitfield = density_grid;
				qobj.density_grid_bitfield_size = density_grid_size;
			} else
			if (data.contains("density_grid_size") && data.contains("density_grid")) {
				uint32_t density_grid_size = data["density_grid_size"];
				auto density_grid_raw = data["density_grid"].get_binary();
				if (density_grid_raw.size() != density_grid_size*sizeof(__half))
					throw std::runtime_error{"size of density_grid and density_grid_size not match."};
				__half* density_grid = nullptr;
				CUDA_CHECK_THROW(cudaMalloc(&density_grid, density_grid_size*sizeof(__half)));
				CUDA_CHECK_THROW(cudaMemcpyAsync(density_grid, density_grid_raw.data(), density_grid_size*sizeof(__half), cudaMemcpyHostToDevice, stream));
				qobj.density_grid = density_grid;
				qobj.density_grid_index = nullptr;
				qobj.density_grid_size = density_grid_size;
			}
			counter--;
			if (last_thread.joinable()) last_thread.join();
			queue.push(qobj);
		}, std::move(path), std::move(last_update_frame_thread), std::ref(read_frame_thread_counter), m_stream.get()
	);
	last_update_frame_thread = std::move(current_thread);
	return true;
}

void Testbed::join_last_update_frame_thread() {
	if (last_update_frame_thread.joinable())
		last_update_frame_thread.join();
}

bool Testbed::load_frame_enqueue(const fs::path& path) { // yin: for ngp flow
	return frame_data_enqueue(path, load_frame_queue, read_compression);
}

bool Testbed::diff_frame_enqueue(const fs::path& path) { // yin: for ngp flow
	return frame_data_enqueue(path, diff_frame_queue, read_compression);
}

void Testbed::sync_grid_frame() { // yin: for ngp flow
	size_t m = n_params();
	if (this_grid_frame.size() != m) {
		this_grid_frame.resize(m);
		this_grid_frame.memset(-128);
	}
	if (last_grid_frame.size() != m) {
		last_grid_frame.resize(m);
		last_grid_frame.memset(-128);
	}
	parallel_for_gpu(m_stream.get(), m, [this_grid_frame=this_grid_frame.data(), last_grid_frame=last_grid_frame.data()] __device__ (size_t i) {
		last_grid_frame[i] = this_grid_frame[i];
	});
}

void Testbed::set_params(__half* params_gpu, size_t n, uint32_t* index_gpu) { // yin: for ngp flow
	size_t m = n_params();
	if (m_network->params() != nullptr){
		if (index_gpu != nullptr)
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->params(), params=params_gpu, index=index_gpu, m] __device__ (size_t i) {
			if (index[i] < m) local_params[index[i]] = (network_precision_t)params[i];
		});
		else
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->params(), params=params_gpu, m] __device__ (size_t i) {
			if (i < m) local_params[i] = (network_precision_t)params[i];
		});
	}
	if (m_network->inference_params() != nullptr && m_network->inference_params() != m_network->params()) {
		if (index_gpu != nullptr)
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->inference_params(), params=params_gpu, index=index_gpu, m] __device__ (size_t i) {
			if (index[i] < m) local_params[index[i]] = (network_precision_t)params[i];
		});
		else
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->inference_params(), params=params_gpu, m] __device__ (size_t i) {
			if (i < m) local_params[i] = (network_precision_t)params[i];
		});
	}
}

void Testbed::set_params_setframe(int64_t frame, __half* params_gpu, size_t n, uint32_t* index_gpu) { // yin: for ngp flow
	size_t m = n_params();
	if (m_network->params() != nullptr){
		if (index_gpu != nullptr)
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->params(), params=params_gpu, index=index_gpu, m, frame, grid_frame=this_grid_frame.data()] __device__ (size_t i) {
			if (index[i] < m) {
				if (grid_frame[index[i]]>=0 && local_params[index[i]] == (network_precision_t)params[i]) return;
				local_params[index[i]] = (network_precision_t)params[i];
				grid_frame[index[i]] = frame;
			}
		});
		else
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->params(), params=params_gpu, m, frame, grid_frame=this_grid_frame.data()] __device__ (size_t i) {
			if (i < m) {
				if (grid_frame[i]>=0 && local_params[i] == (network_precision_t)params[i]) return;
				local_params[i] = (network_precision_t)params[i];
				grid_frame[i] = frame;
			}
		});
	}
	if (m_network->inference_params() != nullptr && m_network->inference_params() != m_network->params()) {
		if (index_gpu != nullptr)
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->inference_params(), params=params_gpu, index=index_gpu, m] __device__ (size_t i) {
			if (index[i] < m) local_params[index[i]] = (network_precision_t)params[i];
		});
		else
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->inference_params(), params=params_gpu, m] __device__ (size_t i) {
			if (i < m) local_params[i] = (network_precision_t)params[i];
		});
	}
}

void Testbed::add_params(__half* params_gpu, size_t n, uint32_t* index_gpu) { // yin: for ngp flow
	size_t m = n_params();
	if (m_network->params() != nullptr){
		if (index_gpu != nullptr)
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->params(), params=params_gpu, index=index_gpu, m] __device__ (size_t i) {
			if (index[i] < m) local_params[index[i]] += (network_precision_t)params[i];
		});
		else
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->params(), params=params_gpu, m] __device__ (size_t i) {
			if (i < m) local_params[i] += (network_precision_t)params[i];
		});
	}
	if (m_network->inference_params() != nullptr && m_network->inference_params() != m_network->params()) {
		if (index_gpu != nullptr)
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->inference_params(), params=params_gpu, index=index_gpu, m] __device__ (size_t i) {
			if (index[i] < m) local_params[index[i]] += (network_precision_t)params[i];
		});
		else
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->inference_params(), params=params_gpu, m] __device__ (size_t i) {
			if (i < m) local_params[i] += (network_precision_t)params[i];
		});
	}
}

void Testbed::add_params_setframe(int64_t frame, __half* params_gpu, size_t n, uint32_t* index_gpu) { // yin: for ngp flow
	size_t m = n_params();
	if (m_network->params() != nullptr){
		if (index_gpu != nullptr)
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->params(), params=params_gpu, index=index_gpu, m, frame, grid_frame=this_grid_frame.data()] __device__ (size_t i) {
			if (index[i] < m) {
				if (grid_frame[index[i]]>=0 && (float)params[i] == 0) return;
				local_params[index[i]] += (network_precision_t)params[i];
				grid_frame[index[i]] = frame;
			}
		});
		else
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->params(), params=params_gpu, m, frame, grid_frame=this_grid_frame.data()] __device__ (size_t i) {
			if (i < m) {
				if (grid_frame[i]>=0 && (float)params[i] == 0) return;
				local_params[i] += (network_precision_t)params[i];
				grid_frame[i] = frame;
			}
		});
	}
	if (m_network->inference_params() != nullptr && m_network->inference_params() != m_network->params()) {
		if (index_gpu != nullptr)
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->inference_params(), params=params_gpu, index=index_gpu, m] __device__ (size_t i) {
			if (index[i] < m) local_params[index[i]] += (network_precision_t)params[i];
		});
		else
		parallel_for_gpu(m_stream.get(), n, [local_params=m_network->inference_params(), params=params_gpu, m] __device__ (size_t i) {
			if (i < m) local_params[i] += (network_precision_t)params[i];
		});
	}
}

void Testbed::set_density_grid(__half* density_grid_gpu, size_t n, uint32_t* index_gpu, uint8_t* bitfield_gpu, size_t bit_n) { // yin: for ngp flow
	size_t m = NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1);
	if (index_gpu != nullptr)
	parallel_for_gpu(m_stream.get(), n, [local_density_grid=m_nerf.density_grid.data(), density_grid=density_grid_gpu, index=index_gpu, m] __device__ (size_t i) {
		if (index[i] < m) local_density_grid[index[i]] = (float)density_grid[i];
	});
	else
	parallel_for_gpu(m_stream.get(), n, [local_density_grid=m_nerf.density_grid.data(), density_grid=density_grid_gpu, m] __device__ (size_t i) {
		if (i < m) local_density_grid[i] = (float)density_grid[i];
	});

	if (bitfield_gpu != nullptr && bit_n > 0)
	CUDA_CHECK_THROW(cudaMemcpyAsync(m_nerf.density_grid_bitfield.data(), bitfield_gpu, bit_n*sizeof(uint8_t), cudaMemcpyHostToDevice, m_stream.get()));
	else {
	if (m_nerf.density_grid.size() == NERF_GRID_N_CELLS() * (m_nerf.max_cascade + 1)) {
		update_density_grid_mean_and_bitfield(m_stream.get());
	} else if (m_nerf.density_grid.size() != 0) {
		// A size of 0 indicates that the density grid was never populated, which is a valid state of a (yet) untrained model.
		throw std::runtime_error{"Incompatible number of grid cascades."};
	}
	}
}

bool Testbed::load_frame_dequeue() { // yin: for ngp flow
	if (load_frame_queue.empty()) return false;
	QueueObj obj = load_frame_queue.front();
	set_params(obj.params, obj.params_size, obj.params_index);
	set_density_grid(obj.density_grid, obj.density_grid_size, obj.density_grid_index, obj.density_grid_bitfield, obj.density_grid_bitfield_size);
	load_frame_queue.pop();
	FreeQueueObj(obj);
	return true;
}

bool Testbed::load_frame_dequeue_setframe(int64_t frame) { // yin: for ngp flow
	sync_grid_frame();
	if (load_frame_queue.empty()) return false;
	QueueObj obj = load_frame_queue.front();
	set_params_setframe(frame, obj.params, obj.params_size, obj.params_index);
	set_density_grid(obj.density_grid, obj.density_grid_size, obj.density_grid_index, obj.density_grid_bitfield, obj.density_grid_bitfield_size);
	load_frame_queue.pop();
	FreeQueueObj(obj);
	return true;
}

bool Testbed::diff_frame_dequeue() { // yin: for ngp flow
	if (diff_frame_queue.empty()) return false;
	QueueObj obj = diff_frame_queue.front();
	add_params(obj.params, obj.params_size, obj.params_index);
	set_density_grid(obj.density_grid, obj.density_grid_size, obj.density_grid_index, obj.density_grid_bitfield, obj.density_grid_bitfield_size);
	diff_frame_queue.pop();
	FreeQueueObj(obj);
	return true;
}

bool Testbed::diff_frame_dequeue_setframe(int64_t frame) { // yin: for ngp flow
	sync_grid_frame();
	if (diff_frame_queue.empty()) return false;
	QueueObj obj = diff_frame_queue.front();
	add_params_setframe(frame, obj.params, obj.params_size, obj.params_index);
	set_density_grid(obj.density_grid, obj.density_grid_size, obj.density_grid_index, obj.density_grid_bitfield, obj.density_grid_bitfield_size);
	diff_frame_queue.pop();
	FreeQueueObj(obj);
	return true;
}

size_t extract_nonzero(__half* data, uint32_t** index, size_t size, cudaStream_t stream) { // yin: for ngp flow
	// float* data_fp = nullptr;
	// CUDA_CHECK_THROW(cudaMalloc(&data_fp, size*sizeof(float)));
	// float* data_fp_host = nullptr;

	// parallel_for_gpu(stream, size, [data=data, data_fp=data_fp] __device__ (size_t i) {
	// 	data_fp[i] = (float)data[i];
	// });
	// data_fp_host = (float*)malloc(size*sizeof(float));
	// CUDA_CHECK_THROW(cudaMemcpyAsync(data_fp_host, data_fp, size*sizeof(float), cudaMemcpyDeviceToHost, stream));
	// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	// float* data0 = data_fp_host;

	unsigned int size_host = 0;
	unsigned int* size_device = nullptr;
	CUDA_CHECK_THROW(cudaMalloc(&size_device, sizeof(unsigned int)));
	CUDA_CHECK_THROW(cudaMalloc(index, size*sizeof(uint32_t)));
	parallel_for_gpu(stream, size, [data=data, index=*index, size_device=size_device] __device__ (size_t i) {
		if (i==0) *size_device = 0;
		__half d = data[i];
		__syncthreads();
		if (d > (__half)0.) {
			auto idx = atomicAdd(size_device, (unsigned int)1);
			index[idx] = (uint32_t)i;
			data[idx] = d;
		}
	});
	CUDA_CHECK_THROW(cudaMemcpyAsync(&size_host, size_device, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
	CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	CUDA_CHECK_THROW(cudaFree(size_device));

	// parallel_for_gpu(stream, size, [data=data, data_fp=data_fp] __device__ (size_t i) {
	// 	data_fp[i] = (float)data[i];
	// });
	// data_fp_host = (float*)malloc(size*sizeof(float));
	// CUDA_CHECK_THROW(cudaMemcpyAsync(data_fp_host, data_fp, size*sizeof(float), cudaMemcpyDeviceToHost, stream));
	// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
	// float* data1 = data_fp_host;

	// uint32_t* index_host = (uint32_t*)malloc(size*sizeof(uint32_t));
	// CUDA_CHECK_THROW(cudaMemcpyAsync(index_host, *index, size*sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
	// CUDA_CHECK_THROW(cudaStreamSynchronize(stream));

	// for (unsigned int i = 0;i<size_host;i+=10000) {
	// 	tlog::info() << data0[index_host[i]] << data1[i] << ' ' << index_host[i];
	// }
	// free(data0);
	// free(data1);

	return (size_t)size_host;
}

bool Testbed::diff_frame_nonzero_dequeue() { // yin: for ngp flow
	if (diff_frame_queue.empty()) return false;
	QueueObj obj = diff_frame_queue.front();
	if (obj.params_index != nullptr) add_params(obj.params, obj.params_size, obj.params_index);
	else {
		obj.params_size = extract_nonzero(obj.params, &obj.params_index, obj.params_size, m_stream.get());
		add_params(obj.params, obj.params_size, obj.params_index);
	}
	if (obj.density_grid_index != nullptr) set_density_grid(obj.density_grid, obj.density_grid_size, obj.density_grid_index);
	else {
		obj.density_grid_size = extract_nonzero(obj.density_grid, &obj.density_grid_index, obj.density_grid_size, m_stream.get());
		set_density_grid(obj.density_grid, obj.density_grid_size, obj.density_grid_index);
	}
	diff_frame_queue.pop();
	FreeQueueObj(obj);
	return true;
}

}

