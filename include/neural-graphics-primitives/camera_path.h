/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   camera_path.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once

#include <neural-graphics-primitives/common.h>

#include <tiny-cuda-nn/common.h>

#include <imgui/imgui.h>
#include <imguizmo/ImGuizmo.h>

#include <chrono>
#include <vector>

struct ImDrawList;

NGP_NAMESPACE_BEGIN

struct CameraKeyframe {
	Eigen::Vector4f R;
	Eigen::Vector3f T;
	float slice;
	float scale; // not a scale factor as in scaling the world, but the value of m_scale (setting the focal plane along with slice)
	float fov;
	float aperture_size;
	int glow_mode;
	float glow_y_cutoff;
	Eigen::Matrix<float, 3, 4> m() const {
		Eigen::Matrix<float, 3, 4> rv;
		rv.col(3) = T;
		rv.block<3,3>(0,0) = Eigen::Quaternionf(R).normalized().toRotationMatrix();
		return rv;
	}

	void from_m(const Eigen::Matrix<float, 3, 4>& rv) {
		T = rv.col(3);
		// auto q = Eigen::Quaternionf(rv.block<3,3>(0,0));
		auto q = Eigen::Quaternionf(rv.block<3,3>(0,0));
		R = Eigen::Vector4f(q.x(), q.y(), q.z(), q.w());
	}

	CameraKeyframe() = default;
	CameraKeyframe(const Eigen::Vector4f &r, const Eigen::Vector3f &t, float sl, float sc, float fv, float df, int gm, float gyc) : R(r), T(t), slice(sl), scale(sc), fov(fv), aperture_size(df), glow_mode(gm), glow_y_cutoff(gyc) {}
	CameraKeyframe(Eigen::Matrix<float, 3, 4> m, float sl, float sc, float fv, float df, int gm, float gyc) : slice(sl), scale(sc), fov(fv), aperture_size(df), glow_mode(gm), glow_y_cutoff(gyc) { T=m.col(3); R=Eigen::Quaternionf(m.block<3,3>(0,0)).coeffs();  }
	CameraKeyframe operator*(float f) const { return {R*f, T*f, slice*f, scale*f, fov*f, aperture_size*f, glow_mode, glow_y_cutoff*f}; }
	CameraKeyframe operator+(const CameraKeyframe &rhs) const {
		Eigen::Vector4f Rr=rhs.R;
		if (Rr.dot(R)<0.f) Rr=-Rr;
		return {R+Rr, T+rhs.T, slice+rhs.slice, scale+rhs.scale, fov+rhs.fov, aperture_size+rhs.aperture_size, glow_mode, glow_y_cutoff+rhs.glow_y_cutoff};
	}
	bool SamePosAs(const CameraKeyframe &rhs) const {
		return (T-rhs.T).norm()<0.0001f && fabsf(R.dot(rhs.R))>=0.999f;
	}
};

CameraKeyframe lerp(const CameraKeyframe& p0, const CameraKeyframe& p1, float t, float t0, float t1);
CameraKeyframe spline(float t, const CameraKeyframe& p0, const CameraKeyframe& p1, const CameraKeyframe& p2, const CameraKeyframe& p3);

struct CameraPath {
	std::vector<CameraKeyframe> keyframes;
	bool update_cam_from_path = false;
	float play_time = 0.f;
	float auto_play_speed = 0.f;
	// If loop is set true, the last frame set will be more like "next to last,"
	// with animation then returning back to the first frame, making a continuous loop.
	// Note that the user does not have to (and should not normally) duplicate the first frame to be the last frame.
	bool loop = false;

	struct RenderSettings {
		Eigen::Vector2i resolution = {1920, 1080};
		int spp = 8;
		float fps = 60.0f;
		float duration_seconds = 5.0f;
		float shutter_fraction = 0.5f;
		int quality = 10;

		uint32_t n_frames() const {
			return (uint32_t)((double)duration_seconds * fps);
		}

		float frame_seconds() const {
			return 1.0f / (duration_seconds * fps);
		}

		float frame_milliseconds() const {
			return 1000.0f / (duration_seconds * fps);
		}

		std::string filename = "video.mp4";
	};

	RenderSettings render_settings;
	bool rendering = false;
	uint32_t render_frame_idx = 0;
	std::chrono::time_point<std::chrono::steady_clock> render_start_time;

	Eigen::Matrix<float, 3, 4> render_frame_end_camera;

	const CameraKeyframe& get_keyframe(int i) {
		if (loop) {
			int size = (int)keyframes.size();
			// add size to ensure no negative value is generated by modulo
			return keyframes[(i + size) % size];
		} else {
			return keyframes[tcnn::clamp(i, 0, (int)keyframes.size()-1)];
		}
	}
	CameraKeyframe eval_camera_path(float t) {
		if (keyframes.empty())
			return {};
		// make room for last frame == first frame when looping
		t *= (float)(loop ? keyframes.size() : keyframes.size()-1);
		int t1 = (int)floorf(t);
		return spline(t-floorf(t), get_keyframe(t1-1), get_keyframe(t1), get_keyframe(t1+1), get_keyframe(t1+2));
	}

	void save(const fs::path& path);
	void load(const fs::path& path, const Eigen::Matrix<float, 3, 4> &first_xform);

#ifdef NGP_GUI
	ImGuizmo::MODE m_gizmo_mode = ImGuizmo::LOCAL;
	ImGuizmo::OPERATION m_gizmo_op = ImGuizmo::TRANSLATE;
	bool imgui_viz(ImDrawList* list, Eigen::Matrix<float, 4, 4>& view2proj, Eigen::Matrix<float, 4, 4>& world2proj, Eigen::Matrix<float, 4, 4>& world2view, Eigen::Vector2f focal, float aspect);
	int imgui(char path_filename_buf[1024], float frame_milliseconds, Eigen::Matrix<float, 3, 4>& camera, float slice_plane_z, float scale, float fov, float aperture_size, float bounding_radius, const Eigen::Matrix<float, 3, 4>& first_xform, int glow_mode, float glow_y_cutoff);
#endif
};

#ifdef NGP_GUI
void add_debug_line(ImDrawList* list, const Eigen::Matrix<float, 4, 4>&proj, Eigen::Vector3f a, Eigen::Vector3f b, uint32_t col = 0xffffffff, float thickness = 1.0f);
void visualize_unit_cube(ImDrawList* list, const Eigen::Matrix<float, 4, 4>& world2proj, const Eigen::Vector3f& a, const Eigen::Vector3f& b, const Eigen::Matrix3f& render_aabb_to_local);
void visualize_nerf_camera(ImDrawList* list, const Eigen::Matrix<float, 4, 4>& world2proj, const Eigen::Matrix<float, 3, 4>& xform, float aspect, uint32_t col = 0x80ffffff, float thickness = 1.0f);
#endif

NGP_NAMESPACE_END

