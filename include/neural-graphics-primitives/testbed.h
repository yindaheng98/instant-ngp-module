/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   testbed.h
 *  @author Thomas Müller & Alex Evans, NVIDIA
 */

#pragma once
#include <thread> // std::thread
#include <queue> // std::queue
#include <neural-graphics-primitives/adam_optimizer.h>
#include <neural-graphics-primitives/camera_path.h>
#include <neural-graphics-primitives/common.h>
#include <neural-graphics-primitives/discrete_distribution.h>
#include <neural-graphics-primitives/nerf.h>
#include <neural-graphics-primitives/nerf_loader.h>
#include <neural-graphics-primitives/render_buffer.h>
#include <neural-graphics-primitives/shared_queue.h>
#include <neural-graphics-primitives/thread_pool.h>
#include <neural-graphics-primitives/trainable_buffer.cuh>

#ifdef NGP_GUI
#  include <neural-graphics-primitives/openxr_hmd.h>
#endif

#include <tiny-cuda-nn/multi_stream.h>
#include <tiny-cuda-nn/random.h>

#include <json/json.hpp>

#ifdef NGP_PYTHON
#  include <pybind11/pybind11.h>
#  include <pybind11/numpy.h>
#endif

#include <thread>

struct GLFWwindow;

namespace tcnn {
template <typename T> class Loss;
template <typename T> class Optimizer;
template <typename T> class Encoding;
template <typename T, typename PARAMS_T> class Network;
template <typename T, typename PARAMS_T, typename COMPUTE_T> class Trainer;
template <uint32_t N_DIMS, uint32_t RANK, typename T> class TrainableBuffer;
}

namespace ngp {

template <typename T> class NerfNetwork;
class TriangleOctree;
class TriangleBvh;
struct Triangle;
class GLTexture;
int do_system(const std::string& cmd);
json merge_parent_network_config(const json& child, const fs::path& child_path);

class Testbed {
public:
	Testbed(ETestbedMode mode = ETestbedMode::None);
	~Testbed();

	Testbed(ETestbedMode mode, const fs::path& data_path) : Testbed(mode) { load_training_data(data_path); }
	Testbed(ETestbedMode mode, const fs::path& data_path, const fs::path& network_config_path) : Testbed(mode, data_path) { reload_network_from_file(network_config_path); }
	Testbed(ETestbedMode mode, const fs::path& data_path, const nlohmann::json& network_config) : Testbed(mode, data_path) { reload_network_from_json(network_config); }

	bool clear_tmp_dir();
	void update_imgui_paths();
	void load_training_data(const fs::path& path);
	void reload_training_data();
	void clear_training_data();

	void set_mode(ETestbedMode mode);

	using distance_fun_t = std::function<void(uint32_t, const vec3*, float*, cudaStream_t)>;
	using normals_fun_t = std::function<void(uint32_t, const vec3*, vec3*, cudaStream_t)>;

	class NerfTracer {
	public:
		NerfTracer() {}

		void init_rays_from_camera(
			uint32_t spp,
			uint32_t padded_output_width,
			uint32_t n_extra_dims,
			const ivec2& resolution,
			const vec2& focal_length,
			const mat4x3& camera_matrix0,
			const mat4x3& camera_matrix1,
			const vec4& rolling_shutter,
			const vec2& screen_center,
			const vec3& parallax_shift,
			bool snap_to_pixel_centers,
			const BoundingBox& render_aabb,
			const mat3& render_aabb_to_local,
			float near_distance,
			float plane_z,
			float aperture_size,
			const Foveation& foveation,
			const Lens& lens,
			const Buffer2DView<const vec4>& envmap,
			const Buffer2DView<const vec2>& distortion,
			vec4* frame_buffer,
			float* depth_buffer,
			const Buffer2DView<const uint8_t>& hidden_area_mask,
			const uint8_t* grid,
			int show_accel,
			uint32_t max_mip,
			float cone_angle_constant,
			ERenderMode render_mode,
			cudaStream_t stream
		);

		uint32_t trace(
			const std::shared_ptr<NerfNetwork<network_precision_t>>& network,
			const BoundingBox& render_aabb,
			const mat3& render_aabb_to_local,
			const BoundingBox& train_aabb,
			const vec2& focal_length,
			float cone_angle_constant,
			const uint8_t* grid,
			ERenderMode render_mode,
			const mat4x3 &camera_matrix,
			float depth_scale,
			int visualized_layer,
			int visualized_dim,
			ENerfActivation rgb_activation,
			ENerfActivation density_activation,
			int show_accel,
			uint32_t max_mip,
			float min_transmittance,
			float glow_y_cutoff,
			int glow_mode,
			const float* extra_dims_gpu,
			cudaStream_t stream,
			GPUMemory<uint32_t>*& grid_hit, // yin: for ngp flow
			bool get_grid_hit, // yin: for ngp flow
			bool get_grid_hit_only, // yin: for ngp flow
			GPUMemory<float>* density_grid, // yin: for ngp flow
			bool debug_density_grid_replay // yin: for ngp flow
		);

		void enlarge(size_t n_elements, uint32_t padded_output_width, uint32_t n_extra_dims, cudaStream_t stream);
		RaysNerfSoa& rays_hit() { return m_rays_hit; }
		RaysNerfSoa& rays_init() { return m_rays[0]; }
		uint32_t n_rays_initialized() const { return m_n_rays_initialized; }

	private:
		RaysNerfSoa m_rays[2];
		RaysNerfSoa m_rays_hit;
		network_precision_t* m_network_output;
		float* m_network_input;
		uint32_t* m_hit_counter;
		uint32_t* m_alive_counter;
		uint32_t m_n_rays_initialized = 0;
		GPUMemoryArena::Allocation m_scratch_alloc;
	};

	struct LevelStats {
		float mean() { return count ? (x / (float)count) : 0.f; }
		float variance() { return count ? (xsquared - (x * x) / (float)count) / (float)count : 0.f; }
		float sigma() { return sqrtf(variance()); }
		float fraczero() { return (float)numzero / float(count + numzero); }
		float fracquant() { return (float)numquant / float(count); }

		float x;
		float xsquared;
		float min;
		float max;
		int numzero;
		int numquant;
		int count;
	};

	// Due to mixed-precision training, small loss values can lead to
	// underflow (round to zero) in the gradient computations. Hence,
	// scale the loss (and thereby gradients) up by this factor and
	// divide it out in the optimizer later on.
	static constexpr float LOSS_SCALE() { return default_loss_scale<network_precision_t>(); }

	struct NetworkDims {
		uint32_t n_input;
		uint32_t n_output;
		uint32_t n_pos;
	};

	NetworkDims network_dims_nerf() const;

	NetworkDims network_dims() const;

	void training_prep_volume(uint32_t batch_size, cudaStream_t stream) {}

	class CudaDevice;

	void render_nerf(
		cudaStream_t stream,
		CudaDevice& device,
		const CudaRenderBufferView& render_buffer,
		const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network,
		const uint8_t* density_grid_bitfield,
		const vec2& focal_length,
		const mat4x3& camera_matrix0,
		const mat4x3& camera_matrix1,
		const vec4& rolling_shutter,
		const vec2& screen_center,
		const Foveation& foveation,
		int visualized_dimension
	);

	void render_frame(
		cudaStream_t stream,
		const mat4x3& camera_matrix0,
		const mat4x3& camera_matrix1,
		const mat4x3& prev_camera_matrix,
		const vec2& screen_center,
		const vec2& relative_focal_length,
		const vec4& nerf_rolling_shutter,
		const Foveation& foveation,
		const Foveation& prev_foveation,
		int visualized_dimension,
		CudaRenderBuffer& render_buffer,
		bool to_srgb = true,
		CudaDevice* device = nullptr
	);
	void render_frame_main(
		CudaDevice& device,
		const mat4x3& camera_matrix0,
		const mat4x3& camera_matrix1,
		const vec2& screen_center,
		const vec2& relative_focal_length,
		const vec4& nerf_rolling_shutter,
		const Foveation& foveation,
		int visualized_dimension
	);
	void render_frame_epilogue(
		cudaStream_t stream,
		const mat4x3& camera_matrix0,
		const mat4x3& prev_camera_matrix,
		const vec2& screen_center,
		const vec2& relative_focal_length,
		const Foveation& foveation,
		const Foveation& prev_foveation,
		CudaRenderBuffer& render_buffer,
		bool to_srgb = true
	);
	void visualize_nerf_cameras(ImDrawList* list, const mat4& world2proj);
	fs::path find_network_config(const fs::path& network_config_path);
	nlohmann::json load_network_config(std::istream& stream, bool is_compressed);
	nlohmann::json load_network_config(const fs::path& network_config_path);
	void reload_network_from_file(const fs::path& path = "");
	void reload_network_from_json(const nlohmann::json& json, const std::string& config_base_path=""); // config_base_path is needed so that if the passed in json uses the 'parent' feature, we know where to look... be sure to use a filename, or if a directory, end with a trailing slash
	void reset_accumulation(bool due_to_camera_movement = false, bool immediate_redraw = true);
	void redraw_next_frame() {
		m_render_skip_due_to_lack_of_camera_movement_counter = 0;
	}
	bool reprojection_available() { return m_dlss; }
	static ELossType string_to_loss_type(const std::string& str);
	void reset_network(bool clear_density_grid = true);
	void create_empty_nerf_dataset(size_t n_images, int aabb_scale = 1, bool is_hdr = false);
	void load_nerf(const fs::path& data_path);
	void load_nerf_post();
	void set_exposure(float exposure) { m_exposure = exposure; }
	void set_max_level(float maxlevel);
	void set_visualized_dim(int dim);
	void set_visualized_layer(int layer);
	void translate_camera(const vec3& rel, const mat3& rot, bool allow_up_down = true);
	mat3 rotation_from_angles(const vec2& angles) const;
	void mouse_drag();
	void mouse_wheel();
	void load_file(const fs::path& path);
	void set_nerf_camera_matrix(const mat4x3& cam);
	vec3 look_at() const;
	void set_look_at(const vec3& pos);
	float scale() const { return m_scale; }
	void set_scale(float scale);
	vec3 view_pos() const { return m_camera[3]; }
	vec3 view_dir() const { return m_camera[2]; }
	vec3 view_up() const { return m_camera[1]; }
	vec3 view_side() const { return m_camera[0]; }
	void set_view_dir(const vec3& dir);
	void first_training_view();
	void last_training_view();
	void previous_training_view();
	void next_training_view();
	void set_camera_to_training_view(int trainview);
	void reset_camera();
	bool keyboard_event();
	void update_density_grid_nerf(float decay, uint32_t n_uniform_density_grid_samples, uint32_t n_nonuniform_density_grid_samples, cudaStream_t stream, bool smallest = false);
	void update_density_grid_mean_and_bitfield(cudaStream_t stream);
	void mark_density_grid_in_sphere_empty(const vec3& pos, float radius, cudaStream_t stream);

	struct NerfCounters {
		GPUMemory<uint32_t> numsteps_counter; // number of steps each ray took
		GPUMemory<uint32_t> numsteps_counter_compacted; // number of steps each ray took
		GPUMemory<float> loss;

		uint32_t rays_per_batch = 1<<12;
		uint32_t n_rays_total = 0;
		uint32_t measured_batch_size = 0;
		uint32_t measured_batch_size_before_compaction = 0;

		void prepare_for_training_steps(cudaStream_t stream);
		float update_after_training(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
	};

	void train_nerf(uint32_t target_batch_size, bool get_loss_scalar, cudaStream_t stream);
	void train_nerf_step(uint32_t target_batch_size, NerfCounters& counters, cudaStream_t stream);
	void set_train(bool mtrain);

	template <typename T>
	void dump_parameters_as_images(const T* params, const std::string& filename_base);

	void prepare_next_camera_path_frame();
	void imgui();
	void training_prep_nerf(uint32_t batch_size, cudaStream_t stream);
	void train(uint32_t batch_size);
	vec2 calc_focal_length(const ivec2& resolution, const vec2& relative_focal_length, int fov_axis, float zoom) const;
	vec2 render_screen_center(const vec2& screen_center) const;
	void optimise_mesh_step(uint32_t N_STEPS);
	void compute_mesh_vertex_colors();
	GPUMemory<float> get_density_on_grid(ivec3 res3d, const BoundingBox& aabb, const mat3& render_aabb_to_local); // network version (nerf or sdf)
	GPUMemory<vec4> get_rgba_on_grid(ivec3 res3d, vec3 ray_dir, bool voxel_centers, float depth, bool density_as_alpha = false);
	int marching_cubes(ivec3 res3d, const BoundingBox& render_aabb, const mat3& render_aabb_to_local, float thresh);

	float get_depth_from_renderbuffer(const CudaRenderBuffer& render_buffer, const vec2& uv);
	vec3 get_3d_pos_from_pixel(const CudaRenderBuffer& render_buffer, const vec2& focus_pixel);
	void autofocus();
	size_t n_params();
	size_t first_encoder_param();
	size_t n_encoding_params();

#ifdef NGP_PYTHON
	pybind11::dict compute_marching_cubes_mesh(ivec3 res3d = ivec3(128), BoundingBox aabb = BoundingBox{vec3(0.0f), vec3(1.0f)}, float thresh=2.5f);
	pybind11::array_t<float> render_to_cpu(int width, int height, int spp, bool linear, float start_t, float end_t, float fps, float shutter_fraction);
	pybind11::array_t<float> view(bool linear, size_t view) const;
	pybind11::array_t<float> screenshot(bool linear, bool front_buffer) const;
	void override_sdf_training_data(pybind11::array_t<float> points, pybind11::array_t<float> distances);
	// void load_params(pybind11::array_t<float> params, pybind11::array_t<int> index); // yin: for ngp flow
	// void diff_params(pybind11::array_t<float> params_diff, pybind11::array_t<int> index); // yin: for ngp flow
	// void load_density_grid(pybind11::array_t<float> density_grid, pybind11::array_t<int> index); // yin: for ngp flow
	// void diff_density_grid(pybind11::array_t<float> density_grid_diff, pybind11::array_t<int> index); // yin: for ngp flow
	// void load_params_enqueue_py(pybind11::array_t<float> params, pybind11::array_t<int> index); // yin: for ngp flow
	// void diff_params_enqueue_py(pybind11::array_t<float> params, pybind11::array_t<int> index); // yin: for ngp flow
	// void load_density_grid_enqueue_py(pybind11::array_t<float> density_grid, pybind11::array_t<int> index); // yin: for ngp flow
	// void diff_density_grid_enqueue_py(pybind11::array_t<float> density_grid, pybind11::array_t<int> index); // yin: for ngp flow
#endif
	bool load_frame_enqueue(const fs::path& path); // yin: for ngp flow
	bool diff_frame_enqueue(const fs::path& path); // yin: for ngp flow
	bool load_frame_dequeue(); // yin: for ngp flow
	bool diff_frame_dequeue(); // yin: for ngp flow
	bool diff_frame_nonzero_dequeue(); // yin: for ngp flow
	bool load_frame_dequeue_setframe(int64_t frame); // yin: for ngp flow
	bool diff_frame_dequeue_setframe(int64_t frame); // yin: for ngp flow
	void enable_residual_regulization(network_precision_t l2_reg); // yin: for ngp flow
	json dump_camera(); // yin: for ngp flow
	json dump_views(); // yin: for ngp flow
	void load_camera(json camera, json views); // yin: for ngp flow
	int64_t max_read_frame_thread_n = 8; // yin: for ngp flow
	bool get_grid_hit = false; // yin: for ngp flow
	bool get_grid_hit_only = false; // yin: for ngp flow
	std::string grid_hit_path; // yin: for ngp flow
	nlohmann::json grid_hit_json; // yin: for ngp flow
	fs::path  grid_hit_json_path; // yin: for ngp flow
	void save_grid_hit(const fs::path& path, bool compress = true); // yin: for ngp flow
	void do_grid_hit(GPUMemory<uint32_t>* grid_hit); // yin: for ngp flow
	void sync_grid_frame(); // yin: for ngp flow
	bool  read_compression = true; // yin: for ngp flow
	uint64_t M_blimit = 10000; // yin: for ngp flow
	float gamma_blimit = 0.5; // yin: for ngp flow
	bool  should_save_image = false; // yin: for ngp flow
	void save_image(CudaRenderBuffer& buffer); // yin: for ngp flow
	std::string save_image_path; // yin: for ngp flow
private:
	GPUMemory<int64_t> last_grid_frame; // yin: for ngp flow
	GPUMemory<int64_t> this_grid_frame; // yin: for ngp flow
	GPUMemory<bool>* grid_hit; // yin: for ngp flow
	struct QueueObj { // yin: for ngp flow
		__half* params;
		uint32_t* params_index;
		size_t params_size;
		__half* density_grid;
		uint32_t* density_grid_index;
		size_t density_grid_size;
		uint8_t* density_grid_bitfield;
		size_t density_grid_bitfield_size;
	};
	void FreeQueueObj(QueueObj obj) {
	if (obj.params != nullptr) cudaFree(obj.params);
	if (obj.params_index != nullptr) cudaFree(obj.params_index);
	if (obj.density_grid != nullptr) cudaFree(obj.density_grid);
	if (obj.density_grid_index != nullptr) cudaFree(obj.density_grid_index);
	if (obj.density_grid_bitfield != nullptr) cudaFree(obj.density_grid_bitfield);

	}
	std::queue<QueueObj> load_frame_queue; // yin: for ngp flow
	std::queue<QueueObj> diff_frame_queue; // yin: for ngp flow
	std::thread last_update_frame_thread; // yin: for ngp flow
	bool frame_data_enqueue(const fs::path& path, std::queue<QueueObj>& queue, bool read_compression); // yin: for ngp flow
	std::atomic<int64_t> read_frame_thread_counter = {0}; // yin: for ngp flow
public:
	void set_params(__half* params_gpu, size_t n, uint32_t* index_gpu = nullptr); // yin: for ngp flow
	void add_params(__half* params_gpu, size_t n, uint32_t* index_gpu = nullptr); // yin: for ngp flow
	void set_params_setframe(int64_t frame, __half* params_gpu, size_t n, uint32_t* index_gpu = nullptr); // yin: for ngp flow
	void add_params_setframe(int64_t frame, __half* params_gpu, size_t n, uint32_t* index_gpu = nullptr); // yin: for ngp flow
	void set_density_grid(__half* density_grid_gpu, size_t n, uint32_t* index_gpu = nullptr, uint8_t* bitfield_gpu = nullptr, size_t bit_n = 0); // yin: for ngp flow
	void join_last_update_frame_thread(); // yin: for ngp flow
// private:
// 	GPUMemory<__half> params_gpu; // yin: for ngp flow
// 	GPUMemory<uint32_t> params_index_gpu; // yin: for ngp flow
// 	GPUMemory<__half> density_grid_gpu; // yin: for ngp flow
// 	GPUMemory<uint32_t> density_grid_index_gpu; // yin: for ngp flow

public:
	void draw_visualizations(ImDrawList* list, const mat4x3& camera_matrix);
	void train_and_render(bool skip_rendering);
	fs::path training_data_path() const;
	void init_window(int resw, int resh, bool hidden = false, bool second_window = false);
	void destroy_window();
	void init_vr();
	void update_vr_performance_settings();
	void apply_camera_smoothing(float elapsed_ms);
	bool begin_frame();
	void handle_user_input();
	vec3 vr_to_world(const vec3& pos) const;
	void begin_vr_frame_and_handle_vr_input();
	void gather_histograms();
	void draw_gui();
	bool frame();
	bool want_repl();
	uint32_t n_dimensions_to_visualize() const;
	float fov() const ;
	void set_fov(float val) ;
	vec2 fov_xy() const ;
	void set_fov_xy(const vec2& val);
	void save_snapshot(const fs::path& path, bool include_optimizer_state, bool compress);
	void load_snapshot(nlohmann::json config);
	void load_snapshot(const fs::path& path);
	void load_snapshot(std::istream& stream, bool is_compressed = true);
	void update_loss_graph();

	void compute_and_save_marching_cubes_mesh(const fs::path& filename, ivec3 res3d = ivec3(128), BoundingBox aabb = {}, float thresh = 2.5f, bool unwrap_it = false);
	ivec3 compute_and_save_png_slices(const fs::path& filename, int res, BoundingBox aabb = {}, float thresh = 2.5f, float density_range = 4.f, bool flip_y_and_z_axes = false);

	fs::path root_dir();
	void set_root_dir(const fs::path& dir);

	////////////////////////////////////////////////////////////////
	// marching cubes related state
	struct MeshState {
		float thresh = 2.5f;
		int res = 256;
		bool unwrap = false;
		float smooth_amount = 2048.f;
		float density_amount = 128.f;
		float inflate_amount = 1.f;
		bool optimize_mesh = false;
		GPUMemory<vec3> verts;
		GPUMemory<vec3> vert_normals;
		GPUMemory<vec3> vert_colors;
		GPUMemory<vec4> verts_smoothed; // homogenous
		GPUMemory<uint32_t> indices;
		GPUMemory<vec3> verts_gradient;
		std::shared_ptr<TrainableBuffer<3, 1, float>> trainable_verts;
		std::shared_ptr<Optimizer<float>> verts_optimizer;

		void clear() {
			indices={};
			verts={};
			vert_normals={};
			vert_colors={};
			verts_smoothed={};
			verts_gradient={};
			trainable_verts=nullptr;
			verts_optimizer=nullptr;
		}
	};
	MeshState m_mesh;
	bool m_want_repl = false;

	bool m_render_window = false;
	bool m_gather_histograms = false;

	bool m_include_optimizer_state_in_snapshot = false;
	bool m_compress_snapshot = true;
	bool m_render_ground_truth = false;
	EGroundTruthRenderMode m_ground_truth_render_mode = EGroundTruthRenderMode::Shade;
	float m_ground_truth_alpha = 1.0f;

	bool m_train = false;
	bool m_training_data_available = false;
	bool m_render = true;
	int m_max_spp = 0;
	ETestbedMode m_testbed_mode = ETestbedMode::None;
	bool m_max_level_rand_training = false;

	// Rendering stuff
	ivec2 m_window_res = ivec2(0);
	bool m_dynamic_res = true;
	float m_dynamic_res_target_fps = 20.0f;
	int m_fixed_res_factor = 8;
	float m_scale = 1.0;
	float m_aperture_size = 0.0f;
	vec2 m_relative_focal_length = vec2(1.0f);
	uint32_t m_fov_axis = 1;
	float m_zoom = 1.f; // 2d zoom factor (for insets?)
	vec2 m_screen_center = vec2(0.5f); // center of 2d zoom

	float m_ndc_znear = 1.0f / 32.0f;
	float m_ndc_zfar = 128.0f;

	mat4x3 m_camera = mat4x3::identity();
	mat4x3 m_smoothed_camera = mat4x3::identity();
	size_t m_render_skip_due_to_lack_of_camera_movement_counter = 0;

	bool m_fps_camera = false;
	bool m_camera_smoothing = false;
	bool m_autofocus = false;
	vec3 m_autofocus_target = vec3(0.5f);

	vec3 m_up_dir = {0.0f, 1.0f, 0.0f};
	vec3 m_sun_dir = normalize(vec3(1.0f));
	float m_bounding_radius = 1;
	float m_exposure = 0.f;

	ERenderMode m_render_mode = ERenderMode::Shade;
	EMeshRenderMode m_mesh_render_mode = EMeshRenderMode::VertexNormals;

	uint32_t m_seed = 1337;

#ifdef NGP_GUI
	GLFWwindow* m_glfw_window = nullptr;
	struct SecondWindow {
		GLFWwindow* window = nullptr;
		GLuint program = 0;
		GLuint vao = 0, vbo = 0;
		void draw(GLuint texture);
	} m_second_window;

	float m_drag_depth = 1.0f;

	// The VAO will be empty, but we need a valid one for attribute-less rendering
	GLuint m_blit_vao = 0;
	GLuint m_blit_program = 0;

	void init_opengl_shaders();
	void blit_texture(const Foveation& foveation, GLint rgba_texture, GLint rgba_filter_mode, GLint depth_texture, GLint framebuffer, const ivec2& offset, const ivec2& resolution);

	void create_second_window();

	std::unique_ptr<OpenXRHMD> m_hmd;
	OpenXRHMD::FrameInfoPtr m_vr_frame_info;
	bool m_vr_use_depth_reproject = false;
	bool m_vr_use_hidden_area_mask = true;

	void set_n_views(size_t n_views);

	std::function<bool()> m_keyboard_event_callback;

	std::shared_ptr<GLTexture> m_pip_render_texture;
	std::vector<std::shared_ptr<GLTexture>> m_rgba_render_textures;
	std::vector<std::shared_ptr<GLTexture>> m_depth_render_textures;
#endif


	std::unique_ptr<CudaRenderBuffer> m_pip_render_buffer;

	SharedQueue<std::unique_ptr<ICallable>> m_task_queue;

	void redraw_gui_next_frame() {
		m_gui_redraw = true;
	}

	bool m_gui_redraw = true;

	struct Nerf {
		struct Training {
			NerfDataset dataset;
			int n_images_for_training = 0; // how many images to train from, as a high watermark compared to the dataset size
			int n_images_for_training_prev = 0; // how many images we saw last time we updated the density grid

			struct ErrorMap {
				GPUMemory<float> data;
				GPUMemory<float> cdf_x_cond_y;
				GPUMemory<float> cdf_y;
				GPUMemory<float> cdf_img;
				std::vector<float> pmf_img_cpu;
				ivec2 resolution = {16, 16};
				ivec2 cdf_resolution = {16, 16};
				bool is_cdf_valid = false;
			} error_map;

			std::vector<TrainingXForm> transforms;
			GPUMemory<TrainingXForm> transforms_gpu;

			std::vector<vec3> cam_pos_gradient;
			GPUMemory<vec3> cam_pos_gradient_gpu;

			std::vector<vec3> cam_rot_gradient;
			GPUMemory<vec3> cam_rot_gradient_gpu;

			GPUMemory<vec3> cam_exposure_gpu;
			std::vector<vec3> cam_exposure_gradient;
			GPUMemory<vec3> cam_exposure_gradient_gpu;

			vec2 cam_focal_length_gradient = vec2(0.0f);
			GPUMemory<vec2> cam_focal_length_gradient_gpu;

			std::vector<AdamOptimizer<vec3>> cam_exposure;
			std::vector<AdamOptimizer<vec3>> cam_pos_offset;
			std::vector<RotationAdamOptimizer> cam_rot_offset;
			AdamOptimizer<vec2> cam_focal_length_offset = AdamOptimizer<vec2>(0.0f);

			GPUMemory<float> extra_dims_gpu; // if the model demands a latent code per training image, we put them in here.
			GPUMemory<float> extra_dims_gradient_gpu;
			std::vector<VarAdamOptimizer> extra_dims_opt;

			std::vector<float> get_extra_dims_cpu(int trainview) const;

			float extrinsic_l2_reg = 1e-4f;
			float extrinsic_learning_rate = 1e-3f;

			float intrinsic_l2_reg = 1e-4f;
			float exposure_l2_reg = 0.0f;

			NerfCounters counters_rgb;

			bool random_bg_color = true;
			bool linear_colors = false;
			ELossType loss_type = ELossType::L2;
			ELossType depth_loss_type = ELossType::L1;
			bool snap_to_pixel_centers = true;
			bool train_envmap = false;

			bool optimize_distortion = false;
			bool optimize_extrinsics = false;
			bool optimize_extra_dims = false;
			bool optimize_focal_length = false;
			bool optimize_exposure = false;
			bool render_error_overlay = false;
			float error_overlay_brightness = 0.125f;
			uint32_t n_steps_between_cam_updates = 16;
			uint32_t n_steps_since_cam_update = 0;

			bool sample_focal_plane_proportional_to_error = false;
			bool sample_image_proportional_to_error = false;
			bool include_sharpness_in_error = false;
			uint32_t n_steps_between_error_map_updates = 128;
			uint32_t n_steps_since_error_map_update = 0;
			uint32_t n_rays_since_error_map_update = 0;

			float near_distance = 0.1f;
			float density_grid_decay = 0.95f;
			default_rng_t density_grid_rng;
			int view = 0;

			float depth_supervision_lambda = 0.f;

			GPUMemory<float> sharpness_grid;

			void set_camera_intrinsics(int frame_idx, float fx, float fy = 0.0f, float cx = -0.5f, float cy = -0.5f, float k1 = 0.0f, float k2 = 0.0f, float p1 = 0.0f, float p2 = 0.0f, float k3 = 0.0f, float k4 = 0.0f, bool is_fisheye = false);
			void set_camera_extrinsics_rolling_shutter(int frame_idx, mat4x3 camera_to_world_start, mat4x3 camera_to_world_end, const vec4& rolling_shutter, bool convert_to_ngp = true);
			void set_camera_extrinsics(int frame_idx, mat4x3 camera_to_world, bool convert_to_ngp = true);
			mat4x3 get_camera_extrinsics(int frame_idx);
			void update_transforms(int first = 0, int last = -1);
			void update_extra_dims();

#ifdef NGP_PYTHON
			void set_image(int frame_idx, pybind11::array_t<float> img, pybind11::array_t<float> depth_img, float depth_scale);
#endif

			void reset_camera_extrinsics();
			void export_camera_extrinsics(const fs::path& path, bool export_extrinsics_in_quat_format = true);
		} training = {};

		GPUMemory<float> density_grid; // NERF_GRIDSIZE()^3 grid of EMA smoothed densities from the network
		GPUMemory<uint8_t> density_grid_bitfield;
		uint8_t* get_density_grid_bitfield_mip(uint32_t mip);
		GPUMemory<float> density_grid_mean;
		uint32_t density_grid_ema_step = 0;

		uint32_t max_cascade = 0;

		ENerfActivation rgb_activation = ENerfActivation::Exponential;
		ENerfActivation density_activation = ENerfActivation::Exponential;

		vec3 light_dir = vec3(0.5f);
		// which training image's latent code should be used for rendering
		int rendering_extra_dims_from_training_view = 0;
		GPUMemory<float> rendering_extra_dims;

		void reset_extra_dims(default_rng_t &rng);
		const float* get_rendering_extra_dims(cudaStream_t stream) const;

		int show_accel = -1;

		float sharpen = 0.f;

		float cone_angle_constant = 1.f/256.f;

		bool visualize_cameras = false;
		bool render_with_lens_distortion = false;
		Lens render_lens = {};

		float render_min_transmittance = 0.01f;
		bool render_gbuffer_hard_edges = false;

		float glow_y_cutoff = 0.f;
		int glow_mode = 0;

		int find_closest_training_view(mat4x3 pose) const;
		void set_rendering_extra_dims_from_training_view(int trainview);
		void set_rendering_extra_dims(const std::vector<float>& vals);
		std::vector<float> get_rendering_extra_dims_cpu() const;
	} m_nerf;

	enum EDataType {
		Float,
		Half,
	};

	struct Image {
		GPUMemory<char> data;

		EDataType type = EDataType::Float;
		ivec2 resolution = ivec2(0);

		GPUMemory<vec2> render_coords;
		GPUMemory<vec3> render_out;

		struct Training {
			GPUMemory<float> positions_tmp;
			GPUMemory<vec2> positions;
			GPUMemory<vec3> targets;

			bool snap_to_pixel_centers = true;
			bool linear_colors = false;
		} training  = {};

		ERandomMode random_mode = ERandomMode::Stratified;
	} m_image;

	struct VolPayload {
		vec3 dir;
		vec4 col;
		uint32_t pixidx;
	};

	struct Volume {
		float albedo = 0.95f;
		float scattering = 0.f;
		float inv_distance_scale = 100.f;
		GPUMemory<char> nanovdb_grid;
		GPUMemory<uint8_t> bitgrid;
		float global_majorant = 1.f;
		vec3 world2index_offset = {0.0f, 0.0f, 0.0f};
		float world2index_scale = 1.f;

		struct Training {
			GPUMemory<vec3> positions = {};
			GPUMemory<vec4> targets = {};
		} training = {};

		// tracing state
		GPUMemory<vec3> pos[2] = {};
		GPUMemory<VolPayload> payload[2] = {};
		GPUMemory<uint32_t> hit_counter = {};
		GPUMemory<vec4> radiance_and_density;
	} m_volume;

	float m_camera_velocity = 1.0f;
	EColorSpace m_color_space = EColorSpace::Linear;
	ETonemapCurve m_tonemap_curve = ETonemapCurve::Identity;
	bool m_dlss = false;
	std::shared_ptr<IDlssProvider> m_dlss_provider;
	float m_dlss_sharpening = 0.0f;

	// 3D stuff
	float m_render_near_distance = 0.0f;
	float m_slice_plane_z = 0.0f;
	bool m_floor_enable = false;
	inline float get_floor_y() const { return m_floor_enable ? m_aabb.min.y + 0.001f : -10000.f; }
	BoundingBox m_raw_aabb;
	BoundingBox m_aabb;
	BoundingBox m_render_aabb;
	mat3 m_render_aabb_to_local = mat3::identity();

	mat4x3 crop_box(bool nerf_space) const;
	std::vector<vec3> crop_box_corners(bool nerf_space) const;
	void set_crop_box(mat4x3 m, bool nerf_space);

	// Rendering/UI bookkeeping
	Ema m_training_prep_ms = {EEmaType::Time, 100};
	Ema m_training_ms = {EEmaType::Time, 100};
	Ema m_render_ms = {EEmaType::Time, 100};
	// The frame contains everything, i.e. training + rendering + GUI and buffer swapping
	Ema m_frame_ms = {EEmaType::Time, 100};
	std::chrono::time_point<std::chrono::steady_clock> m_last_frame_time_point;
	std::chrono::time_point<std::chrono::steady_clock> m_last_gui_draw_time_point;
	std::chrono::time_point<std::chrono::steady_clock> m_training_start_time_point;
	vec4 m_background_color = {0.0f, 0.0f, 0.0f, 1.0f};

	bool m_vsync = false;
	bool m_render_transparency_as_checkerboard = false;

	// Visualization of neuron activations
	int m_visualized_dimension = -1;
	int m_visualized_layer = 0;

	struct View {
		std::shared_ptr<CudaRenderBuffer> render_buffer;
		ivec2 full_resolution = {1, 1};
		int visualized_dimension = 0;

		mat4x3 camera0 = mat4x3::identity();
		mat4x3 camera1 = mat4x3::identity();
		mat4x3 prev_camera = mat4x3::identity();

		Foveation foveation;
		Foveation prev_foveation;

		vec2 relative_focal_length;
		vec2 screen_center;

		CudaDevice* device = nullptr;
	};

	std::vector<View> m_views;
	ivec2 m_n_views = {1, 1};

	bool m_single_view = true;

	float m_picture_in_picture_res = 0.f; // if non zero, requests a small second picture :)

	struct ImGuiVars {
		static const uint32_t MAX_PATH_LEN = 1024;

		bool enabled = true; // tab to toggle
		char cam_path_path[MAX_PATH_LEN] = "cam.json";
		char extrinsics_path[MAX_PATH_LEN] = "extrinsics.json";
		char mesh_path[MAX_PATH_LEN] = "base.obj";
		char snapshot_path[MAX_PATH_LEN] = "base.ingp";
		char video_path[MAX_PATH_LEN] = "video.mp4";
	} m_imgui;

	fs::path m_root_dir = "";

	bool m_visualize_unit_cube = false;
	bool m_edit_render_aabb = false;
	bool m_edit_world_transform = true;

	bool m_snap_to_pixel_centers = false;

	vec3 m_parallax_shift = {0.0f, 0.0f, 0.0f}; // to shift the viewer's origin by some amount in camera space

	// CUDA stuff
	StreamAndEvent m_stream;

	// Hashgrid encoding analysis
	float m_quant_percent = 0.f;
	std::vector<LevelStats> m_level_stats;
	std::vector<LevelStats> m_first_layer_column_stats;
	uint32_t m_n_levels = 0;
	uint32_t m_n_features_per_level = 0;
	uint32_t m_histo_level = 0; // collect a histogram for this level
	uint32_t m_base_grid_resolution;
	float m_per_level_scale;
	float m_histo[257] = {};
	float m_histo_scale = 1.f;

	uint32_t m_training_step = 0;
	uint32_t m_training_batch_size = 1 << 18;
	Ema m_loss_scalar = {EEmaType::Time, 100};
	std::vector<float> m_loss_graph = std::vector<float>(256, 0.0f);
	size_t m_loss_graph_samples = 0;

	bool m_train_encoding = true;
	bool m_train_network = true;

	class CudaDevice {
	public:
		struct Data {
			GPUMemory<uint8_t> density_grid_bitfield;
			uint8_t* density_grid_bitfield_ptr;

			GPUMemory<network_precision_t> params;
			std::shared_ptr<Buffer2D<uint8_t>> hidden_area_mask;
		};

		CudaDevice(int id, bool is_primary);

		CudaDevice(const CudaDevice&) = delete;
		CudaDevice& operator=(const CudaDevice&) = delete;

		CudaDevice(CudaDevice&&) = default;
		CudaDevice& operator=(CudaDevice&&) = default;

		ScopeGuard device_guard();

		int id() const {
			return m_id;
		}

		bool is_primary() const {
			return m_is_primary;
		}

		std::string name() const {
			return cuda_device_name(m_id);
		}

		int compute_capability() const {
			return cuda_compute_capability(m_id);
		}

		cudaStream_t stream() const {
			return m_stream->get();
		}

		void wait_for(cudaStream_t stream) const {
			CUDA_CHECK_THROW(cudaEventRecord(m_primary_device_event.event, stream));
			m_stream->wait_for(m_primary_device_event.event);
		}

		void signal(cudaStream_t stream) const {
			m_stream->signal(stream);
		}

		const CudaRenderBufferView& render_buffer_view() const {
			return m_render_buffer_view;
		}

		void set_render_buffer_view(const CudaRenderBufferView& view) {
			m_render_buffer_view = view;
		}

		Data& data() const {
			return *m_data;
		}

		bool dirty() const {
			return m_dirty;
		}

		void set_dirty(bool value) {
			m_dirty = value;
		}

		void set_network(const std::shared_ptr<Network<float, network_precision_t>>& network);
		void set_nerf_network(const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network);

		const std::shared_ptr<Network<float, network_precision_t>>& network() const {
			return m_network;
		}

		const std::shared_ptr<NerfNetwork<network_precision_t>>& nerf_network() const {
			return m_nerf_network;
		}

		void clear() {
			m_data = std::make_unique<Data>();
			m_render_buffer_view = {};
			m_network = {};
			m_nerf_network = {};
			set_dirty(true);
		}

		template <class F>
		auto enqueue_task(F&& f) -> std::future<std::result_of_t<F()>> {
			if (is_primary()) {
				return std::async(std::launch::deferred, std::forward<F>(f));
			} else {
				return m_render_worker->enqueue_task(std::forward<F>(f));
			}
		}

	private:
		int m_id;
		bool m_is_primary;
		std::unique_ptr<StreamAndEvent> m_stream;
		struct Event {
			Event() {
				CUDA_CHECK_THROW(cudaEventCreate(&event));
			}

			~Event() {
				cudaEventDestroy(event);
			}

			Event(const Event&) = delete;
			Event& operator=(const Event&) = delete;
			Event(Event&& other) { *this = std::move(other); }
			Event& operator=(Event&& other) {
				std::swap(event, other.event);
				return *this;
			}

			cudaEvent_t event = {};
		};
		Event m_primary_device_event;
		std::unique_ptr<Data> m_data;
		CudaRenderBufferView m_render_buffer_view = {};

		std::shared_ptr<Network<float, network_precision_t>> m_network;
		std::shared_ptr<NerfNetwork<network_precision_t>> m_nerf_network;

		bool m_dirty = true;

		std::unique_ptr<ThreadPool> m_render_worker;
	};

	void sync_device(CudaRenderBuffer& render_buffer, CudaDevice& device);
	ScopeGuard use_device(cudaStream_t stream, CudaRenderBuffer& render_buffer, CudaDevice& device);
	void set_all_devices_dirty();

	std::vector<CudaDevice> m_devices;
	CudaDevice& primary_device() {
		return m_devices.front();
	}

	ThreadPool m_thread_pool;
	std::vector<std::future<void>> m_render_futures;

	bool m_use_aux_devices = false;
	bool m_foveated_rendering = false;
	bool m_dynamic_foveated_rendering = true;
	float m_foveated_rendering_full_res_diameter = 0.55f;
	float m_foveated_rendering_scaling = 1.0f;
	float m_foveated_rendering_max_scaling = 2.0f;
	bool m_foveated_rendering_visualize = false;

	fs::path m_data_path;
	fs::path m_network_config_path = "base.json";

	nlohmann::json m_network_config;

	default_rng_t m_rng;

	CudaRenderBuffer m_windowless_render_surface{std::make_shared<CudaSurface2D>()};

	uint32_t network_width(uint32_t layer) const;
	uint32_t network_num_forward_activations() const;

	// Network & training stuff
	std::shared_ptr<Loss<network_precision_t>> m_loss;
	std::shared_ptr<Optimizer<network_precision_t>> m_optimizer;
	std::shared_ptr<Encoding<network_precision_t>> m_encoding;
	std::shared_ptr<Network<float, network_precision_t>> m_network;
	std::shared_ptr<Trainer<float, network_precision_t, network_precision_t>> m_trainer;

	struct TrainableEnvmap {
		std::shared_ptr<Optimizer<float>> optimizer;
		std::shared_ptr<TrainableBuffer<4, 2, float>> envmap;
		std::shared_ptr<Trainer<float, float, float>> trainer;

		ivec2 resolution;
		ELossType loss_type;

		Buffer2DView<const vec4> inference_view() const {
			if (!envmap) {
				return {};
			}

			return {(const vec4*)envmap->inference_params(), resolution};
		}

		Buffer2DView<const vec4> view() const {
			if (!envmap) {
				return {};
			}

			return {(const vec4*)envmap->params(), resolution};
		}
	} m_envmap;

	struct TrainableDistortionMap {
		std::shared_ptr<Optimizer<float>> optimizer;
		std::shared_ptr<TrainableBuffer<2, 2, float>> map;
		std::shared_ptr<Trainer<float, float, float>> trainer;
		ivec2 resolution;

		Buffer2DView<const vec2> inference_view() const {
			if (!map) {
				return {};
			}

			return {(const vec2*)map->inference_params(), resolution};
		}

		Buffer2DView<const vec2> view() const {
			if (!map) {
				return {};
			}

			return {(const vec2*)map->params(), resolution};
		}
	} m_distortion;

	std::shared_ptr<NerfNetwork<network_precision_t>> m_nerf_network;
};

}
