/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/** @file   main.cu
 *  @author Thomas Müller, NVIDIA
 */

#include <neural-graphics-primitives/testbed.h>

#include <tiny-cuda-nn/common.h>

#include <args/args.hxx>

#include <filesystem/path.h>

using namespace args;
using namespace ngp;
using namespace std;

namespace ngp {

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

int main_func(const std::vector<std::string>& arguments) {
	ArgumentParser parser{
		"Instant Neural Graphics Primitives\n"
		"Version " NGP_VERSION,
		"",
	};

	HelpFlag help_flag{
		parser,
		"HELP",
		"Display this help menu.",
		{'h', "help"},
	};

	ValueFlag<string> network_config_flag{
		parser,
		"CONFIG",
		"Path to the network config. Uses the scene's default if unspecified.",
		{'n', 'c', "network", "config"},
	};

	Flag vr_flag{
		parser,
		"VR",
		"Enables VR",
		{"vr"}
	};

	ValueFlag<string> snapshot_flag{
		parser,
		"SNAPSHOT",
		"Optional snapshot to load upon startup.",
		{"snapshot", "load_snapshot"},
	};

	ValueFlag<uint64_t> width_flag{
		parser,
		"WIDTH",
		"Resolution width of the GUI.",
		{"width"},
	};

	ValueFlag<uint64_t> height_flag{
		parser,
		"HEIGHT",
		"Resolution height of the GUI.",
		{"height"},
	};

	Flag version_flag{
		parser,
		"VERSION",
		"Display the version of instant neural graphics primitives.",
		{'v', "version"},
	};

	PositionalList<string> files{
		parser,
		"files",
		"Files to be loaded. Can be a scene, network config, snapshot, camera path, or a combination of those.",
	};

	ValueFlag<string> init_flag{
		parser,
		"INIT",
		"The bson intra frame for init.",
		{"init"},
	};

	ValueFlag<int64_t> start_flag{
		parser,
		"START",
		"The start frame number.",
		{"start"},
	};

	ValueFlag<int64_t> end_flag{
		parser,
		"START",
		"The end frame number.",
		{"end"},
	};

	ValueFlag<string> frameformat_flag{
		parser,
		"FRAMEFPRMAT",
		"The path format of exported video frames (.bson).",
		{"frameformat"},
	};

	Flag diff_flag{
		parser,
		"DIFF",
		"Use diff mode.",
		{"diff"},
	};

	Flag nocompress_flag{
		parser,
		"NOCOMPRESS",
		"BSOM files are not compressed.",
		{"nocompress"},
	};

	ValueFlag<string> savecam_flag{
		parser,
		"SAVECAM",
		"Path to save camera.",
		{"savecam"},
	};

	Flag trainingcam_flag{
		parser,
		"TRAININGCAM",
		"A special mode to export training view cams.",
		{"trainingcam"},
	};

	// Parse command line arguments and react to parsing
	// errors using exceptions.
	try {
		if (arguments.empty()) {
			tlog::error() << "Number of arguments must be bigger than 0.";
			return -3;
		}

		parser.Prog(arguments.front());
		parser.ParseArgs(begin(arguments) + 1, end(arguments));
	} catch (const Help&) {
		cout << parser;
		return 0;
	} catch (const ParseError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -1;
	} catch (const ValidationError& e) {
		cerr << e.what() << endl;
		cerr << parser;
		return -2;
	}

	if (version_flag) {
		tlog::none() << "Instant Neural Graphics Primitives v" NGP_VERSION;
		return 0;
	}

	Testbed testbed;

	for (auto file : get(files)) {
		testbed.load_file(file);
	}

	if (snapshot_flag) {
		testbed.load_snapshot(static_cast<fs::path>(get(snapshot_flag)));
	} else if (network_config_flag) {
		testbed.reload_network_from_file(get(network_config_flag));
	}

	std::ofstream cam_out;
	if (savecam_flag) {
		std::ofstream f{get(savecam_flag), std::ios::out | std::ios::binary};
		cam_out = std::move(f);
	}

	testbed.m_train = false;

	testbed.init_window(width_flag ? get(width_flag) : 1920, height_flag ? get(height_flag) : 1080);

	if (vr_flag) {
		testbed.init_vr();
	}

	if (nocompress_flag) {
		testbed.read_compression = false;
	}

	// testbed.set_params_load_cache_size(1048576);
	// testbed.set_density_grid_load_cache_size(1048576);
	if (!init_flag || !start_flag || !end_flag || !frameformat_flag) {
		throw std::runtime_error("This is a player! Please specify --init, --start, --end and --frameformat!");
	}
	string init = get(init_flag);
	int64_t start = get(start_flag);
	int64_t end = get(end_flag);
	string frameformat = get(frameformat_flag);
	int64_t current = end;
	std::vector<int64_t> frame_sequence(testbed.max_read_frame_thread_n * 2);
	int64_t current_loading = 0;
	int64_t current_display = 0;
	auto last_frame_time = std::chrono::steady_clock::now();
	// Render/training loop
	while (testbed.frame()) {
		if (current >= end) {
			if (testbed.load_frame_enqueue(init)) {
				frame_sequence[current_loading] = start - 1;
				current_loading = (current_loading + 1) % frame_sequence.size();
				last_frame_time = std::chrono::steady_clock::now();
				tlog::info() << "ok diff_frame_enqueue init" << ' ' << init;
				current = start;
			}
		} else {
			std::string path = string_sprintf(frameformat.c_str(), current);
			if (diff_flag) {
				if (testbed.diff_frame_enqueue(path)) {
					frame_sequence[current_loading] = current;
					current_loading = (current_loading + 1) % frame_sequence.size();
					auto now = std::chrono::steady_clock::now();
					auto time_period = std::chrono::duration<float>(now - last_frame_time).count();
					current++;
					tlog::info() << (current - start) / time_period << "FPS ok diff_frame_enqueue" << ' ' << path;
				}
			} else {
				if (testbed.load_frame_enqueue(path)) {
					frame_sequence[current_loading] = current;
					current_loading = (current_loading + 1) % frame_sequence.size();
					auto now = std::chrono::steady_clock::now();
					auto time_period = std::chrono::duration<float>(now - last_frame_time).count();
					current++;
					tlog::info() << (current - start) / time_period << "FPS ok load_frame_enqueue" << ' ' << path;
				}
			}
		}

		if (savecam_flag) {
			// yin: get training view
			if (trainingcam_flag) {
				testbed.m_nerf.training.view++;
				testbed.set_camera_to_training_view(testbed.m_nerf.training.view);
				if (testbed.m_nerf.training.view >= testbed.m_nerf.training.dataset.n_images + 1) return 0;
			}
			nlohmann::json cameras_json;
			cameras_json["views"] = testbed.dump_views();
			cameras_json["camera"] = testbed.dump_camera();
			cameras_json["frame"] = frame_sequence[current_display];
			if (savecam_flag) cam_out << cameras_json.dump() << endl;
			else tlog::info() << cameras_json.dump();
		}

		auto start = std::chrono::steady_clock::now();
		if (testbed.diff_frame_dequeue()) {
			current_display = (current_display + 1) % frame_sequence.size();
			auto end = std::chrono::steady_clock::now();
			tlog::info() << std::chrono::duration<float>(end - start).count() << "s ok diff_frame_dequeue";
		}
		if (testbed.load_frame_dequeue()) {
			current_display = (current_display + 1) % frame_sequence.size();
			auto end = std::chrono::steady_clock::now();
			tlog::info() << std::chrono::duration<float>(end - start).count() << "s ok load_frame_dequeue";
		}
		testbed.reset_accumulation();
	}
	testbed.join_last_update_frame_thread();

	return 0;
}

}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
	SetConsoleOutputCP(CP_UTF8);
#else
int main(int argc, char* argv[]) {
#endif
	try {
		std::vector<std::string> arguments;
		for (int i = 0; i < argc; ++i) {
#ifdef _WIN32
			arguments.emplace_back(ngp::utf16_to_utf8(argv[i]));
#else
			arguments.emplace_back(argv[i]);
#endif
		}

		return ngp::main_func(arguments);
	} catch (const exception& e) {
		tlog::error() << fmt::format("Uncaught exception: {}", e.what());
		return 1;
	}
}
