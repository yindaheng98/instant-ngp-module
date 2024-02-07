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

	ValueFlag<string> mode_flag{
		parser,
		"MODE",
		"Deprecated. Do not use.",
		{'m', "mode"},
	};

	ValueFlag<string> network_config_flag{
		parser,
		"CONFIG",
		"Path to the network config. Uses the scene's default if unspecified.",
		{'n', 'c', "network", "config"},
	};

	Flag no_gui_flag{
		parser,
		"NO_GUI",
		"Disables the GUI and instead reports training progress on the command line.",
		{"no-gui"},
	};

	Flag vr_flag{
		parser,
		"VR",
		"Enables VR",
		{"vr"}
	};

	Flag no_train_flag{
		parser,
		"NO_TRAIN",
		"Disables training on startup.",
		{"no-train"},
	};

	ValueFlag<string> scene_flag{
		parser,
		"SCENE",
		"The scene to load. Can be NeRF dataset, a *.obj/*.stl mesh for training a SDF, an image, or a *.nvdb volume.",
		{'s', "scene"},
	};

	ValueFlag<string> snapshot_flag{
		parser,
		"SNAPSHOT",
		"Optional snapshot to load upon startup.",
		{"snapshot", "load_snapshot"},
	};

	ValueFlag<uint32_t> width_flag{
		parser,
		"WIDTH",
		"Resolution width of the GUI.",
		{"width"},
	};

	ValueFlag<uint32_t> height_flag{
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

	ValueFlag<uint32_t> start_flag{
		parser,
		"START",
		"The start frame number.",
		{"start"},
	};

	ValueFlag<uint32_t> end_flag{
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

	if (mode_flag) {
		tlog::warning() << "The '--mode' argument is no longer in use. It has no effect. The mode is automatically chosen based on the scene.";
	}

	Testbed testbed;

	for (auto file : get(files)) {
		testbed.load_file(file);
	}

	if (scene_flag) {
		testbed.load_training_data(get(scene_flag));
	}

	if (snapshot_flag) {
		testbed.load_snapshot(static_cast<fs::path>(get(snapshot_flag)));
	} else if (network_config_flag) {
		testbed.reload_network_from_file(get(network_config_flag));
	}

	testbed.m_train = !no_train_flag;

#ifdef NGP_GUI
	bool gui = !no_gui_flag;
#else
	bool gui = false;
#endif

	if (gui) {
		testbed.init_window(width_flag ? get(width_flag) : 1920, height_flag ? get(height_flag) : 1080);
	}

	if (vr_flag) {
		testbed.init_vr();
	}

	// testbed.set_params_load_cache_size(1048576);
	// testbed.set_density_grid_load_cache_size(1048576);
	if (!init_flag || !start_flag || !end_flag || !frameformat_flag) {
		throw std::runtime_error("This is a player! Please specify --init, --start, --end and --frameformat!");
	}
	string init = get(init_flag);
	uint32_t start = get(start_flag);
	uint32_t end = get(end_flag);
	string frameformat = get(frameformat_flag);
	uint32_t current = end;
	auto last_frame_time = std::chrono::steady_clock::now();
	// Render/training loop
	while (testbed.frame()) {
		if (!gui) {
			tlog::info() << "iteration=" << testbed.m_training_step << " loss=" << testbed.m_loss_scalar.val();
		}
		if (current >= end) {
			if (testbed.load_frame_enqueue(init)) {
				last_frame_time = std::chrono::steady_clock::now();
				tlog::info() << "ok diff_frame_enqueue init" << ' ' << init;
				current = start;
			}
		} else {
			std::string path = string_sprintf(frameformat.c_str(), current);
			if (diff_flag) {
				if (testbed.diff_frame_enqueue(path)) {
					auto now = std::chrono::steady_clock::now();
					auto time_period = std::chrono::duration<float>(now - last_frame_time).count();
					current++;
					tlog::info() << (current - start) / time_period << "FPS ok diff_frame_enqueue" << ' ' << path;
				}
			} else {
				if (testbed.load_frame_enqueue(path)) {
					auto now = std::chrono::steady_clock::now();
					auto time_period = std::chrono::duration<float>(now - last_frame_time).count();
					current++;
					tlog::info() << (current - start) / time_period << "FPS ok load_frame_enqueue" << ' ' << path;
				}
			}
		}
		auto start = std::chrono::steady_clock::now();
		if (testbed.diff_frame_dequeue()) {
			auto end = std::chrono::steady_clock::now();
			tlog::info() << std::chrono::duration<float>(end - start).count() << "s ok diff_frame_dequeue";
		}
		if (testbed.load_frame_dequeue()) {
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
