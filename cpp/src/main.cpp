// US-LesionAssist: research-only ONNX Runtime C++ image inference demo.
// CPU ORT execution is deliberate: avoid claiming CUDA acceleration until verified.
#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string model = "exports/unet_v1.onnx";
    std::string image;
    std::string input_bin;
    std::string reference_logits;
    std::string output_dir = "cpp_results";
    double threshold = 0.5;
    int warmup = 10;
    int runs = 100;
};

static std::string help_text() {
    return R"(US-LesionAssist C++ ONNX Runtime inference (research only)
Usage:
  lesion_infer --model exports/unet_v1.onnx --image samples/demo.png
  lesion_infer --model exports/unet_v1.onnx --input-bin exports/parity_input.bin --reference-logits exports/parity_ort_logits.bin
Optional: --out cpp_results --threshold 0.5 --warmup 10 --runs 100
The model input is one NCHW grayscale float32 image of static spatial shape.
CPU inference; measurement excludes image I/O and preprocessing.
)";
}

static Options parse_args(int argc, char* argv[]) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << help_text();
            std::exit(0);
        }
        if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
        const std::string val = argv[++i];
        if (arg == "--model") opt.model = val;
        else if (arg == "--image") opt.image = val;
        else if (arg == "--input-bin") opt.input_bin = val;
        else if (arg == "--reference-logits") opt.reference_logits = val;
        else if (arg == "--out") opt.output_dir = val;
        else if (arg == "--threshold") opt.threshold = std::stod(val);
        else if (arg == "--warmup") opt.warmup = std::stoi(val);
        else if (arg == "--runs") opt.runs = std::stoi(val);
        else throw std::runtime_error("Unknown argument: " + arg);
    }
    if (opt.image.empty() == opt.input_bin.empty()) {
        throw std::runtime_error("Specify exactly one of --image or --input-bin");
    }
    if (opt.threshold <= 0.0 || opt.threshold >= 1.0) {
        throw std::runtime_error("Threshold must be strictly between 0 and 1");
    }
    if (opt.runs < 1 || opt.warmup < 0) {
        throw std::runtime_error("Runs >= 1 and warmup >= 0 required");
    }
    return opt;
}

static std::vector<float> read_binary_f32(const std::string& filename, std::size_t expected) {
    std::ifstream file(fs::u8path(filename), std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot read float32 binary: " + filename);
    const auto bytes = file.tellg();
    if (bytes != static_cast<std::streamoff>(expected * sizeof(float))) {
        throw std::runtime_error("Unexpected binary size: " + filename);
    }
    file.seekg(0, std::ios::beg);
    std::vector<float> result(expected);
    if (!file.read(reinterpret_cast<char*>(result.data()), bytes)) {
        throw std::runtime_error("Failed reading float32 data: " + filename);
    }
    return result;
}

static void write_binary_f32(const fs::path& path, const float* data, std::size_t elements) {
    std::ofstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to create " + path.string());
    file.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(elements * sizeof(float)));
}

static double percentile(std::vector<double> data, double q) {
    if (data.empty()) throw std::runtime_error("No benchmark samples");
    std::sort(data.begin(), data.end());
    const double pos = q * static_cast<double>(data.size() - 1);
    const auto low = static_cast<std::size_t>(std::floor(pos));
    const auto high = static_cast<std::size_t>(std::ceil(pos));
    return data[low] * (1.0 - (pos - low)) + data[high] * (pos - low);
}

int main(int argc, char* argv[]) {
    try {
        const Options opt = parse_args(argc, argv);
        const fs::path out_dir = fs::u8path(opt.output_dir);
        fs::create_directories(out_dir);
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "US-LesionAssist");
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        options.SetIntraOpNumThreads(4);
#ifdef _WIN32
        const auto model_path = fs::u8path(opt.model).wstring();
        Ort::Session session(env, model_path.c_str(), options);
#else
        Ort::Session session(env, opt.model.c_str(), options);
#endif
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            throw std::runtime_error("This demo expects one input and one output tensor");
        }
        const auto shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[0] != 1 || shape[1] != 1 || shape[2] <= 0 || shape[3] <= 0) {
            throw std::runtime_error("Expected static [1,1,H,W] ONNX input; export using export_onnx.py");
        }
        const int height = static_cast<int>(shape[2]);
        const int width = static_cast<int>(shape[3]);
        const std::size_t element_count = static_cast<std::size_t>(width) * height;
        std::vector<float> data;
        cv::Mat display;
        if (!opt.input_bin.empty()) {
            data = read_binary_f32(opt.input_bin, element_count);
            cv::Mat normalized(height, width, CV_32FC1, data.data());
            cv::Mat gray;
            normalized.convertTo(gray, CV_8U, 255.0);
            cv::cvtColor(gray, display, cv::COLOR_GRAY2BGR);
        } else {
            display = cv::imread(opt.image, cv::IMREAD_COLOR);
            if (display.empty()) throw std::runtime_error("Failed to load image; use a simple ASCII path, e.g. samples/demo.png");
            cv::Mat gray, resized;
            cv::cvtColor(display, gray, cv::COLOR_BGR2GRAY);
            cv::resize(gray, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
            cv::Mat float_image;
            resized.convertTo(float_image, CV_32FC1, 1.0 / 255.0);
            if (!float_image.isContinuous()) float_image = float_image.clone();
            data.assign(reinterpret_cast<const float*>(float_image.data),
                        reinterpret_cast<const float*>(float_image.data) + element_count);
        }
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 4> input_shape = {1, 1, height, width};
        Ort::Value input = Ort::Value::CreateTensor<float>(
            memory_info, data.data(), data.size(), input_shape.data(), input_shape.size());
        const char* input_names[] = {"images"};
        const char* output_names[] = {"logits"};

        auto infer = [&]() {
            return session.Run(Ort::RunOptions{nullptr}, input_names, &input, 1, output_names, 1);
        };
        for (int i = 0; i < opt.warmup; ++i) (void)infer();
        std::vector<double> times;
        times.reserve(static_cast<std::size_t>(opt.runs));
        for (int i = 0; i < opt.runs; ++i) {
            auto start = std::chrono::steady_clock::now();
            (void)infer();
            auto end = std::chrono::steady_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
        }
        auto outputs = infer();
        if (outputs.size() != 1 || !outputs[0].IsTensor()) {
            throw std::runtime_error("Model did not return one tensor");
        }
        const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        if (out_shape.size() != 4 || out_shape[0] != 1 || out_shape[1] != 1 ||
            out_shape[2] != height || out_shape[3] != width) {
            throw std::runtime_error("Unexpected ONNX output shape");
        }
        const float* logits = outputs[0].GetTensorData<float>();
        write_binary_f32(out_dir / "cpp_logits.bin", logits, element_count);
        const double logit_cutoff = std::log(opt.threshold / (1.0 - opt.threshold));
        cv::Mat mask(height, width, CV_8UC1, cv::Scalar(0));
        std::size_t selected = 0;
        for (int y = 0; y < height; ++y) {
            auto* row = mask.ptr<std::uint8_t>(y);
            for (int x = 0; x < width; ++x) {
                const bool active = logits[static_cast<std::size_t>(y) * width + x] >= logit_cutoff;
                row[x] = active ? 255 : 0;
                selected += active ? 1 : 0;
            }
        }
        cv::Mat full_mask, overlay;
        cv::resize(mask, full_mask, display.size(), 0, 0, cv::INTER_NEAREST);
        cv::Mat colored = display.clone();
        colored.setTo(cv::Scalar(45, 230, 60), full_mask);
        cv::addWeighted(display, 0.78, colored, 0.22, 0, overlay);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(full_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(overlay, contours, -1, cv::Scalar(20, 255, 20),
                         std::max(2, display.cols / 200));
        if (!cv::imwrite((out_dir / "mask.png").string(), full_mask) ||
            !cv::imwrite((out_dir / "overlay.png").string(), overlay)) {
            throw std::runtime_error("Failed writing output PNG images");
        }

        double max_abs_reference = -1.0;
        double mask_agreement = -1.0;
        if (!opt.reference_logits.empty()) {
            const auto ref = read_binary_f32(opt.reference_logits, element_count);
            std::size_t same = 0;
            max_abs_reference = 0.0;
            for (std::size_t k = 0; k < element_count; ++k) {
                max_abs_reference = std::max(max_abs_reference,
                                              std::abs(static_cast<double>(logits[k]) - ref[k]));
                same += (logits[k] >= logit_cutoff) == (ref[k] >= logit_cutoff) ? 1u : 0u;
            }
            mask_agreement = static_cast<double>(same) / element_count;
        }
        const double median_ms = percentile(times, 0.5);
        const double p95_ms = percentile(times, 0.95);
        const double area = static_cast<double>(selected) / element_count;
        std::ofstream report(out_dir / "benchmark.json");
        report << std::fixed << std::setprecision(8)
               << "{\n"
               << "  \"runtime\": \"onnxruntime-cpp-cpu\",\n"
               << "  \"input_height\": " << height << ",\n"
               << "  \"input_width\": " << width << ",\n"
               << "  \"threshold\": " << opt.threshold << ",\n"
               << "  \"mask_area_fraction\": " << area << ",\n"
               << "  \"warmup_runs\": " << opt.warmup << ",\n"
               << "  \"timed_runs\": " << opt.runs << ",\n"
               << "  \"median_inference_ms\": " << median_ms << ",\n"
               << "  \"p95_inference_ms\": " << p95_ms << ",\n"
               << "  \"max_abs_diff_vs_reference_logits\": " << max_abs_reference << ",\n"
               << "  \"binary_mask_agreement_vs_reference\": " << mask_agreement << ",\n"
               << "  \"benchmark_scope\": \"CPU ORT Run only; excludes image I/O, preprocessing and rendering\",\n"
               << "  \"disclaimer\": \"Research only; not clinically validated\"\n"
               << "}\n";
        std::cout << "ONNX Runtime C++ CPU inference complete: " << width << "x" << height << "\n"
                  << "Mask area: " << std::setprecision(2) << area * 100.0 << "%\n"
                  << "Median/p95 inference: " << median_ms << " / " << p95_ms << " ms\n"
                  << "Results: " << out_dir.string() << "/overlay.png, mask.png, benchmark.json\n";
        if (max_abs_reference >= 0.0) {
            std::cout << "Numerical parity max_abs_diff: " << max_abs_reference
                      << ", mask agreement: " << mask_agreement << "\n";
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n" << help_text();
        return 1;
    }
}
