#include "offline_diarizer.h"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace q3asr {

namespace {

bool file_exists(const std::string & path) {
    return std::filesystem::exists(path);
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

bool parse_float_after_key(const std::string & yaml, const std::string & key, float & out_value) {
    std::istringstream stream(yaml);
    std::string line;
    const std::string needle = key + ":";

    while (std::getline(stream, line)) {
        line = trim(line);
        if (line.rfind(needle, 0) != 0) {
            continue;
        }

        const std::string value_text = trim(line.substr(needle.size()));
        if (value_text.empty()) {
            return false;
        }

        try {
            out_value = std::stof(value_text);
            return true;
        } catch (...) {
            return false;
        }
    }

    return false;
}

std::string read_text_file(const std::string & path) {
    std::ifstream stream(path);
    if (!stream) {
        return {};
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

} // namespace

OfflineDiarizer::~OfflineDiarizer() {
    reset();
}

void OfflineDiarizer::reset() {
    free_diarization_gguf_model(segmentation_model_);
    free_diarization_gguf_model(embedding_model_);
    assets_ = {};
    config_ = {};
    error_msg_.clear();
}

bool OfflineDiarizer::parse_community1_config(const std::string & config_path) {
    const std::string yaml = read_text_file(config_path);
    if (yaml.empty()) {
        error_msg_ = "Failed to read Community-1 config: " + config_path;
        return false;
    }

    float value = 0.0f;
    if (!parse_float_after_key(yaml, "threshold", value)) {
        error_msg_ = "Failed to parse clustering threshold from Community-1 config";
        return false;
    }
    config_.clustering_threshold = value;

    if (!parse_float_after_key(yaml, "Fa", value)) {
        error_msg_ = "Failed to parse clustering Fa from Community-1 config";
        return false;
    }
    config_.clustering_fa = value;

    if (!parse_float_after_key(yaml, "Fb", value)) {
        error_msg_ = "Failed to parse clustering Fb from Community-1 config";
        return false;
    }
    config_.clustering_fb = value;

    if (!parse_float_after_key(yaml, "min_duration_off", value)) {
        error_msg_ = "Failed to parse segmentation min_duration_off from Community-1 config";
        return false;
    }
    config_.min_duration_off = value;

    return true;
}

bool OfflineDiarizer::load_community1(const offline_diarizer_assets & assets) {
    reset();
    assets_ = assets;

    const std::string config_path = assets.community1_bundle_dir + "/config.yaml";
    if (!file_exists(config_path)) {
        error_msg_ = "Missing Community-1 config.yaml in bundle directory";
        return false;
    }
    if (!file_exists(assets.segmentation_model_path)) {
        error_msg_ = "Missing official segmentation GGUF: " + assets.segmentation_model_path;
        return false;
    }
    if (!file_exists(assets.embedding_model_path)) {
        error_msg_ = "Missing official embedding GGUF: " + assets.embedding_model_path;
        return false;
    }
    if (!file_exists(assets.plda_model_path)) {
        error_msg_ = "Missing Community-1 PLDA model: " + assets.plda_model_path;
        return false;
    }
    if (!file_exists(assets.xvec_transform_path)) {
        error_msg_ = "Missing Community-1 xvec transform: " + assets.xvec_transform_path;
        return false;
    }

    if (!parse_community1_config(config_path)) {
        return false;
    }

    DiarizationGGUFLoader loader;
    if (!loader.load(assets.segmentation_model_path, segmentation_model_)) {
        error_msg_ = loader.get_error();
        return false;
    }
    if (!loader.load(assets.embedding_model_path, embedding_model_)) {
        error_msg_ = loader.get_error();
        return false;
    }

    if (
        segmentation_model_.serialization_format != "pytorch" ||
        segmentation_model_.kind != "speaker-segmentation" ||
        segmentation_model_.source_repo != "pyannote/segmentation-3.0"
    ) {
        error_msg_ = "Community-1 offline diarizer requires the official pyannote segmentation GGUF";
        return false;
    }

    if (
        embedding_model_.serialization_format != "pytorch" ||
        embedding_model_.kind != "speaker-embedding" ||
        embedding_model_.source_repo != "pyannote/wespeaker-voxceleb-resnet34-LM"
    ) {
        error_msg_ = "Community-1 offline diarizer requires the official pyannote wespeaker GGUF";
        return false;
    }

    return true;
}

std::string OfflineDiarizer::native_execution_gap() const {
    return "Missing native segmentation forward, native speaker-embedding forward, PLDA scoring, and VBx decoding.";
}

} // namespace q3asr
