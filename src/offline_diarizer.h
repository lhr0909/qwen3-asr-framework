#pragma once

#include "diarization_gguf.h"

#include <string>

namespace q3asr {

struct offline_diarizer_config {
    float clustering_threshold = 0.6f;
    float clustering_fa = 0.07f;
    float clustering_fb = 0.8f;
    float min_duration_off = 0.0f;
};

struct offline_diarizer_assets {
    std::string community1_bundle_dir;
    std::string segmentation_model_path;
    std::string embedding_model_path;
    std::string plda_model_path;
    std::string xvec_transform_path;
};

class OfflineDiarizer {
public:
    ~OfflineDiarizer();

    bool load_community1(const offline_diarizer_assets & assets);

    const std::string & get_error() const { return error_msg_; }
    const offline_diarizer_assets & assets() const { return assets_; }
    const offline_diarizer_config & config() const { return config_; }
    const diarization_gguf_model & segmentation_model() const { return segmentation_model_; }
    const diarization_gguf_model & embedding_model() const { return embedding_model_; }

    bool native_execution_available() const { return false; }
    std::string native_execution_gap() const;

private:
    void reset();
    bool parse_community1_config(const std::string & config_path);

    offline_diarizer_assets assets_{};
    offline_diarizer_config config_{};
    diarization_gguf_model segmentation_model_{};
    diarization_gguf_model embedding_model_{};
    std::string error_msg_;
};

} // namespace q3asr
