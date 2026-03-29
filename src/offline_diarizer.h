#pragma once

#include "diarization_gguf.h"

#include <string>
#include <vector>

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
    std::string clustering_model_path;
};

struct offline_diarizer_problem {
    int num_chunks = 0;
    int num_frames = 0;
    int num_speakers = 0;
    int embedding_dim = 0;
    int num_clusters = 0;
    int min_clusters = 1;
    int max_clusters = 1;
    float min_active_ratio = 0.2f;
    std::vector<float> binary_segmentations;
    std::vector<float> embeddings;

    float segmentation_at(int chunk_index, int frame_index, int speaker_index) const;
    const float * embedding_ptr(int chunk_index, int speaker_index) const;
};

struct offline_diarizer_result {
    int num_chunks = 0;
    int num_speakers = 0;
    int num_clusters = 0;
    int embedding_dim = 0;
    std::vector<int> hard_clusters;
    std::vector<float> soft_clusters;
    std::vector<float> centroids;

    int hard_cluster_at(int chunk_index, int speaker_index) const;
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
    const diarization_gguf_model & clustering_model() const { return clustering_model_; }

    bool native_execution_available() const { return false; }
    bool native_clustering_available() const { return plda_ready_; }
    bool cluster(const offline_diarizer_problem & problem, offline_diarizer_result & result);
    std::string native_execution_gap() const;

private:
    void reset();
    bool parse_community1_config(const std::string & config_path);
    bool load_clustering_model(const std::string & path);

    offline_diarizer_assets assets_{};
    offline_diarizer_config config_{};
    diarization_gguf_model segmentation_model_{};
    diarization_gguf_model embedding_model_{};
    diarization_gguf_model clustering_model_{};
    std::vector<float> xvec_mean1_{};
    std::vector<float> xvec_mean2_{};
    std::vector<float> xvec_lda_{};
    std::vector<float> plda_mu_{};
    std::vector<float> plda_transform_{};
    std::vector<float> plda_phi_{};
    int xvec_input_dim_ = 0;
    int xvec_output_dim_ = 0;
    int plda_dim_ = 0;
    bool plda_ready_ = false;
    std::string error_msg_;
};

} // namespace q3asr
