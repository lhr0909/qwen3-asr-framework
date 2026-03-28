#pragma once

#include <vector>

namespace q3asr {

enum class DiarizationAggregationStrategy {
    Mean,
    Hamming,
    First,
};

struct diarization_window {
    float start_time_seconds = 0.0f;
    float frame_step_seconds = 0.0f;
    int num_frames = 0;
    int num_speakers = 0;
    std::vector<float> values;

    float extent_end_seconds() const {
        return start_time_seconds + frame_step_seconds * static_cast<float>(num_frames);
    }

    float at(int frame_index, int speaker_index) const {
        return values[static_cast<size_t>(frame_index) * static_cast<size_t>(num_speakers) +
                      static_cast<size_t>(speaker_index)];
    }

    float & at(int frame_index, int speaker_index) {
        return values[static_cast<size_t>(frame_index) * static_cast<size_t>(num_speakers) +
                      static_cast<size_t>(speaker_index)];
    }
};

struct diarization_result {
    diarization_window aggregated_scores;
    diarization_window binarized_scores;
    std::vector<int> local_to_global;
};

struct diarization_config {
    float step_seconds = 0.5f;
    float latency_seconds = 0.5f;
    float tau_active = 0.6f;
    float rho_update = 0.3f;
    float delta_new = 1.0f;
    float gamma = 3.0f;
    float beta = 10.0f;
    int max_speakers = 20;
    bool normalize_embedding_weights = false;
};

class DelayedDiarizationAggregation {
public:
    DelayedDiarizationAggregation(
        float step_seconds,
        float latency_seconds,
        DiarizationAggregationStrategy strategy = DiarizationAggregationStrategy::Hamming
    );

    int num_overlapping_windows() const { return num_overlapping_windows_; }

    diarization_window aggregate(const std::vector<diarization_window> & buffers) const;

private:
    float step_seconds_ = 0.0f;
    float latency_seconds_ = 0.0f;
    DiarizationAggregationStrategy strategy_ = DiarizationAggregationStrategy::Hamming;
    int num_overlapping_windows_ = 0;
};

class OnlineSpeakerClustering {
public:
    OnlineSpeakerClustering(float tau_active, float rho_update, float delta_new, int max_speakers);

    void reset();
    int num_known_speakers() const;

    std::vector<int> identify(
        const diarization_window & segmentation,
        const float * embeddings,
        int embedding_dim
    );

    diarization_window apply_mapping(
        const diarization_window & segmentation,
        const std::vector<int> & local_to_global
    ) const;

private:
    int next_free_center() const;
    int add_center(const float * embedding, int embedding_dim);
    void update_center(int center_index, const float * embedding, int embedding_dim);

    float tau_active_ = 0.0f;
    float rho_update_ = 0.0f;
    float delta_new_ = 0.0f;
    int max_speakers_ = 0;
    int embedding_dim_ = 0;
    std::vector<float> centers_;
    std::vector<bool> active_centers_;
    std::vector<bool> blocked_centers_;
};

class DiartDiarizer {
public:
    explicit DiartDiarizer(const diarization_config & config = {});

    void reset();

    const diarization_config & config() const { return config_; }

    diarization_window compute_overlap_aware_weights(const diarization_window & segmentation) const;

    std::vector<float> normalize_embeddings(
        const float * embeddings,
        int num_speakers,
        int embedding_dim,
        float target_norm = 1.0f
    ) const;

    diarization_result process_window(
        const diarization_window & segmentation,
        const float * embeddings,
        int embedding_dim
    );

private:
    diarization_window binarize(const diarization_window & scores) const;

    diarization_config config_{};
    OnlineSpeakerClustering clustering_;
    DelayedDiarizationAggregation prediction_aggregation_;
    std::vector<diarization_window> prediction_buffer_;
};

} // namespace q3asr
