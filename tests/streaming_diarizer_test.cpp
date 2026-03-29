#include "streaming_diarizer.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

bool approx_equal(float lhs, float rhs, float epsilon = 1.0e-4f) {
    return std::fabs(lhs - rhs) <= epsilon;
}

q3asr::diarization_window make_constant_window(
    float start_time_seconds,
    float frame_step_seconds,
    int num_frames,
    int num_speakers,
    float value
) {
    q3asr::diarization_window window;
    window.start_time_seconds = start_time_seconds;
    window.frame_step_seconds = frame_step_seconds;
    window.num_frames = num_frames;
    window.num_speakers = num_speakers;
    window.values.assign(static_cast<size_t>(num_frames) * static_cast<size_t>(num_speakers), value);
    return window;
}

bool test_overlap_penalty() {
    q3asr::diarization_config config;
    config.gamma = 1.0f;
    config.beta = 1.0f;
    q3asr::StreamingDiarizer diarizer(config);

    q3asr::diarization_window segmentation;
    segmentation.start_time_seconds = 0.0f;
    segmentation.frame_step_seconds = 0.1f;
    segmentation.num_frames = 2;
    segmentation.num_speakers = 2;
    segmentation.values = {
        0.9f, 0.1f,
        0.5f, 0.5f,
    };

    const q3asr::diarization_window weights = diarizer.compute_overlap_aware_weights(segmentation);
    bool ok = true;
    for (int frame = 0; frame < segmentation.num_frames; ++frame) {
        const float a = segmentation.at(frame, 0);
        const float b = segmentation.at(frame, 1);
        const float exp_a = std::exp(a);
        const float exp_b = std::exp(b);
        const float softmax_a = exp_a / (exp_a + exp_b);
        const float softmax_b = exp_b / (exp_a + exp_b);
        ok = ok && approx_equal(weights.at(frame, 0), a * softmax_a);
        ok = ok && approx_equal(weights.at(frame, 1), b * softmax_b);
        ok = ok && weights.at(frame, 0) >= 1.0e-8f;
        ok = ok && weights.at(frame, 1) >= 1.0e-8f;
    }
    return ok;
}

bool test_embedding_normalization() {
    q3asr::StreamingDiarizer diarizer;
    const float embeddings[] = {
        3.0f, 4.0f,
        0.0f, 2.0f,
    };

    const std::vector<float> normalized = diarizer.normalize_embeddings(embeddings, 2, 2);
    const float norm0 = std::sqrt(normalized[0] * normalized[0] + normalized[1] * normalized[1]);
    const float norm1 = std::sqrt(normalized[2] * normalized[2] + normalized[3] * normalized[3]);

    return approx_equal(norm0, 1.0f) &&
           approx_equal(norm1, 1.0f) &&
           approx_equal(normalized[0], 0.6f) &&
           approx_equal(normalized[1], 0.8f) &&
           approx_equal(normalized[2], 0.0f) &&
           approx_equal(normalized[3], 1.0f);
}

bool test_delayed_aggregation() {
    const int frames = 500;
    const int speakers = 2;
    const float duration = 5.0f;
    const float step = 0.5f;
    const float latency = 2.0f;
    const float frame_step = duration / static_cast<float>(frames);

    q3asr::DelayedDiarizationAggregation mean(step, latency, q3asr::DiarizationAggregationStrategy::Mean);
    q3asr::DelayedDiarizationAggregation hamming(step, latency, q3asr::DiarizationAggregationStrategy::Hamming);
    q3asr::DelayedDiarizationAggregation first(step, latency, q3asr::DiarizationAggregationStrategy::First);

    std::vector<q3asr::diarization_window> buffers;
    for (int index = 0; index < mean.num_overlapping_windows(); ++index) {
        q3asr::diarization_window window = make_constant_window(
            (10.0f + static_cast<float>(index)) * step,
            frame_step,
            frames,
            speakers,
            static_cast<float>(index + 1)
        );
        buffers.push_back(std::move(window));
    }

    const q3asr::diarization_window mean_output = mean.aggregate(buffers);
    const q3asr::diarization_window hamming_output = hamming.aggregate(buffers);
    const q3asr::diarization_window first_output = first.aggregate(buffers);

    bool ok = true;
    ok = ok && mean.num_overlapping_windows() == 4;
    ok = ok && hamming.num_overlapping_windows() == 4;
    ok = ok && first.num_overlapping_windows() == 4;
    ok = ok && mean_output.num_frames == 51;
    ok = ok && hamming_output.num_frames == 51;
    ok = ok && first_output.num_frames == 51;

    for (float value : mean_output.values) {
        ok = ok && approx_equal(value, 2.5f);
    }
    for (float value : hamming_output.values) {
        ok = ok && value > 2.5f;
        ok = ok && value < 4.0f;
    }
    for (float value : first_output.values) {
        ok = ok && approx_equal(value, 1.0f);
    }

    if (!ok) {
        std::cerr << "mean_frames=" << mean_output.num_frames << "\n";
        std::cerr << "hamming_frames=" << hamming_output.num_frames << "\n";
        std::cerr << "first_frames=" << first_output.num_frames << "\n";
        if (!mean_output.values.empty()) {
            std::cerr << "mean_first=" << mean_output.values.front() << "\n";
        }
        if (!hamming_output.values.empty()) {
            std::cerr << "hamming_first=" << hamming_output.values.front() << "\n";
        }
        if (!first_output.values.empty()) {
            std::cerr << "first_first=" << first_output.values.front() << "\n";
        }
    }

    return ok;
}

bool test_online_clustering() {
    q3asr::OnlineSpeakerClustering clustering(0.6f, 0.3f, 0.2f, 4);

    q3asr::diarization_window first_seg;
    first_seg.start_time_seconds = 0.0f;
    first_seg.frame_step_seconds = 0.1f;
    first_seg.num_frames = 4;
    first_seg.num_speakers = 2;
    first_seg.values = {
        0.9f, 0.1f,
        0.9f, 0.1f,
        0.9f, 0.1f,
        0.9f, 0.1f,
    };
    const float first_embeddings[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    const std::vector<int> first_map = clustering.identify(first_seg, first_embeddings, 2);
    bool ok = first_map.size() == 2 && first_map[0] == 0 && first_map[1] == -1 && clustering.num_known_speakers() == 1;

    q3asr::diarization_window second_seg = first_seg;
    second_seg.start_time_seconds = 0.5f;
    second_seg.values = {
        0.95f, 0.92f,
        0.95f, 0.92f,
        0.95f, 0.92f,
        0.95f, 0.92f,
    };
    const float second_embeddings[] = {
        0.0f, 1.0f,
        1.0f, 0.0f,
    };

    const std::vector<int> second_map = clustering.identify(second_seg, second_embeddings, 2);
    ok = ok && second_map.size() == 2 && second_map[0] == 1 && second_map[1] == 0;
    ok = ok && clustering.num_known_speakers() == 2;
    return ok;
}

bool test_pipeline() {
    q3asr::diarization_config config;
    config.step_seconds = 0.5f;
    config.latency_seconds = 1.0f;
    config.tau_active = 0.6f;
    config.rho_update = 0.3f;
    config.delta_new = 0.2f;
    config.max_speakers = 4;

    q3asr::StreamingDiarizer diarizer(config);

    q3asr::diarization_window seg1;
    seg1.start_time_seconds = 0.0f;
    seg1.frame_step_seconds = 0.1f;
    seg1.num_frames = 10;
    seg1.num_speakers = 2;
    seg1.values.assign(20, 0.0f);
    for (int frame = 0; frame < seg1.num_frames; ++frame) {
        seg1.at(frame, 0) = 0.95f;
    }
    const float emb1[] = {
        1.0f, 0.0f,
        0.0f, 1.0f,
    };

    const q3asr::diarization_result first = diarizer.process_window(seg1, emb1, 2);
    bool ok = first.local_to_global.size() == 2 && first.local_to_global[0] == 0;
    ok = ok && first.binarized_scores.num_speakers == 4;

    q3asr::diarization_window seg2 = seg1;
    seg2.start_time_seconds = 0.5f;
    for (int frame = 0; frame < seg2.num_frames; ++frame) {
        seg2.at(frame, 0) = 0.91f;
        seg2.at(frame, 1) = 0.93f;
    }
    const float emb2[] = {
        0.0f, 1.0f,
        1.0f, 0.0f,
    };

    const q3asr::diarization_result second = diarizer.process_window(seg2, emb2, 2);
    ok = ok && second.local_to_global.size() == 2;
    ok = ok && second.local_to_global[0] == 1;
    ok = ok && second.local_to_global[1] == 0;
    ok = ok && second.aggregated_scores.num_speakers == 4;

    float speaker0_energy = 0.0f;
    float speaker1_energy = 0.0f;
    for (int frame = 0; frame < second.aggregated_scores.num_frames; ++frame) {
        speaker0_energy += second.aggregated_scores.at(frame, 0);
        speaker1_energy += second.aggregated_scores.at(frame, 1);
    }
    ok = ok && speaker0_energy > 0.0f;
    ok = ok && speaker1_energy > 0.0f;
    return ok;
}

} // namespace

int main() {
    const bool overlap_ok = test_overlap_penalty();
    const bool normalize_ok = test_embedding_normalization();
    const bool aggregation_ok = test_delayed_aggregation();
    const bool clustering_ok = test_online_clustering();
    const bool pipeline_ok = test_pipeline();
    const bool ok = overlap_ok && normalize_ok && aggregation_ok && clustering_ok && pipeline_ok;

    if (!ok) {
        std::cerr << "overlap_ok=" << overlap_ok << "\n";
        std::cerr << "normalize_ok=" << normalize_ok << "\n";
        std::cerr << "aggregation_ok=" << aggregation_ok << "\n";
        std::cerr << "clustering_ok=" << clustering_ok << "\n";
        std::cerr << "pipeline_ok=" << pipeline_ok << "\n";
        std::cerr << "streaming_diarizer_test failed\n";
        return 1;
    }

    std::cout << "streaming_diarizer_test passed\n";
    return 0;
}
