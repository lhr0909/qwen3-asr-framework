#include "streaming_diarizer.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace q3asr {

namespace {

constexpr float kInvalidCost = 1.0e10f;
constexpr float kMinWeight = 1.0e-8f;
constexpr float kEpsilon = 1.0e-6f;
constexpr float kPi = 3.14159265358979323846f;

void validate_window(const diarization_window & window) {
    if (window.frame_step_seconds <= 0.0f) {
        throw std::invalid_argument("diarization window frame step must be positive");
    }
    if (window.num_frames < 0 || window.num_speakers < 0) {
        throw std::invalid_argument("diarization window dimensions must be non-negative");
    }
    const size_t expected_size =
        static_cast<size_t>(window.num_frames) * static_cast<size_t>(window.num_speakers);
    if (window.values.size() != expected_size) {
        throw std::invalid_argument("diarization window payload does not match its dimensions");
    }
}

bool contains_nan(const float * values, int count) {
    for (int i = 0; i < count; ++i) {
        if (std::isnan(values[i])) {
            return true;
        }
    }
    return false;
}

std::vector<float> softmax_by_frame_ggml(
    const std::vector<float> & values,
    int num_frames,
    int num_speakers,
    float beta
) {
    if (num_frames == 0 || num_speakers == 0) {
        return {};
    }

    const size_t bytes = values.size() * sizeof(float);
    const size_t tensor_bytes = bytes * 2;
    const size_t meta_bytes =
        ggml_tensor_overhead() * 4 + ggml_graph_overhead() + ggml_graph_overhead_custom(16, false);
    ggml_init_params params = {
        tensor_bytes + meta_bytes + 4096,
        nullptr,
        false,
    };

    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        throw std::runtime_error("failed to allocate ggml context for diarization softmax");
    }

    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, num_speakers, num_frames);
    ggml_tensor * scaled = ggml_scale(ctx, input, beta);
    ggml_tensor * probs = ggml_soft_max(ctx, scaled);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, probs);

    std::memcpy(input->data, values.data(), bytes);
    ggml_graph_compute_with_ctx(ctx, graph, 1);

    std::vector<float> result(values.size());
    std::memcpy(result.data(), probs->data, bytes);
    ggml_free(ctx);
    return result;
}

std::vector<int> hungarian_rows_le_cols(
    const std::vector<float> & costs,
    int num_rows,
    int num_cols
) {
    std::vector<double> u(static_cast<size_t>(num_rows) + 1, 0.0);
    std::vector<double> v(static_cast<size_t>(num_cols) + 1, 0.0);
    std::vector<int> p(static_cast<size_t>(num_cols) + 1, 0);
    std::vector<int> way(static_cast<size_t>(num_cols) + 1, 0);

    for (int row = 1; row <= num_rows; ++row) {
        p[0] = row;
        int col0 = 0;
        std::vector<double> minv(static_cast<size_t>(num_cols) + 1, std::numeric_limits<double>::infinity());
        std::vector<bool> used(static_cast<size_t>(num_cols) + 1, false);

        do {
            used[static_cast<size_t>(col0)] = true;
            const int row0 = p[static_cast<size_t>(col0)];
            double delta = std::numeric_limits<double>::infinity();
            int col1 = 0;

            for (int col = 1; col <= num_cols; ++col) {
                if (used[static_cast<size_t>(col)]) {
                    continue;
                }
                const double cur =
                    static_cast<double>(costs[static_cast<size_t>(row0 - 1) * static_cast<size_t>(num_cols) +
                                              static_cast<size_t>(col - 1)]) -
                    u[static_cast<size_t>(row0)] - v[static_cast<size_t>(col)];
                if (cur < minv[static_cast<size_t>(col)]) {
                    minv[static_cast<size_t>(col)] = cur;
                    way[static_cast<size_t>(col)] = col0;
                }
                if (minv[static_cast<size_t>(col)] < delta) {
                    delta = minv[static_cast<size_t>(col)];
                    col1 = col;
                }
            }

            for (int col = 0; col <= num_cols; ++col) {
                if (used[static_cast<size_t>(col)]) {
                    u[static_cast<size_t>(p[static_cast<size_t>(col)])] += delta;
                    v[static_cast<size_t>(col)] -= delta;
                } else {
                    minv[static_cast<size_t>(col)] -= delta;
                }
            }

            col0 = col1;
        } while (p[static_cast<size_t>(col0)] != 0);

        do {
            const int col1 = way[static_cast<size_t>(col0)];
            p[static_cast<size_t>(col0)] = p[static_cast<size_t>(col1)];
            col0 = col1;
        } while (col0 != 0);
    }

    std::vector<int> assignment(static_cast<size_t>(num_rows), -1);
    for (int col = 1; col <= num_cols; ++col) {
        if (p[static_cast<size_t>(col)] != 0) {
            assignment[static_cast<size_t>(p[static_cast<size_t>(col)] - 1)] = col - 1;
        }
    }
    return assignment;
}

std::vector<int> hungarian_assign(const std::vector<float> & costs, int num_rows, int num_cols) {
    if (num_rows <= num_cols) {
        return hungarian_rows_le_cols(costs, num_rows, num_cols);
    }

    std::vector<float> transposed(static_cast<size_t>(num_rows) * static_cast<size_t>(num_cols), 0.0f);
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < num_cols; ++col) {
            transposed[static_cast<size_t>(col) * static_cast<size_t>(num_rows) + static_cast<size_t>(row)] =
                costs[static_cast<size_t>(row) * static_cast<size_t>(num_cols) + static_cast<size_t>(col)];
        }
    }

    const std::vector<int> transposed_assignment = hungarian_rows_le_cols(transposed, num_cols, num_rows);
    std::vector<int> assignment(static_cast<size_t>(num_rows), -1);
    for (int col = 0; col < num_cols; ++col) {
        const int row = transposed_assignment[static_cast<size_t>(col)];
        if (row >= 0) {
            assignment[static_cast<size_t>(row)] = col;
        }
    }
    return assignment;
}

std::vector<int> compute_assignments(const std::vector<float> & costs, int num_rows, int num_cols) {
    if (num_rows == 0 || num_cols == 0) {
        return {};
    }
    return hungarian_assign(costs, num_rows, num_cols);
}

std::vector<bool> compute_row_mapped(
    const std::vector<float> & costs,
    int num_rows,
    int num_cols,
    const std::vector<int> & assignments
) {
    std::vector<bool> mapped(static_cast<size_t>(num_rows), false);
    for (int row = 0; row < num_rows; ++row) {
        const int col = assignments[static_cast<size_t>(row)];
        if (col < 0 || col >= num_cols) {
            continue;
        }
        bool row_has_valid_target = false;
        for (int target = 0; target < num_cols; ++target) {
            if (costs[static_cast<size_t>(row) * static_cast<size_t>(num_cols) + static_cast<size_t>(target)] <
                kInvalidCost * 0.5f) {
                row_has_valid_target = true;
                break;
            }
        }
        if (!row_has_valid_target) {
            continue;
        }
        mapped[static_cast<size_t>(row)] =
            costs[static_cast<size_t>(row) * static_cast<size_t>(num_cols) + static_cast<size_t>(col)] <
            kInvalidCost * 0.5f;
    }
    return mapped;
}

std::vector<float> crop_window_values(
    const diarization_window & window,
    float focus_start,
    float focus_end,
    int & out_num_frames
) {
    validate_window(window);
    if (window.num_frames == 0 || window.num_speakers == 0) {
        out_num_frames = 0;
        return {};
    }

    const float rel_start = (focus_start - window.start_time_seconds) / window.frame_step_seconds;
    int start_index = std::max(0, static_cast<int>(std::ceil(rel_start - 1.0f - kEpsilon)));
    const int desired_frames =
        std::max(1, static_cast<int>(std::lround((focus_end - focus_start) / window.frame_step_seconds)) + 1);
    if (start_index + desired_frames > window.num_frames) {
        start_index = std::max(0, window.num_frames - desired_frames);
    }
    const int end_index = std::min(window.num_frames - 1, start_index + desired_frames - 1);

    if (end_index < start_index) {
        out_num_frames = 0;
        return {};
    }

    out_num_frames = end_index - start_index + 1;
    std::vector<float> cropped(static_cast<size_t>(out_num_frames) * static_cast<size_t>(window.num_speakers), 0.0f);
    for (int frame = 0; frame < out_num_frames; ++frame) {
        const int source_frame = start_index + frame;
        std::memcpy(
            cropped.data() + static_cast<size_t>(frame) * static_cast<size_t>(window.num_speakers),
            window.values.data() + static_cast<size_t>(source_frame) * static_cast<size_t>(window.num_speakers),
            static_cast<size_t>(window.num_speakers) * sizeof(float)
        );
    }
    return cropped;
}

std::vector<float> aggregate_region(
    const std::vector<diarization_window> & buffers,
    float focus_start,
    float focus_end,
    DiarizationAggregationStrategy strategy,
    int & out_num_frames,
    int & out_num_speakers
) {
    if (buffers.empty()) {
        out_num_frames = 0;
        out_num_speakers = 0;
        return {};
    }

    out_num_speakers = buffers.front().num_speakers;

    if (strategy == DiarizationAggregationStrategy::First) {
        return crop_window_values(buffers.front(), focus_start, focus_end, out_num_frames);
    }

    std::vector<std::vector<float>> crops;
    std::vector<std::vector<float>> weight_crops;
    crops.reserve(buffers.size());
    weight_crops.reserve(buffers.size());
    out_num_frames = -1;

    for (const diarization_window & buffer : buffers) {
        int crop_frames = 0;
        std::vector<float> crop = crop_window_values(buffer, focus_start, focus_end, crop_frames);
        if (out_num_frames < 0) {
            out_num_frames = crop_frames;
        } else if (crop_frames != out_num_frames) {
            throw std::runtime_error("delayed aggregation expected aligned crops with identical frame counts");
        }
        crops.push_back(std::move(crop));

        if (strategy == DiarizationAggregationStrategy::Hamming) {
            std::vector<float> full_window(static_cast<size_t>(buffer.num_frames), 0.0f);
            if (buffer.num_frames == 1) {
                full_window[0] = 1.0f;
            } else {
                const float denom = static_cast<float>(buffer.num_frames - 1);
                for (int frame = 0; frame < buffer.num_frames; ++frame) {
                    full_window[static_cast<size_t>(frame)] =
                        0.54f - 0.46f * std::cos((2.0f * kPi * static_cast<float>(frame)) / denom);
                }
            }

            diarization_window weight_window;
            weight_window.start_time_seconds = buffer.start_time_seconds;
            weight_window.frame_step_seconds = buffer.frame_step_seconds;
            weight_window.num_frames = buffer.num_frames;
            weight_window.num_speakers = 1;
            weight_window.values = std::move(full_window);

            int weight_frames = 0;
            std::vector<float> weight_crop = crop_window_values(weight_window, focus_start, focus_end, weight_frames);
            if (weight_frames != out_num_frames) {
                throw std::runtime_error("hamming aggregation weight crop does not match data crop");
            }
            weight_crops.push_back(std::move(weight_crop));
        }
    }

    if (out_num_frames <= 0 || out_num_speakers <= 0) {
        return {};
    }

    std::vector<float> aggregated(
        static_cast<size_t>(out_num_frames) * static_cast<size_t>(out_num_speakers),
        0.0f
    );

    for (int frame = 0; frame < out_num_frames; ++frame) {
        for (int speaker = 0; speaker < out_num_speakers; ++speaker) {
            float numerator = 0.0f;
            float denominator = 0.0f;
            for (size_t buffer_index = 0; buffer_index < crops.size(); ++buffer_index) {
                const float weight =
                    strategy == DiarizationAggregationStrategy::Hamming
                        ? weight_crops[buffer_index][static_cast<size_t>(frame)]
                        : 1.0f;
                numerator += crops[buffer_index][static_cast<size_t>(frame) * static_cast<size_t>(out_num_speakers) +
                                                static_cast<size_t>(speaker)] *
                             weight;
                denominator += weight;
            }
            aggregated[static_cast<size_t>(frame) * static_cast<size_t>(out_num_speakers) +
                       static_cast<size_t>(speaker)] = numerator / std::max(denominator, kEpsilon);
        }
    }

    return aggregated;
}

diarization_window make_output_window(
    float focus_start,
    float focus_end,
    int num_frames,
    int num_speakers,
    std::vector<float> values
) {
    diarization_window output;
    output.start_time_seconds = focus_start;
    output.frame_step_seconds =
        num_frames > 0 ? (focus_end - focus_start) / static_cast<float>(num_frames) : 0.0f;
    output.num_frames = num_frames;
    output.num_speakers = num_speakers;
    output.values = std::move(values);
    return output;
}

diarization_window prepend_first_output(
    const diarization_window & output_window,
    float output_region_end,
    const std::vector<diarization_window> & buffers
) {
    if (buffers.size() != 1 || std::fabs(buffers.back().start_time_seconds) > kEpsilon) {
        return output_window;
    }

    int first_frames = 0;
    std::vector<float> first_output = crop_window_values(buffers.front(), 0.0f, output_region_end, first_frames);
    if (first_frames <= 0) {
        return output_window;
    }

    const int tail_frames = output_window.num_frames;
    if (tail_frames > first_frames) {
        throw std::runtime_error("cannot prepend aggregated output with a larger tail than the first chunk");
    }

    for (int frame = 0; frame < tail_frames; ++frame) {
        const int target_frame = first_frames - tail_frames + frame;
        std::memcpy(
            first_output.data() + static_cast<size_t>(target_frame) * static_cast<size_t>(output_window.num_speakers),
            output_window.values.data() + static_cast<size_t>(frame) * static_cast<size_t>(output_window.num_speakers),
            static_cast<size_t>(output_window.num_speakers) * sizeof(float)
        );
    }

    diarization_window prepended;
    prepended.start_time_seconds = 0.0f;
    prepended.frame_step_seconds = output_region_end / static_cast<float>(first_frames);
    prepended.num_frames = first_frames;
    prepended.num_speakers = output_window.num_speakers;
    prepended.values = std::move(first_output);
    return prepended;
}

std::vector<float> cosine_distance_matrix(
    const float * embeddings,
    int num_local_speakers,
    int embedding_dim,
    const std::vector<float> & centers,
    int num_global_speakers
) {
    std::vector<float> distances(
        static_cast<size_t>(num_local_speakers) * static_cast<size_t>(num_global_speakers),
        kInvalidCost
    );

    std::vector<float> center_norms(static_cast<size_t>(num_global_speakers), 0.0f);
    for (int center = 0; center < num_global_speakers; ++center) {
        float norm_sq = 0.0f;
        for (int dim = 0; dim < embedding_dim; ++dim) {
            const float value = centers[static_cast<size_t>(center) * static_cast<size_t>(embedding_dim) +
                                        static_cast<size_t>(dim)];
            norm_sq += value * value;
        }
        center_norms[static_cast<size_t>(center)] = std::sqrt(norm_sq);
    }

    for (int local = 0; local < num_local_speakers; ++local) {
        const float * embedding = embeddings + static_cast<size_t>(local) * static_cast<size_t>(embedding_dim);
        if (contains_nan(embedding, embedding_dim)) {
            continue;
        }

        float embedding_norm_sq = 0.0f;
        for (int dim = 0; dim < embedding_dim; ++dim) {
            embedding_norm_sq += embedding[dim] * embedding[dim];
        }
        const float embedding_norm = std::sqrt(embedding_norm_sq);
        if (!(embedding_norm > kEpsilon)) {
            continue;
        }

        for (int center = 0; center < num_global_speakers; ++center) {
            const float center_norm = center_norms[static_cast<size_t>(center)];
            if (!(center_norm > kEpsilon)) {
                continue;
            }

            float dot = 0.0f;
            for (int dim = 0; dim < embedding_dim; ++dim) {
                dot += embedding[dim] *
                       centers[static_cast<size_t>(center) * static_cast<size_t>(embedding_dim) +
                               static_cast<size_t>(dim)];
            }

            const float cosine = dot / (embedding_norm * center_norm);
            distances[static_cast<size_t>(local) * static_cast<size_t>(num_global_speakers) +
                      static_cast<size_t>(center)] = 1.0f - cosine;
        }
    }

    return distances;
}

} // namespace

DelayedDiarizationAggregation::DelayedDiarizationAggregation(
    float step_seconds,
    float latency_seconds,
    DiarizationAggregationStrategy strategy
) :
    step_seconds_(step_seconds),
    latency_seconds_(latency_seconds <= 0.0f ? step_seconds : latency_seconds),
    strategy_(strategy) {
    if (step_seconds_ <= 0.0f || latency_seconds_ < step_seconds_) {
        throw std::invalid_argument("delayed aggregation requires 0 < step <= latency");
    }
    num_overlapping_windows_ = static_cast<int>(std::lround(latency_seconds_ / step_seconds_));
}

diarization_window DelayedDiarizationAggregation::aggregate(const std::vector<diarization_window> & buffers) const {
    if (buffers.empty()) {
        return {};
    }

    for (const diarization_window & buffer : buffers) {
        validate_window(buffer);
    }

    const float focus_start = buffers.back().extent_end_seconds() - latency_seconds_;
    const float focus_end = focus_start + step_seconds_;

    int num_frames = 0;
    int num_speakers = 0;
    std::vector<float> values = aggregate_region(buffers, focus_start, focus_end, strategy_, num_frames, num_speakers);
    diarization_window output = make_output_window(focus_start, focus_end, num_frames, num_speakers, std::move(values));
    return prepend_first_output(output, focus_end, buffers);
}

OnlineSpeakerClustering::OnlineSpeakerClustering(
    float tau_active,
    float rho_update,
    float delta_new,
    int max_speakers
) :
    tau_active_(tau_active),
    rho_update_(rho_update),
    delta_new_(delta_new),
    max_speakers_(max_speakers) {
    reset();
}

void OnlineSpeakerClustering::reset() {
    embedding_dim_ = 0;
    centers_.clear();
    active_centers_.assign(static_cast<size_t>(max_speakers_), false);
    blocked_centers_.assign(static_cast<size_t>(max_speakers_), false);
}

int OnlineSpeakerClustering::num_known_speakers() const {
    return static_cast<int>(std::count(active_centers_.begin(), active_centers_.end(), true));
}

int OnlineSpeakerClustering::next_free_center() const {
    for (int center = 0; center < max_speakers_; ++center) {
        if (!active_centers_[static_cast<size_t>(center)] && !blocked_centers_[static_cast<size_t>(center)]) {
            return center;
        }
    }
    return -1;
}

int OnlineSpeakerClustering::add_center(const float * embedding, int embedding_dim) {
    const int center = next_free_center();
    if (center < 0) {
        throw std::runtime_error("no free global speaker centers remain");
    }
    std::memcpy(
        centers_.data() + static_cast<size_t>(center) * static_cast<size_t>(embedding_dim),
        embedding,
        static_cast<size_t>(embedding_dim) * sizeof(float)
    );
    active_centers_[static_cast<size_t>(center)] = true;
    return center;
}

void OnlineSpeakerClustering::update_center(int center_index, const float * embedding, int embedding_dim) {
    float * center = centers_.data() + static_cast<size_t>(center_index) * static_cast<size_t>(embedding_dim);
    for (int dim = 0; dim < embedding_dim; ++dim) {
        center[dim] += embedding[dim];
    }
}

std::vector<int> OnlineSpeakerClustering::identify(
    const diarization_window & segmentation,
    const float * embeddings,
    int embedding_dim
) {
    validate_window(segmentation);
    if (embedding_dim <= 0) {
        throw std::invalid_argument("speaker embeddings must have a positive dimension");
    }
    if (segmentation.num_speakers == 0) {
        return {};
    }

    std::vector<int> active_speakers;
    std::vector<int> long_speakers;
    active_speakers.reserve(static_cast<size_t>(segmentation.num_speakers));
    long_speakers.reserve(static_cast<size_t>(segmentation.num_speakers));

    for (int speaker = 0; speaker < segmentation.num_speakers; ++speaker) {
        float max_value = -std::numeric_limits<float>::infinity();
        float sum_value = 0.0f;
        for (int frame = 0; frame < segmentation.num_frames; ++frame) {
            const float value = segmentation.at(frame, speaker);
            max_value = std::max(max_value, value);
            sum_value += value;
        }

        const float mean_value = sum_value / std::max(segmentation.num_frames, 1);
        const float * embedding = embeddings + static_cast<size_t>(speaker) * static_cast<size_t>(embedding_dim);
        const bool has_valid_embedding = !contains_nan(embedding, embedding_dim);
        if (max_value >= tau_active_ && has_valid_embedding) {
            active_speakers.push_back(speaker);
        }
        if (mean_value >= rho_update_) {
            long_speakers.push_back(speaker);
        }
    }

    std::vector<int> local_to_global(static_cast<size_t>(segmentation.num_speakers), -1);

    if (embedding_dim_ == 0) {
        embedding_dim_ = embedding_dim;
        centers_.assign(static_cast<size_t>(max_speakers_) * static_cast<size_t>(embedding_dim_), 0.0f);
    } else if (embedding_dim_ != embedding_dim) {
        throw std::invalid_argument("speaker embedding dimension changed after clustering state was initialized");
    }

    if (num_known_speakers() == 0) {
        for (const int speaker : active_speakers) {
            local_to_global[static_cast<size_t>(speaker)] =
                add_center(embeddings + static_cast<size_t>(speaker) * static_cast<size_t>(embedding_dim), embedding_dim);
        }
        return local_to_global;
    }

    std::vector<float> dist_matrix =
        cosine_distance_matrix(embeddings, segmentation.num_speakers, embedding_dim, centers_, max_speakers_);

    std::vector<bool> active_local(static_cast<size_t>(segmentation.num_speakers), false);
    for (const int speaker : active_speakers) {
        active_local[static_cast<size_t>(speaker)] = true;
    }

    for (int local = 0; local < segmentation.num_speakers; ++local) {
        if (!active_local[static_cast<size_t>(local)]) {
            for (int global = 0; global < max_speakers_; ++global) {
                dist_matrix[static_cast<size_t>(local) * static_cast<size_t>(max_speakers_) +
                            static_cast<size_t>(global)] = kInvalidCost;
            }
        }
    }
    for (int global = 0; global < max_speakers_; ++global) {
        if (!active_centers_[static_cast<size_t>(global)] || blocked_centers_[static_cast<size_t>(global)]) {
            for (int local = 0; local < segmentation.num_speakers; ++local) {
                dist_matrix[static_cast<size_t>(local) * static_cast<size_t>(max_speakers_) +
                            static_cast<size_t>(global)] = kInvalidCost;
            }
        }
    }

    std::vector<int> assignments = compute_assignments(dist_matrix, segmentation.num_speakers, max_speakers_);
    std::vector<bool> mapped = compute_row_mapped(dist_matrix, segmentation.num_speakers, max_speakers_, assignments);

    std::vector<float> thresholded_matrix = dist_matrix;
    for (int local = 0; local < segmentation.num_speakers; ++local) {
        if (!mapped[static_cast<size_t>(local)]) {
            continue;
        }
        const int global = assignments[static_cast<size_t>(local)];
        const float cost = thresholded_matrix[static_cast<size_t>(local) * static_cast<size_t>(max_speakers_) +
                                              static_cast<size_t>(global)];
        if (cost >= delta_new_) {
            for (int candidate = 0; candidate < max_speakers_; ++candidate) {
                thresholded_matrix[static_cast<size_t>(local) * static_cast<size_t>(max_speakers_) +
                                   static_cast<size_t>(candidate)] = kInvalidCost;
            }
        }
    }

    assignments = compute_assignments(thresholded_matrix, segmentation.num_speakers, max_speakers_);
    mapped = compute_row_mapped(thresholded_matrix, segmentation.num_speakers, max_speakers_, assignments);

    std::vector<int> missed_speakers;
    missed_speakers.reserve(active_speakers.size());
    for (const int speaker : active_speakers) {
        if (!mapped[static_cast<size_t>(speaker)]) {
            missed_speakers.push_back(speaker);
        }
    }

    auto is_long_speaker = [&long_speakers](int speaker) {
        return std::find(long_speakers.begin(), long_speakers.end(), speaker) != long_speakers.end();
    };

    auto active_center_preference = [this](int center) {
        return active_centers_[static_cast<size_t>(center)] && !blocked_centers_[static_cast<size_t>(center)];
    };

    std::vector<int> assigned_targets;
    assigned_targets.reserve(assignments.size());
    for (int local = 0; local < segmentation.num_speakers; ++local) {
        if (mapped[static_cast<size_t>(local)]) {
            assigned_targets.push_back(assignments[static_cast<size_t>(local)]);
        }
    }

    const int free_center_count = max_speakers_ - num_known_speakers() -
                                  static_cast<int>(std::count(blocked_centers_.begin(), blocked_centers_.end(), true));
    std::vector<int> new_center_speakers;
    new_center_speakers.reserve(missed_speakers.size());

    for (const int speaker : missed_speakers) {
        const bool has_space = static_cast<int>(new_center_speakers.size()) < free_center_count;
        if (has_space && is_long_speaker(speaker)) {
            new_center_speakers.push_back(speaker);
            continue;
        }

        std::vector<int> preferences(static_cast<size_t>(max_speakers_));
        std::iota(preferences.begin(), preferences.end(), 0);
        std::sort(preferences.begin(), preferences.end(), [&](int lhs, int rhs) {
            const float lhs_cost = dist_matrix[static_cast<size_t>(speaker) * static_cast<size_t>(max_speakers_) +
                                               static_cast<size_t>(lhs)];
            const float rhs_cost = dist_matrix[static_cast<size_t>(speaker) * static_cast<size_t>(max_speakers_) +
                                               static_cast<size_t>(rhs)];
            return lhs_cost < rhs_cost;
        });

        for (const int preferred_center : preferences) {
            if (!active_center_preference(preferred_center)) {
                continue;
            }
            if (std::find(assigned_targets.begin(), assigned_targets.end(), preferred_center) != assigned_targets.end()) {
                continue;
            }
            assignments[static_cast<size_t>(speaker)] = preferred_center;
            mapped[static_cast<size_t>(speaker)] = true;
            assigned_targets.push_back(preferred_center);
            break;
        }
    }

    for (int local = 0; local < segmentation.num_speakers; ++local) {
        if (!mapped[static_cast<size_t>(local)]) {
            continue;
        }
        if (!is_long_speaker(local)) {
            continue;
        }
        if (std::find(missed_speakers.begin(), missed_speakers.end(), local) != missed_speakers.end()) {
            continue;
        }
        update_center(
            assignments[static_cast<size_t>(local)],
            embeddings + static_cast<size_t>(local) * static_cast<size_t>(embedding_dim),
            embedding_dim
        );
    }

    for (const int speaker : new_center_speakers) {
        assignments[static_cast<size_t>(speaker)] =
            add_center(embeddings + static_cast<size_t>(speaker) * static_cast<size_t>(embedding_dim), embedding_dim);
        mapped[static_cast<size_t>(speaker)] = true;
    }

    for (int local = 0; local < segmentation.num_speakers; ++local) {
        if (mapped[static_cast<size_t>(local)]) {
            local_to_global[static_cast<size_t>(local)] = assignments[static_cast<size_t>(local)];
        }
    }

    return local_to_global;
}

diarization_window OnlineSpeakerClustering::apply_mapping(
    const diarization_window & segmentation,
    const std::vector<int> & local_to_global
) const {
    validate_window(segmentation);
    if (static_cast<int>(local_to_global.size()) != segmentation.num_speakers) {
        throw std::invalid_argument("local/global speaker map size does not match the local segmentation width");
    }

    diarization_window mapped;
    mapped.start_time_seconds = segmentation.start_time_seconds;
    mapped.frame_step_seconds = segmentation.frame_step_seconds;
    mapped.num_frames = segmentation.num_frames;
    mapped.num_speakers = max_speakers_;
    mapped.values.assign(static_cast<size_t>(mapped.num_frames) * static_cast<size_t>(mapped.num_speakers), 0.0f);

    for (int local = 0; local < segmentation.num_speakers; ++local) {
        const int global = local_to_global[static_cast<size_t>(local)];
        if (global < 0 || global >= max_speakers_) {
            continue;
        }
        for (int frame = 0; frame < segmentation.num_frames; ++frame) {
            mapped.at(frame, global) = segmentation.at(frame, local);
        }
    }

    return mapped;
}

StreamingDiarizer::StreamingDiarizer(const diarization_config & config) :
    config_(config),
    clustering_(config.tau_active, config.rho_update, config.delta_new, config.max_speakers),
    prediction_aggregation_(config.step_seconds, config.latency_seconds, DiarizationAggregationStrategy::Hamming) {
    if (config_.step_seconds <= 0.0f || config_.latency_seconds < config_.step_seconds) {
        throw std::invalid_argument("streaming diarizer requires 0 < step <= latency");
    }
}

void StreamingDiarizer::reset() {
    clustering_.reset();
    prediction_buffer_.clear();
}

diarization_window StreamingDiarizer::compute_overlap_aware_weights(const diarization_window & segmentation) const {
    validate_window(segmentation);
    diarization_window weights = segmentation;
    if (segmentation.num_frames == 0 || segmentation.num_speakers == 0) {
        return weights;
    }

    const std::vector<float> probs =
        softmax_by_frame_ggml(segmentation.values, segmentation.num_frames, segmentation.num_speakers, config_.beta);

    for (size_t i = 0; i < weights.values.size(); ++i) {
        const float seg = segmentation.values[i];
        const float prob = probs[i];
        weights.values[i] =
            std::max(kMinWeight, std::pow(seg, config_.gamma) * std::pow(prob, config_.gamma));
    }

    if (config_.normalize_embedding_weights) {
        for (int speaker = 0; speaker < weights.num_speakers; ++speaker) {
            float min_value = std::numeric_limits<float>::infinity();
            float max_value = -std::numeric_limits<float>::infinity();
            for (int frame = 0; frame < weights.num_frames; ++frame) {
                const float value = weights.at(frame, speaker);
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
            }

            const float denom = max_value - min_value;
            for (int frame = 0; frame < weights.num_frames; ++frame) {
                float value = denom > kEpsilon ? (weights.at(frame, speaker) - min_value) / denom : kMinWeight;
                if (std::isnan(value)) {
                    value = kMinWeight;
                }
                weights.at(frame, speaker) = std::max(kMinWeight, value);
            }
        }
    }

    return weights;
}

std::vector<float> StreamingDiarizer::normalize_embeddings(
    const float * embeddings,
    int num_speakers,
    int embedding_dim,
    float target_norm
) const {
    if (num_speakers < 0 || embedding_dim <= 0) {
        throw std::invalid_argument("embedding normalization requires non-negative speakers and positive dimension");
    }

    std::vector<float> normalized(
        static_cast<size_t>(num_speakers) * static_cast<size_t>(embedding_dim),
        std::numeric_limits<float>::quiet_NaN()
    );

    for (int speaker = 0; speaker < num_speakers; ++speaker) {
        const float * source = embeddings + static_cast<size_t>(speaker) * static_cast<size_t>(embedding_dim);
        if (contains_nan(source, embedding_dim)) {
            continue;
        }

        float norm_sq = 0.0f;
        for (int dim = 0; dim < embedding_dim; ++dim) {
            norm_sq += source[dim] * source[dim];
        }
        const float norm = std::sqrt(norm_sq);
        if (!(norm > kEpsilon)) {
            continue;
        }

        const float scale = target_norm / norm;
        for (int dim = 0; dim < embedding_dim; ++dim) {
            normalized[static_cast<size_t>(speaker) * static_cast<size_t>(embedding_dim) + static_cast<size_t>(dim)] =
                source[dim] * scale;
        }
    }

    return normalized;
}

diarization_result StreamingDiarizer::process_window(
    const diarization_window & segmentation,
    const float * embeddings,
    int embedding_dim
) {
    validate_window(segmentation);
    if (segmentation.num_speakers == 0) {
        return {};
    }

    std::vector<float> normalized_embeddings =
        normalize_embeddings(embeddings, segmentation.num_speakers, embedding_dim);
    const std::vector<int> local_to_global =
        clustering_.identify(segmentation, normalized_embeddings.data(), embedding_dim);

    diarization_window permuted = clustering_.apply_mapping(segmentation, local_to_global);
    prediction_buffer_.push_back(permuted);

    diarization_result result;
    result.local_to_global = local_to_global;
    result.aggregated_scores = prediction_aggregation_.aggregate(prediction_buffer_);
    result.binarized_scores = binarize(result.aggregated_scores);

    if (static_cast<int>(prediction_buffer_.size()) == prediction_aggregation_.num_overlapping_windows()) {
        prediction_buffer_.erase(prediction_buffer_.begin());
    }

    return result;
}

diarization_window StreamingDiarizer::binarize(const diarization_window & scores) const {
    diarization_window result = scores;
    for (float & value : result.values) {
        value = value >= config_.tau_active ? 1.0f : 0.0f;
    }
    return result;
}

} // namespace q3asr
