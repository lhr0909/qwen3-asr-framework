#include "offline_diarizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace q3asr {

namespace {

constexpr float kEpsilon = 1.0e-6f;
constexpr int kInactiveCluster = -2;
constexpr double kPi = 3.14159265358979323846;

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

size_t tensor_element_count(const ggml_tensor * tensor) {
    return static_cast<size_t>(ggml_nelements(tensor));
}

std::vector<int64_t> tensor_shape(const ggml_tensor * tensor) {
    const int dims = ggml_n_dims(tensor);
    std::vector<int64_t> shape(static_cast<size_t>(dims), 0);
    for (int i = 0; i < dims; ++i) {
        shape[static_cast<size_t>(i)] = tensor->ne[dims - 1 - i];
    }
    return shape;
}

bool tensor_is_f32(const ggml_tensor * tensor) {
    return tensor != nullptr && tensor->type == GGML_TYPE_F32;
}

std::vector<float> tensor_values_f32(const ggml_tensor * tensor) {
    if (!tensor_is_f32(tensor)) {
        return {};
    }
    std::vector<float> values(tensor_element_count(tensor), 0.0f);
    std::memcpy(values.data(), tensor->data, values.size() * sizeof(float));
    return values;
}

float dot_product(const float * lhs, const float * rhs, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        sum += lhs[i] * rhs[i];
    }
    return sum;
}

float l2_norm(const float * values, int count) {
    return std::sqrt(std::max(0.0f, dot_product(values, values, count)));
}

void normalize_scaled(std::vector<float> & values, float scale) {
    const float norm = l2_norm(values.data(), static_cast<int>(values.size()));
    if (!std::isfinite(norm) || norm <= kEpsilon) {
        return;
    }
    const float factor = scale / norm;
    for (float & value : values) {
        value *= factor;
    }
}

float cosine_similarity(const float * lhs, const float * rhs, int count) {
    const float lhs_norm = l2_norm(lhs, count);
    const float rhs_norm = l2_norm(rhs, count);
    if (!std::isfinite(lhs_norm) || !std::isfinite(rhs_norm) || lhs_norm <= kEpsilon || rhs_norm <= kEpsilon) {
        return 0.0f;
    }
    return dot_product(lhs, rhs, count) / (lhs_norm * rhs_norm);
}

bool contains_nan(const float * values, int count) {
    for (int i = 0; i < count; ++i) {
        if (std::isnan(values[i])) {
            return true;
        }
    }
    return false;
}

std::vector<int> hungarian_rows_le_cols(const std::vector<float> & costs, int num_rows, int num_cols) {
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

std::vector<float> softmax_rows(const std::vector<float> & logits, int num_rows, int num_cols, float scale) {
    std::vector<float> result(logits.size(), 0.0f);
    for (int row = 0; row < num_rows; ++row) {
        float max_value = -std::numeric_limits<float>::infinity();
        for (int col = 0; col < num_cols; ++col) {
            max_value = std::max(
                max_value,
                logits[static_cast<size_t>(row) * static_cast<size_t>(num_cols) + static_cast<size_t>(col)] * scale
            );
        }

        double denom = 0.0;
        for (int col = 0; col < num_cols; ++col) {
            const float value =
                logits[static_cast<size_t>(row) * static_cast<size_t>(num_cols) + static_cast<size_t>(col)] * scale;
            const double exp_value = std::exp(static_cast<double>(value - max_value));
            result[static_cast<size_t>(row) * static_cast<size_t>(num_cols) + static_cast<size_t>(col)] =
                static_cast<float>(exp_value);
            denom += exp_value;
        }

        if (denom <= 0.0) {
            continue;
        }
        const float inv_denom = static_cast<float>(1.0 / denom);
        for (int col = 0; col < num_cols; ++col) {
            result[static_cast<size_t>(row) * static_cast<size_t>(num_cols) + static_cast<size_t>(col)] *= inv_denom;
        }
    }
    return result;
}

std::vector<int> centroid_linkage_init(
    const std::vector<float> & embeddings,
    int num_embeddings,
    int embedding_dim,
    float threshold
) {
    struct cluster_state {
        bool active = true;
        int count = 0;
        std::vector<int> members;
        std::vector<float> centroid;
    };

    std::vector<cluster_state> clusters(static_cast<size_t>(num_embeddings));
    for (int i = 0; i < num_embeddings; ++i) {
        cluster_state & cluster = clusters[static_cast<size_t>(i)];
        cluster.count = 1;
        cluster.members.push_back(i);
        cluster.centroid.assign(
            embeddings.begin() + static_cast<ptrdiff_t>(i) * embedding_dim,
            embeddings.begin() + static_cast<ptrdiff_t>(i + 1) * embedding_dim
        );
    }

    int active_clusters = num_embeddings;
    while (active_clusters > 1) {
        float best_distance = std::numeric_limits<float>::infinity();
        int best_i = -1;
        int best_j = -1;

        for (int i = 0; i < num_embeddings; ++i) {
            if (!clusters[static_cast<size_t>(i)].active) {
                continue;
            }
            for (int j = i + 1; j < num_embeddings; ++j) {
                if (!clusters[static_cast<size_t>(j)].active) {
                    continue;
                }

                float sum = 0.0f;
                for (int d = 0; d < embedding_dim; ++d) {
                    const float delta =
                        clusters[static_cast<size_t>(i)].centroid[static_cast<size_t>(d)] -
                        clusters[static_cast<size_t>(j)].centroid[static_cast<size_t>(d)];
                    sum += delta * delta;
                }
                const float distance = std::sqrt(std::max(0.0f, sum));
                if (distance < best_distance) {
                    best_distance = distance;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i < 0 || best_j < 0 || best_distance > threshold) {
            break;
        }

        cluster_state & keep = clusters[static_cast<size_t>(best_i)];
        cluster_state & drop = clusters[static_cast<size_t>(best_j)];
        const int merged_count = keep.count + drop.count;

        for (int d = 0; d < embedding_dim; ++d) {
            keep.centroid[static_cast<size_t>(d)] =
                (keep.centroid[static_cast<size_t>(d)] * static_cast<float>(keep.count) +
                 drop.centroid[static_cast<size_t>(d)] * static_cast<float>(drop.count)) /
                static_cast<float>(merged_count);
        }

        keep.members.insert(keep.members.end(), drop.members.begin(), drop.members.end());
        keep.count = merged_count;
        drop.active = false;
        drop.members.clear();
        drop.centroid.clear();
        --active_clusters;
    }

    std::vector<int> assignments(static_cast<size_t>(num_embeddings), -1);
    int cluster_index = 0;
    for (const cluster_state & cluster : clusters) {
        if (!cluster.active) {
            continue;
        }
        for (int member : cluster.members) {
            assignments[static_cast<size_t>(member)] = cluster_index;
        }
        ++cluster_index;
    }
    return assignments;
}

std::vector<int> kmeans_refine(
    const std::vector<float> & normalized_embeddings,
    int num_embeddings,
    int embedding_dim,
    int num_clusters
) {
    std::vector<float> centroids(static_cast<size_t>(num_clusters) * static_cast<size_t>(embedding_dim), 0.0f);
    for (int cluster = 0; cluster < num_clusters; ++cluster) {
        const int source = std::min(cluster, num_embeddings - 1);
        std::memcpy(
            centroids.data() + static_cast<size_t>(cluster) * static_cast<size_t>(embedding_dim),
            normalized_embeddings.data() + static_cast<size_t>(source) * static_cast<size_t>(embedding_dim),
            static_cast<size_t>(embedding_dim) * sizeof(float)
        );
    }

    std::vector<int> assignments(static_cast<size_t>(num_embeddings), 0);
    std::vector<float> new_centroids(centroids.size(), 0.0f);
    std::vector<int> counts(static_cast<size_t>(num_clusters), 0);

    for (int iter = 0; iter < 50; ++iter) {
        bool changed = false;
        for (int i = 0; i < num_embeddings; ++i) {
            int best_cluster = 0;
            float best_similarity = -std::numeric_limits<float>::infinity();
            const float * embedding =
                normalized_embeddings.data() + static_cast<size_t>(i) * static_cast<size_t>(embedding_dim);
            for (int cluster = 0; cluster < num_clusters; ++cluster) {
                const float * centroid = centroids.data() + static_cast<size_t>(cluster) * static_cast<size_t>(embedding_dim);
                const float similarity = cosine_similarity(embedding, centroid, embedding_dim);
                if (similarity > best_similarity) {
                    best_similarity = similarity;
                    best_cluster = cluster;
                }
            }
            if (assignments[static_cast<size_t>(i)] != best_cluster) {
                assignments[static_cast<size_t>(i)] = best_cluster;
                changed = true;
            }
        }

        std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (int i = 0; i < num_embeddings; ++i) {
            const int cluster = assignments[static_cast<size_t>(i)];
            ++counts[static_cast<size_t>(cluster)];
            const float * embedding =
                normalized_embeddings.data() + static_cast<size_t>(i) * static_cast<size_t>(embedding_dim);
            float * centroid = new_centroids.data() + static_cast<size_t>(cluster) * static_cast<size_t>(embedding_dim);
            for (int d = 0; d < embedding_dim; ++d) {
                centroid[d] += embedding[d];
            }
        }

        for (int cluster = 0; cluster < num_clusters; ++cluster) {
            float * centroid = new_centroids.data() + static_cast<size_t>(cluster) * static_cast<size_t>(embedding_dim);
            if (counts[static_cast<size_t>(cluster)] == 0) {
                std::memcpy(
                    centroid,
                    normalized_embeddings.data() +
                        static_cast<size_t>(std::min(cluster, num_embeddings - 1)) * static_cast<size_t>(embedding_dim),
                    static_cast<size_t>(embedding_dim) * sizeof(float)
                );
                continue;
            }
            const float inv_count = 1.0f / static_cast<float>(counts[static_cast<size_t>(cluster)]);
            for (int d = 0; d < embedding_dim; ++d) {
                centroid[d] *= inv_count;
            }
            const float norm = l2_norm(centroid, embedding_dim);
            if (std::isfinite(norm) && norm > kEpsilon) {
                const float inv_norm = 1.0f / norm;
                for (int d = 0; d < embedding_dim; ++d) {
                    centroid[d] *= inv_norm;
                }
            }
        }

        centroids.swap(new_centroids);
        if (!changed) {
            break;
        }
    }

    return assignments;
}

struct vbx_output {
    int num_frames = 0;
    int num_speakers = 0;
    std::vector<double> gamma;
    std::vector<double> pi;
};

vbx_output run_vbx(
    const std::vector<float> & features,
    int num_frames,
    int feature_dim,
    const std::vector<float> & phi,
    float Fa,
    float Fb,
    const std::vector<float> & gamma_init,
    int num_speakers
) {
    vbx_output result;
    result.num_frames = num_frames;
    result.num_speakers = num_speakers;
    result.gamma.assign(gamma_init.begin(), gamma_init.end());
    result.pi.assign(static_cast<size_t>(num_speakers), 1.0 / static_cast<double>(num_speakers));

    std::vector<double> G(static_cast<size_t>(num_frames), 0.0);
    std::vector<double> rho(static_cast<size_t>(num_frames) * static_cast<size_t>(feature_dim), 0.0);
    for (int frame = 0; frame < num_frames; ++frame) {
        double sumsq = 0.0;
        for (int d = 0; d < feature_dim; ++d) {
            const double value = features[static_cast<size_t>(frame) * static_cast<size_t>(feature_dim) +
                                          static_cast<size_t>(d)];
            sumsq += value * value;
            rho[static_cast<size_t>(frame) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)] =
                value * std::sqrt(static_cast<double>(phi[static_cast<size_t>(d)]));
        }
        G[static_cast<size_t>(frame)] = -0.5 * (sumsq + feature_dim * std::log(2.0 * kPi));
    }

    std::vector<double> invL(static_cast<size_t>(num_speakers) * static_cast<size_t>(feature_dim), 0.0);
    std::vector<double> alpha(static_cast<size_t>(num_speakers) * static_cast<size_t>(feature_dim), 0.0);
    double last_elbo = -std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < 20; ++iter) {
        std::vector<double> gamma_sum(static_cast<size_t>(num_speakers), 0.0);
        for (int frame = 0; frame < num_frames; ++frame) {
            for (int speaker = 0; speaker < num_speakers; ++speaker) {
                gamma_sum[static_cast<size_t>(speaker)] +=
                    result.gamma[static_cast<size_t>(frame) * static_cast<size_t>(num_speakers) +
                                 static_cast<size_t>(speaker)];
            }
        }

        for (int speaker = 0; speaker < num_speakers; ++speaker) {
            for (int d = 0; d < feature_dim; ++d) {
                const double inv =
                    1.0 / (1.0 + static_cast<double>(Fa / Fb) * gamma_sum[static_cast<size_t>(speaker)] *
                                      static_cast<double>(phi[static_cast<size_t>(d)]));
                invL[static_cast<size_t>(speaker) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)] = inv;
            }

            for (int d = 0; d < feature_dim; ++d) {
                double weighted_sum = 0.0;
                for (int frame = 0; frame < num_frames; ++frame) {
                    weighted_sum +=
                        result.gamma[static_cast<size_t>(frame) * static_cast<size_t>(num_speakers) +
                                     static_cast<size_t>(speaker)] *
                        rho[static_cast<size_t>(frame) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)];
                }
                alpha[static_cast<size_t>(speaker) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)] =
                    static_cast<double>(Fa / Fb) *
                    invL[static_cast<size_t>(speaker) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)] *
                    weighted_sum;
            }
        }

        std::vector<double> log_p(static_cast<size_t>(num_frames) * static_cast<size_t>(num_speakers), 0.0);
        for (int speaker = 0; speaker < num_speakers; ++speaker) {
            double speaker_penalty = 0.0;
            for (int d = 0; d < feature_dim; ++d) {
                const double inv =
                    invL[static_cast<size_t>(speaker) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)];
                const double a =
                    alpha[static_cast<size_t>(speaker) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)];
                speaker_penalty += (inv + a * a) * static_cast<double>(phi[static_cast<size_t>(d)]);
            }

            for (int frame = 0; frame < num_frames; ++frame) {
                double rho_dot_alpha = 0.0;
                for (int d = 0; d < feature_dim; ++d) {
                    rho_dot_alpha +=
                        rho[static_cast<size_t>(frame) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)] *
                        alpha[static_cast<size_t>(speaker) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)];
                }

                log_p[static_cast<size_t>(frame) * static_cast<size_t>(num_speakers) + static_cast<size_t>(speaker)] =
                    static_cast<double>(Fa) * (rho_dot_alpha - 0.5 * speaker_penalty + G[static_cast<size_t>(frame)]);
            }
        }

        std::vector<double> log_pi(static_cast<size_t>(num_speakers), 0.0);
        for (int speaker = 0; speaker < num_speakers; ++speaker) {
            log_pi[static_cast<size_t>(speaker)] =
                std::log(result.pi[static_cast<size_t>(speaker)] + 1.0e-8);
        }

        double log_pX = 0.0;
        std::vector<double> next_gamma(result.gamma.size(), 0.0);
        for (int frame = 0; frame < num_frames; ++frame) {
            double max_log = -std::numeric_limits<double>::infinity();
            for (int speaker = 0; speaker < num_speakers; ++speaker) {
                max_log = std::max(
                    max_log,
                    log_p[static_cast<size_t>(frame) * static_cast<size_t>(num_speakers) + static_cast<size_t>(speaker)] +
                        log_pi[static_cast<size_t>(speaker)]
                );
            }

            double denom = 0.0;
            for (int speaker = 0; speaker < num_speakers; ++speaker) {
                denom += std::exp(
                    log_p[static_cast<size_t>(frame) * static_cast<size_t>(num_speakers) + static_cast<size_t>(speaker)] +
                        log_pi[static_cast<size_t>(speaker)] - max_log
                );
            }

            const double log_px = max_log + std::log(denom);
            log_pX += log_px;
            for (int speaker = 0; speaker < num_speakers; ++speaker) {
                next_gamma[static_cast<size_t>(frame) * static_cast<size_t>(num_speakers) + static_cast<size_t>(speaker)] =
                    std::exp(
                        log_p[static_cast<size_t>(frame) * static_cast<size_t>(num_speakers) +
                              static_cast<size_t>(speaker)] +
                            log_pi[static_cast<size_t>(speaker)] - log_px
                    );
            }
        }

        result.gamma.swap(next_gamma);
        std::fill(result.pi.begin(), result.pi.end(), 0.0);
        for (int frame = 0; frame < num_frames; ++frame) {
            for (int speaker = 0; speaker < num_speakers; ++speaker) {
                result.pi[static_cast<size_t>(speaker)] +=
                    result.gamma[static_cast<size_t>(frame) * static_cast<size_t>(num_speakers) +
                                 static_cast<size_t>(speaker)];
            }
        }
        const double pi_sum = std::accumulate(result.pi.begin(), result.pi.end(), 0.0);
        if (pi_sum > 0.0) {
            for (double & value : result.pi) {
                value /= pi_sum;
            }
        }

        double regularizer = 0.0;
        for (int speaker = 0; speaker < num_speakers; ++speaker) {
            for (int d = 0; d < feature_dim; ++d) {
                const double inv =
                    invL[static_cast<size_t>(speaker) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)];
                const double a =
                    alpha[static_cast<size_t>(speaker) * static_cast<size_t>(feature_dim) + static_cast<size_t>(d)];
                regularizer += std::log(inv) - inv - a * a + 1.0;
            }
        }

        const double elbo = log_pX + static_cast<double>(Fb) * 0.5 * regularizer;
        if (iter > 0 && elbo - last_elbo < 1.0e-4) {
            break;
        }
        last_elbo = elbo;
    }

    return result;
}

std::vector<int> constrained_argmax(
    const std::vector<float> & soft_clusters,
    int num_chunks,
    int num_speakers,
    int num_clusters
) {
    std::vector<int> hard_clusters(static_cast<size_t>(num_chunks) * static_cast<size_t>(num_speakers), kInactiveCluster);
    std::vector<float> costs(static_cast<size_t>(num_speakers) * static_cast<size_t>(num_clusters), 0.0f);

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        float max_score = -std::numeric_limits<float>::infinity();
        float min_score = std::numeric_limits<float>::infinity();
        for (int speaker = 0; speaker < num_speakers; ++speaker) {
            for (int cluster = 0; cluster < num_clusters; ++cluster) {
                const float score =
                    soft_clusters[(static_cast<size_t>(chunk) * static_cast<size_t>(num_speakers) +
                                   static_cast<size_t>(speaker)) *
                                      static_cast<size_t>(num_clusters) +
                                  static_cast<size_t>(cluster)];
                if (std::isnan(score)) {
                    continue;
                }
                max_score = std::max(max_score, score);
                min_score = std::min(min_score, score);
            }
        }
        if (!std::isfinite(max_score)) {
            max_score = 0.0f;
            min_score = -1.0f;
        }
        const float replacement = std::isfinite(min_score) ? min_score : -1.0f;

        for (int speaker = 0; speaker < num_speakers; ++speaker) {
            for (int cluster = 0; cluster < num_clusters; ++cluster) {
                float score =
                    soft_clusters[(static_cast<size_t>(chunk) * static_cast<size_t>(num_speakers) +
                                   static_cast<size_t>(speaker)) *
                                      static_cast<size_t>(num_clusters) +
                                  static_cast<size_t>(cluster)];
                if (std::isnan(score)) {
                    score = replacement;
                }
                costs[static_cast<size_t>(speaker) * static_cast<size_t>(num_clusters) + static_cast<size_t>(cluster)] =
                    max_score - score;
            }
        }

        const std::vector<int> assignment = hungarian_assign(costs, num_speakers, num_clusters);
        for (int speaker = 0; speaker < num_speakers; ++speaker) {
            hard_clusters[static_cast<size_t>(chunk) * static_cast<size_t>(num_speakers) + static_cast<size_t>(speaker)] =
                assignment[static_cast<size_t>(speaker)] >= 0 ? assignment[static_cast<size_t>(speaker)]
                                                              : kInactiveCluster;
        }
    }

    return hard_clusters;
}

} // namespace

float offline_diarizer_problem::segmentation_at(int chunk_index, int frame_index, int speaker_index) const {
    return binary_segmentations[
        (static_cast<size_t>(chunk_index) * static_cast<size_t>(num_frames) + static_cast<size_t>(frame_index)) *
            static_cast<size_t>(num_speakers) +
        static_cast<size_t>(speaker_index)
    ];
}

const float * offline_diarizer_problem::embedding_ptr(int chunk_index, int speaker_index) const {
    return embeddings.data() +
           (static_cast<size_t>(chunk_index) * static_cast<size_t>(num_speakers) + static_cast<size_t>(speaker_index)) *
               static_cast<size_t>(embedding_dim);
}

int offline_diarizer_result::hard_cluster_at(int chunk_index, int speaker_index) const {
    return hard_clusters[
        static_cast<size_t>(chunk_index) * static_cast<size_t>(num_speakers) + static_cast<size_t>(speaker_index)
    ];
}

OfflineDiarizer::~OfflineDiarizer() {
    reset();
}

void OfflineDiarizer::reset() {
    free_diarization_gguf_model(segmentation_model_);
    free_diarization_gguf_model(embedding_model_);
    free_diarization_gguf_model(clustering_model_);
    assets_ = {};
    config_ = {};
    xvec_mean1_.clear();
    xvec_mean2_.clear();
    xvec_lda_.clear();
    plda_mu_.clear();
    plda_transform_.clear();
    plda_phi_.clear();
    xvec_input_dim_ = 0;
    xvec_output_dim_ = 0;
    plda_dim_ = 0;
    plda_ready_ = false;
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

bool OfflineDiarizer::load_clustering_model(const std::string & path) {
    DiarizationGGUFLoader loader;
    if (!loader.load(path, clustering_model_)) {
        error_msg_ = loader.get_error();
        return false;
    }

    if (
        clustering_model_.serialization_format != "numpy" ||
        clustering_model_.kind != "speaker-clustering" ||
        clustering_model_.source_repo != "pyannote/speaker-diarization-community-1"
    ) {
        error_msg_ = "Community-1 offline diarizer requires the converted Community-1 clustering GGUF";
        return false;
    }

    const ggml_tensor * mean1_tensor = clustering_model_.find_tensor("xvec.mean1");
    const ggml_tensor * mean2_tensor = clustering_model_.find_tensor("xvec.mean2");
    const ggml_tensor * lda_tensor = clustering_model_.find_tensor("xvec.lda");
    const ggml_tensor * plda_mu_tensor = clustering_model_.find_tensor("plda.mu");
    const ggml_tensor * plda_transform_tensor = clustering_model_.find_tensor("plda.transform");
    const ggml_tensor * plda_psi_tensor = clustering_model_.find_tensor("plda.psi");

    if (
        mean1_tensor == nullptr || mean2_tensor == nullptr || lda_tensor == nullptr || plda_mu_tensor == nullptr ||
        plda_transform_tensor == nullptr || plda_psi_tensor == nullptr
    ) {
        error_msg_ = "Community-1 clustering GGUF is missing one or more required tensors";
        return false;
    }

    xvec_mean1_ = tensor_values_f32(mean1_tensor);
    xvec_mean2_ = tensor_values_f32(mean2_tensor);
    xvec_lda_ = tensor_values_f32(lda_tensor);
    plda_mu_ = tensor_values_f32(plda_mu_tensor);
    plda_transform_ = tensor_values_f32(plda_transform_tensor);
    plda_phi_ = tensor_values_f32(plda_psi_tensor);

    const std::vector<int64_t> lda_shape = tensor_shape(lda_tensor);
    const std::vector<int64_t> transform_shape = tensor_shape(plda_transform_tensor);
    if (lda_shape.size() != 2 || transform_shape.size() != 2) {
        error_msg_ = "Community-1 clustering GGUF tensors have unexpected rank";
        return false;
    }

    xvec_input_dim_ = static_cast<int>(lda_shape[0]);
    xvec_output_dim_ = static_cast<int>(lda_shape[1]);
    if (xvec_input_dim_ <= 0 || xvec_output_dim_ <= 0) {
        error_msg_ = "Community-1 clustering GGUF has invalid x-vector transform dimensions";
        return false;
    }

    if (static_cast<int>(xvec_mean1_.size()) != xvec_input_dim_ || static_cast<int>(xvec_mean2_.size()) != xvec_output_dim_) {
        error_msg_ = "Community-1 clustering GGUF x-vector statistics do not match the transform dimensions";
        return false;
    }

    if (transform_shape[0] != transform_shape[1]) {
        error_msg_ = "Community-1 clustering GGUF PLDA transform must be square";
        return false;
    }
    plda_dim_ = static_cast<int>(transform_shape[0]);
    if (
        static_cast<int>(plda_mu_.size()) != plda_dim_ || static_cast<int>(plda_phi_.size()) != plda_dim_ ||
        transform_shape[1] != xvec_output_dim_
    ) {
        error_msg_ = "Community-1 clustering GGUF PLDA tensors do not match the x-vector output dimension";
        return false;
    }

    plda_ready_ = true;
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
    if (!file_exists(assets.clustering_model_path)) {
        error_msg_ = "Missing Community-1 clustering GGUF: " + assets.clustering_model_path;
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

    return load_clustering_model(assets.clustering_model_path);
}

bool OfflineDiarizer::cluster(const offline_diarizer_problem & problem, offline_diarizer_result & result) {
    result = {};

    if (!plda_ready_) {
        error_msg_ = "Community-1 clustering assets are not loaded";
        return false;
    }

    if (
        problem.num_chunks <= 0 || problem.num_frames <= 0 || problem.num_speakers <= 0 || problem.embedding_dim <= 0
    ) {
        error_msg_ = "Offline diarizer problem has invalid dimensions";
        return false;
    }

    const size_t expected_segmentation_size =
        static_cast<size_t>(problem.num_chunks) * static_cast<size_t>(problem.num_frames) *
        static_cast<size_t>(problem.num_speakers);
    const size_t expected_embedding_size =
        static_cast<size_t>(problem.num_chunks) * static_cast<size_t>(problem.num_speakers) *
        static_cast<size_t>(problem.embedding_dim);

    if (problem.binary_segmentations.size() != expected_segmentation_size) {
        error_msg_ = "Offline diarizer problem segmentation payload size does not match its dimensions";
        return false;
    }
    if (problem.embeddings.size() != expected_embedding_size) {
        error_msg_ = "Offline diarizer problem embedding payload size does not match its dimensions";
        return false;
    }
    if (problem.embedding_dim != xvec_input_dim_) {
        error_msg_ = "Offline diarizer embedding dimension does not match the Community-1 x-vector transform";
        return false;
    }

    const int requested_clusters = std::max(0, problem.num_clusters);
    const int min_clusters = std::max(1, problem.min_clusters);
    const int max_clusters = std::max(min_clusters, problem.max_clusters);

    std::vector<float> train_embeddings;
    std::vector<int> train_chunk_idx;
    std::vector<int> train_speaker_idx;
    std::vector<float> normalized_train_embeddings;

    const float min_clean_frames = problem.min_active_ratio * static_cast<float>(problem.num_frames);
    for (int chunk = 0; chunk < problem.num_chunks; ++chunk) {
        for (int speaker = 0; speaker < problem.num_speakers; ++speaker) {
            int clean_frames = 0;
            for (int frame = 0; frame < problem.num_frames; ++frame) {
                int active_count = 0;
                for (int local_speaker = 0; local_speaker < problem.num_speakers; ++local_speaker) {
                    if (problem.segmentation_at(chunk, frame, local_speaker) > 0.5f) {
                        ++active_count;
                    }
                }
                if (active_count == 1 && problem.segmentation_at(chunk, frame, speaker) > 0.5f) {
                    ++clean_frames;
                }
            }

            const float * embedding = problem.embedding_ptr(chunk, speaker);
            if (clean_frames < min_clean_frames || contains_nan(embedding, problem.embedding_dim)) {
                continue;
            }

            train_embeddings.insert(
                train_embeddings.end(),
                embedding,
                embedding + static_cast<ptrdiff_t>(problem.embedding_dim)
            );
            train_chunk_idx.push_back(chunk);
            train_speaker_idx.push_back(speaker);

            std::vector<float> normalized(
                embedding,
                embedding + static_cast<ptrdiff_t>(problem.embedding_dim)
            );
            normalize_scaled(normalized, 1.0f);
            normalized_train_embeddings.insert(
                normalized_train_embeddings.end(),
                normalized.begin(),
                normalized.end()
            );
        }
    }

    const int num_train_embeddings = static_cast<int>(train_chunk_idx.size());
    if (num_train_embeddings == 0) {
        error_msg_ = "Offline diarizer could not find any valid training embeddings";
        return false;
    }

    std::vector<float> centroids;
    int num_result_clusters = 1;
    bool constrained_assignment = true;

    if (num_train_embeddings == 1) {
        centroids.assign(train_embeddings.begin(), train_embeddings.end());
        num_result_clusters = 1;
    } else {
        const std::vector<int> ahc_clusters = centroid_linkage_init(
            normalized_train_embeddings,
            num_train_embeddings,
            problem.embedding_dim,
            config_.clustering_threshold
        );
        const int ahc_cluster_count = 1 + *std::max_element(ahc_clusters.begin(), ahc_clusters.end());

        std::vector<float> qinit(static_cast<size_t>(num_train_embeddings) * static_cast<size_t>(ahc_cluster_count), 0.0f);
        for (int i = 0; i < num_train_embeddings; ++i) {
            qinit[static_cast<size_t>(i) * static_cast<size_t>(ahc_cluster_count) +
                  static_cast<size_t>(ahc_clusters[static_cast<size_t>(i)])] = 1.0f;
        }
        qinit = softmax_rows(qinit, num_train_embeddings, ahc_cluster_count, 7.0f);

        std::vector<float> features(static_cast<size_t>(num_train_embeddings) * static_cast<size_t>(plda_dim_), 0.0f);
        std::vector<float> centered(static_cast<size_t>(xvec_input_dim_), 0.0f);
        std::vector<float> projected(static_cast<size_t>(xvec_output_dim_), 0.0f);
        std::vector<float> transformed(static_cast<size_t>(plda_dim_), 0.0f);
        for (int index = 0; index < num_train_embeddings; ++index) {
            const float * embedding =
                train_embeddings.data() + static_cast<size_t>(index) * static_cast<size_t>(problem.embedding_dim);
            for (int d = 0; d < xvec_input_dim_; ++d) {
                centered[static_cast<size_t>(d)] = embedding[d] - xvec_mean1_[static_cast<size_t>(d)];
            }
            normalize_scaled(centered, std::sqrt(static_cast<float>(xvec_input_dim_)));

            std::fill(projected.begin(), projected.end(), 0.0f);
            for (int in_dim = 0; in_dim < xvec_input_dim_; ++in_dim) {
                const float value = centered[static_cast<size_t>(in_dim)];
                const size_t offset = static_cast<size_t>(in_dim) * static_cast<size_t>(xvec_output_dim_);
                for (int out_dim = 0; out_dim < xvec_output_dim_; ++out_dim) {
                    projected[static_cast<size_t>(out_dim)] += value * xvec_lda_[offset + static_cast<size_t>(out_dim)];
                }
            }
            for (int out_dim = 0; out_dim < xvec_output_dim_; ++out_dim) {
                projected[static_cast<size_t>(out_dim)] -= xvec_mean2_[static_cast<size_t>(out_dim)];
            }
            normalize_scaled(projected, std::sqrt(static_cast<float>(xvec_output_dim_)));

            for (int out_dim = 0; out_dim < plda_dim_; ++out_dim) {
                float value = 0.0f;
                const size_t offset = static_cast<size_t>(out_dim) * static_cast<size_t>(xvec_output_dim_);
                for (int in_dim = 0; in_dim < xvec_output_dim_; ++in_dim) {
                    value += (projected[static_cast<size_t>(in_dim)] - plda_mu_[static_cast<size_t>(in_dim)]) *
                             plda_transform_[offset + static_cast<size_t>(in_dim)];
                }
                transformed[static_cast<size_t>(out_dim)] = value;
            }

            std::memcpy(
                features.data() + static_cast<size_t>(index) * static_cast<size_t>(plda_dim_),
                transformed.data(),
                static_cast<size_t>(plda_dim_) * sizeof(float)
            );
        }

        const vbx_output vbx = run_vbx(
            features,
            num_train_embeddings,
            plda_dim_,
            plda_phi_,
            config_.clustering_fa,
            config_.clustering_fb,
            qinit,
            ahc_cluster_count
        );

        std::vector<int> kept_speakers;
        for (int speaker = 0; speaker < vbx.num_speakers; ++speaker) {
            if (vbx.pi[static_cast<size_t>(speaker)] > 1.0e-7) {
                kept_speakers.push_back(speaker);
            }
        }
        if (kept_speakers.empty()) {
            kept_speakers.push_back(static_cast<int>(
                std::distance(vbx.pi.begin(), std::max_element(vbx.pi.begin(), vbx.pi.end()))
            ));
        }

        num_result_clusters = static_cast<int>(kept_speakers.size());
        centroids.assign(static_cast<size_t>(num_result_clusters) * static_cast<size_t>(problem.embedding_dim), 0.0f);
        std::vector<double> centroid_weights(static_cast<size_t>(num_result_clusters), 0.0);
        for (int cluster = 0; cluster < num_result_clusters; ++cluster) {
            const int speaker = kept_speakers[static_cast<size_t>(cluster)];
            for (int frame = 0; frame < num_train_embeddings; ++frame) {
                const double weight =
                    vbx.gamma[static_cast<size_t>(frame) * static_cast<size_t>(vbx.num_speakers) +
                              static_cast<size_t>(speaker)];
                centroid_weights[static_cast<size_t>(cluster)] += weight;
                const float * embedding =
                    train_embeddings.data() + static_cast<size_t>(frame) * static_cast<size_t>(problem.embedding_dim);
                float * centroid =
                    centroids.data() + static_cast<size_t>(cluster) * static_cast<size_t>(problem.embedding_dim);
                for (int d = 0; d < problem.embedding_dim; ++d) {
                    centroid[d] += static_cast<float>(weight) * embedding[d];
                }
            }
        }
        for (int cluster = 0; cluster < num_result_clusters; ++cluster) {
            const double weight = centroid_weights[static_cast<size_t>(cluster)];
            if (weight <= 0.0) {
                continue;
            }
            const float inv_weight = static_cast<float>(1.0 / weight);
            float * centroid =
                centroids.data() + static_cast<size_t>(cluster) * static_cast<size_t>(problem.embedding_dim);
            for (int d = 0; d < problem.embedding_dim; ++d) {
                centroid[d] *= inv_weight;
            }
        }

        int target_clusters = requested_clusters;
        if (num_result_clusters < min_clusters) {
            target_clusters = min_clusters;
        } else if (num_result_clusters > max_clusters) {
            target_clusters = max_clusters;
        }

        if (target_clusters > 0 && target_clusters != num_result_clusters) {
            constrained_assignment = false;
            const std::vector<int> kmeans_clusters = kmeans_refine(
                normalized_train_embeddings,
                num_train_embeddings,
                problem.embedding_dim,
                target_clusters
            );

            num_result_clusters = target_clusters;
            centroids.assign(static_cast<size_t>(num_result_clusters) * static_cast<size_t>(problem.embedding_dim), 0.0f);
            std::vector<int> cluster_counts(static_cast<size_t>(num_result_clusters), 0);
            for (int index = 0; index < num_train_embeddings; ++index) {
                const int cluster = kmeans_clusters[static_cast<size_t>(index)];
                ++cluster_counts[static_cast<size_t>(cluster)];
                const float * embedding =
                    train_embeddings.data() + static_cast<size_t>(index) * static_cast<size_t>(problem.embedding_dim);
                float * centroid =
                    centroids.data() + static_cast<size_t>(cluster) * static_cast<size_t>(problem.embedding_dim);
                for (int d = 0; d < problem.embedding_dim; ++d) {
                    centroid[d] += embedding[d];
                }
            }
            for (int cluster = 0; cluster < num_result_clusters; ++cluster) {
                const int count = cluster_counts[static_cast<size_t>(cluster)];
                if (count <= 0) {
                    continue;
                }
                const float inv_count = 1.0f / static_cast<float>(count);
                float * centroid =
                    centroids.data() + static_cast<size_t>(cluster) * static_cast<size_t>(problem.embedding_dim);
                for (int d = 0; d < problem.embedding_dim; ++d) {
                    centroid[d] *= inv_count;
                }
            }
        }
    }

    result.num_chunks = problem.num_chunks;
    result.num_speakers = problem.num_speakers;
    result.num_clusters = num_result_clusters;
    result.embedding_dim = problem.embedding_dim;
    result.centroids = centroids;
    result.soft_clusters.assign(
        static_cast<size_t>(problem.num_chunks) * static_cast<size_t>(problem.num_speakers) *
            static_cast<size_t>(num_result_clusters),
        std::numeric_limits<float>::quiet_NaN()
    );

    for (int chunk = 0; chunk < problem.num_chunks; ++chunk) {
        for (int speaker = 0; speaker < problem.num_speakers; ++speaker) {
            const float * embedding = problem.embedding_ptr(chunk, speaker);
            const bool valid = !contains_nan(embedding, problem.embedding_dim);
            for (int cluster = 0; cluster < num_result_clusters; ++cluster) {
                float score = std::numeric_limits<float>::quiet_NaN();
                if (valid) {
                    const float * centroid =
                        result.centroids.data() +
                        static_cast<size_t>(cluster) * static_cast<size_t>(problem.embedding_dim);
                    score = 1.0f + cosine_similarity(embedding, centroid, problem.embedding_dim);
                }
                result.soft_clusters[(static_cast<size_t>(chunk) * static_cast<size_t>(problem.num_speakers) +
                                      static_cast<size_t>(speaker)) *
                                         static_cast<size_t>(num_result_clusters) +
                                     static_cast<size_t>(cluster)] = score;
            }
        }
    }

    if (constrained_assignment) {
        float min_score = std::numeric_limits<float>::infinity();
        for (float score : result.soft_clusters) {
            if (!std::isnan(score)) {
                min_score = std::min(min_score, score);
            }
        }
        const float inactive_score = std::isfinite(min_score) ? (min_score - 1.0f) : -1.0f;
        for (int chunk = 0; chunk < problem.num_chunks; ++chunk) {
            for (int speaker = 0; speaker < problem.num_speakers; ++speaker) {
                float active_sum = 0.0f;
                for (int frame = 0; frame < problem.num_frames; ++frame) {
                    active_sum += problem.segmentation_at(chunk, frame, speaker);
                }
                if (active_sum > 0.0f) {
                    continue;
                }
                for (int cluster = 0; cluster < num_result_clusters; ++cluster) {
                    result.soft_clusters[(static_cast<size_t>(chunk) * static_cast<size_t>(problem.num_speakers) +
                                          static_cast<size_t>(speaker)) *
                                             static_cast<size_t>(num_result_clusters) +
                                         static_cast<size_t>(cluster)] = inactive_score;
                }
            }
        }
        result.hard_clusters = constrained_argmax(
            result.soft_clusters,
            problem.num_chunks,
            problem.num_speakers,
            num_result_clusters
        );
    } else {
        result.hard_clusters.assign(
            static_cast<size_t>(problem.num_chunks) * static_cast<size_t>(problem.num_speakers),
            0
        );
        for (int chunk = 0; chunk < problem.num_chunks; ++chunk) {
            for (int speaker = 0; speaker < problem.num_speakers; ++speaker) {
                int best_cluster = 0;
                float best_score = -std::numeric_limits<float>::infinity();
                for (int cluster = 0; cluster < num_result_clusters; ++cluster) {
                    const float score =
                        result.soft_clusters[(static_cast<size_t>(chunk) * static_cast<size_t>(problem.num_speakers) +
                                              static_cast<size_t>(speaker)) *
                                                 static_cast<size_t>(num_result_clusters) +
                                             static_cast<size_t>(cluster)];
                    if (score > best_score) {
                        best_score = score;
                        best_cluster = cluster;
                    }
                }
                result.hard_clusters[static_cast<size_t>(chunk) * static_cast<size_t>(problem.num_speakers) +
                                     static_cast<size_t>(speaker)] = best_cluster;
            }
        }
    }

    error_msg_.clear();
    return true;
}

std::string OfflineDiarizer::native_execution_gap() const {
    return "Missing native segmentation forward and native speaker-embedding forward. Community-1 offline clustering is available for precomputed features.";
}

} // namespace q3asr
