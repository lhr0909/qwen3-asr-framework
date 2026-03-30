#include "offline_diarizer.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace {

using json = nlohmann::json;

bool approx_equal(float lhs, float rhs, float epsilon = 1.0e-5f) {
    return std::fabs(lhs - rhs) <= epsilon;
}

std::string get_flag(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

float cosine_similarity(const float * lhs, const float * rhs, int count) {
    float dot = 0.0f;
    float lhs_norm = 0.0f;
    float rhs_norm = 0.0f;
    for (int i = 0; i < count; ++i) {
        dot += lhs[i] * rhs[i];
        lhs_norm += lhs[i] * lhs[i];
        rhs_norm += rhs[i] * rhs[i];
    }

    lhs_norm = std::sqrt(lhs_norm);
    rhs_norm = std::sqrt(rhs_norm);
    if (lhs_norm <= 1.0e-6f || rhs_norm <= 1.0e-6f) {
        return 0.0f;
    }
    return dot / (lhs_norm * rhs_norm);
}

std::vector<int> best_centroid_permutation(
    const std::vector<float> & actual,
    const std::vector<float> & expected,
    int num_clusters,
    int embedding_dim
) {
    std::vector<int> permutation(static_cast<size_t>(num_clusters), 0);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<int> best = permutation;
    float best_score = -std::numeric_limits<float>::infinity();

    do {
        float score = 0.0f;
        for (int cluster = 0; cluster < num_clusters; ++cluster) {
            const float * actual_centroid =
                actual.data() + static_cast<size_t>(cluster) * static_cast<size_t>(embedding_dim);
            const float * expected_centroid =
                expected.data() + static_cast<size_t>(permutation[static_cast<size_t>(cluster)]) *
                                      static_cast<size_t>(embedding_dim);
            score += cosine_similarity(actual_centroid, expected_centroid, embedding_dim);
        }
        if (score > best_score) {
            best_score = score;
            best = permutation;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    return best;
}

q3asr::offline_diarizer_problem load_fixture(const std::string & path, std::vector<int> & expected_hard_clusters, std::vector<float> & expected_centroids) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open fixture: " + path);
    }

    const json fixture = json::parse(stream, nullptr, true, true);
    q3asr::offline_diarizer_problem problem;
    problem.num_chunks = fixture.at("num_chunks").get<int>();
    problem.num_frames = fixture.at("num_frames").get<int>();
    problem.num_speakers = fixture.at("num_speakers").get<int>();
    problem.embedding_dim = fixture.at("embedding_dim").get<int>();
    problem.num_clusters = fixture.at("num_clusters").get<int>();
    problem.min_clusters = problem.num_clusters;
    problem.max_clusters = problem.num_clusters;
    problem.binary_segmentations = fixture.at("binary_segmentations").get<std::vector<float>>();
    problem.embeddings = fixture.at("embeddings").get<std::vector<float>>();
    expected_hard_clusters = fixture.at("expected_hard_clusters").get<std::vector<int>>();
    expected_centroids = fixture.at("expected_centroids").get<std::vector<float>>();
    return problem;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string cwd = std::filesystem::current_path().string();
    q3asr::offline_diarizer_assets assets;
    assets.community1_bundle_dir = get_flag(
        argc,
        argv,
        "--community1-bundle-dir",
        cwd + "/models/hf/diarization/pyannote-speaker-diarization-community-1"
    );
    assets.segmentation_model_path = get_flag(
        argc,
        argv,
        "--segmentation-model",
        cwd + "/models/gguf/pyannote-speaker-diarization-community-1-segmentation-pytorch-f32.gguf"
    );
    assets.embedding_model_path = get_flag(
        argc,
        argv,
        "--embedding-model",
        cwd + "/models/gguf/pyannote-speaker-diarization-community-1-embedding-pytorch-f32.gguf"
    );
    assets.clustering_model_path = get_flag(
        argc,
        argv,
        "--clustering-model",
        cwd + "/models/gguf/pyannote-speaker-diarization-community-1-plda-f32.gguf"
    );
    const std::string reference_fixture = get_flag(
        argc,
        argv,
        "--reference-fixture",
        cwd + "/testdata/diarization/community1-offline-20s-long-audio.json"
    );

    q3asr::OfflineDiarizer diarizer;
    if (!diarizer.load_community1(assets)) {
        std::cerr << diarizer.get_error() << "\n";
        std::cerr << "offline_diarizer_test failed\n";
        return 1;
    }

    bool ok = true;
    ok = ok && approx_equal(diarizer.config().clustering_threshold, 0.6f);
    ok = ok && approx_equal(diarizer.config().clustering_fa, 0.07f);
    ok = ok && approx_equal(diarizer.config().clustering_fb, 0.8f);
    ok = ok && approx_equal(diarizer.config().min_duration_off, 0.0f);
    ok = ok && diarizer.segmentation_model().find_tensor("sincnet.wav_norm1d.weight") != nullptr;
    ok = ok && diarizer.embedding_model().find_tensor("resnet.conv1.weight") != nullptr;
    ok = ok && diarizer.clustering_model().find_tensor("xvec.lda") != nullptr;
    ok = ok && diarizer.native_clustering_available();
    ok = ok && !diarizer.native_execution_available();
    ok = ok && diarizer.native_execution_gap().find("segmentation forward") != std::string::npos;

    std::vector<int> expected_hard_clusters;
    std::vector<float> expected_centroids;
    q3asr::offline_diarizer_problem problem = load_fixture(
        reference_fixture,
        expected_hard_clusters,
        expected_centroids
    );

    q3asr::offline_diarizer_result result;
    if (!diarizer.cluster(problem, result)) {
        std::cerr << diarizer.get_error() << "\n";
        std::cerr << "offline_diarizer_test failed\n";
        return 1;
    }

    ok = ok && result.num_clusters == problem.num_clusters;
    ok = ok && static_cast<int>(result.hard_clusters.size()) == problem.num_chunks * problem.num_speakers;
    ok = ok && static_cast<int>(result.centroids.size()) == problem.num_clusters * problem.embedding_dim;

    const std::vector<int> permutation = best_centroid_permutation(
        result.centroids,
        expected_centroids,
        problem.num_clusters,
        problem.embedding_dim
    );
    std::vector<int> predicted_to_expected(static_cast<size_t>(problem.num_clusters), 0);
    for (int cluster = 0; cluster < problem.num_clusters; ++cluster) {
        predicted_to_expected[static_cast<size_t>(cluster)] = permutation[static_cast<size_t>(cluster)];
        const float * actual_centroid =
            result.centroids.data() + static_cast<size_t>(cluster) * static_cast<size_t>(problem.embedding_dim);
        const float * expected_centroid =
            expected_centroids.data() +
            static_cast<size_t>(permutation[static_cast<size_t>(cluster)]) * static_cast<size_t>(problem.embedding_dim);
        ok = ok && cosine_similarity(actual_centroid, expected_centroid, problem.embedding_dim) > 0.999f;
    }

    for (size_t i = 0; i < result.hard_clusters.size(); ++i) {
        const int predicted_cluster = result.hard_clusters[i];
        const int mapped_cluster =
            predicted_cluster >= 0 ? predicted_to_expected[static_cast<size_t>(predicted_cluster)] : predicted_cluster;
        ok = ok && mapped_cluster == expected_hard_clusters[i];
    }

    if (!ok) {
        std::cerr << "offline_diarizer_test failed\n";
        return 1;
    }

    std::cout << "offline_diarizer_test passed\n";
    return 0;
}
