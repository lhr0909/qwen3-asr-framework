#include "offline_diarizer.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

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
        cwd + "/models/gguf/pyannote-segmentation-3.0-pytorch-f32.gguf"
    );
    assets.embedding_model_path = get_flag(
        argc,
        argv,
        "--embedding-model",
        cwd + "/models/gguf/pyannote-wespeaker-voxceleb-resnet34-LM-pytorch-f32.gguf"
    );
    assets.plda_model_path = get_flag(
        argc,
        argv,
        "--plda-model",
        assets.community1_bundle_dir + "/plda/plda.npz"
    );
    assets.xvec_transform_path = get_flag(
        argc,
        argv,
        "--xvec-transform",
        assets.community1_bundle_dir + "/plda/xvec_transform.npz"
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
    ok = ok && diarizer.segmentation_model().find_tensor("sincnet.conv1d.1.weight") != nullptr;
    ok = ok && diarizer.embedding_model().find_tensor("resnet.conv1.weight") != nullptr;
    ok = ok && !diarizer.native_execution_available();
    ok = ok && diarizer.native_execution_gap().find("VBx") != std::string::npos;

    if (!ok) {
        std::cerr << "offline_diarizer_test failed\n";
        return 1;
    }

    std::cout << "offline_diarizer_test passed\n";
    return 0;
}
