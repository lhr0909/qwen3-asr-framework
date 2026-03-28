#include "diarization_gguf.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

std::string get_flag(int argc, char ** argv, const std::string & name, const std::string & fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

bool expect_segmentation_model(const std::string & path) {
    q3asr::DiarizationGGUFLoader loader;
    q3asr::diarization_gguf_model model;
    if (!loader.load(path, model)) {
        std::cerr << "Failed to load segmentation GGUF: " << loader.get_error() << "\n";
        return false;
    }

    bool ok = true;
    ok = ok && model.architecture == "q3asr-diarization";
    ok = ok && model.kind == "speaker-segmentation";
    ok = ok && model.source_repo == "onnx-community/pyannote-segmentation-3.0";
    ok = ok && model.source_file == "onnx/model_fp16.onnx";
    ok = ok && model.ir_version == 8;
    ok = ok && model.initializer_count == 40;
    ok = ok && model.inputs.size() == 1;
    ok = ok && model.inputs[0].name == "input_values";
    ok = ok && model.outputs.size() == 1;
    ok = ok && model.outputs[0].name == "logits";
    ok = ok && model.find_op_count("LSTM") == 4;
    ok = ok && model.find_op_count("Conv") == 2;
    ok = ok && static_cast<int32_t>(model.tensor_names.size()) == model.initializer_count;
    ok = ok && model.find_tensor("sincnet.conv1d.1.weight") != nullptr;

    q3asr::free_diarization_gguf_model(model);
    return ok;
}

bool expect_embedding_model(const std::string & path) {
    q3asr::DiarizationGGUFLoader loader;
    q3asr::diarization_gguf_model model;
    if (!loader.load(path, model)) {
        std::cerr << "Failed to load embedding GGUF: " << loader.get_error() << "\n";
        return false;
    }

    bool ok = true;
    ok = ok && model.architecture == "q3asr-diarization";
    ok = ok && model.kind == "speaker-embedding";
    ok = ok && model.source_repo == "chengdongliang/wespeaker";
    ok = ok && model.source_file == "voxceleb_resnet34_LM.onnx";
    ok = ok && model.ir_version == 7;
    ok = ok && model.initializer_count == 75;
    ok = ok && model.inputs.size() == 1;
    ok = ok && model.inputs[0].name == "feats";
    ok = ok && model.outputs.size() == 1;
    ok = ok && model.outputs[0].name == "embs";
    ok = ok && model.find_op_count("Conv") == 36;
    ok = ok && model.find_op_count("Relu") == 33;
    ok = ok && static_cast<int32_t>(model.tensor_names.size()) == model.initializer_count;
    ok = ok && model.find_tensor("model.seg_1.weight") != nullptr;

    q3asr::free_diarization_gguf_model(model);
    return ok;
}

} // namespace

int main(int argc, char ** argv) {
    const std::string cwd = std::filesystem::current_path().string();
    const std::string segmentation_default = cwd + "/models/gguf/pyannote-segmentation-3.0-fp16.gguf";
    const std::string embedding_default = cwd + "/models/gguf/wespeaker-voxceleb-resnet34-LM-f32.gguf";

    const std::string segmentation_path =
        get_flag(argc, argv, std::string("--segmentation-model"), segmentation_default);
    const std::string embedding_path =
        get_flag(argc, argv, std::string("--embedding-model"), embedding_default);

    if (!std::filesystem::exists(segmentation_path) || !std::filesystem::exists(embedding_path)) {
        std::cerr << "Missing diarization GGUF test artifacts\n";
        return 2;
    }

    const bool ok =
        expect_segmentation_model(segmentation_path) &&
        expect_embedding_model(embedding_path);

    if (!ok) {
        std::cerr << "diarization_gguf_test failed\n";
        return 1;
    }

    std::cout << "diarization_gguf_test passed\n";
    return 0;
}
