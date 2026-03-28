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
    ok = ok && model.serialization_format == "onnx";
    ok = ok && model.source_repo == "onnx-community/pyannote-segmentation-3.0";
    ok = ok && model.source_file == "onnx/model_fp16.onnx";
    ok = ok && model.ir_version == 8;
    ok = ok && model.initializer_count == 40;
    ok = ok && model.tensor_count == model.initializer_count;
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
    ok = ok && model.serialization_format == "onnx";
    ok = ok && model.source_repo == "chengdongliang/wespeaker";
    ok = ok && model.source_file == "voxceleb_resnet34_LM.onnx";
    ok = ok && model.ir_version == 7;
    ok = ok && model.initializer_count == 75;
    ok = ok && model.tensor_count == model.initializer_count;
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

bool expect_segmentation_model_pytorch(const std::string & path) {
    q3asr::DiarizationGGUFLoader loader;
    q3asr::diarization_gguf_model model;
    if (!loader.load(path, model)) {
        std::cerr << "Failed to load PyTorch segmentation GGUF: " << loader.get_error() << "\n";
        return false;
    }

    bool ok = true;
    ok = ok && model.architecture == "q3asr-diarization";
    ok = ok && model.kind == "speaker-segmentation";
    ok = ok && model.serialization_format == "pytorch";
    ok = ok && model.source_repo == "pyannote/segmentation-3.0";
    ok = ok && model.source_file == "pytorch_model.bin";
    ok = ok && model.tensor_count == 54;
    ok = ok && model.top_level_key_count == 5;
    ok = ok && model.model_module == "pyannote.audio.models.segmentation.PyanNet";
    ok = ok && model.model_class == "PyanNet";
    ok = ok && model.torch_version == "2.0.1+cu117";
    ok = ok && model.pyannote_audio_version == "3.0.0";
    ok = ok && !model.config_json.empty();
    ok = ok && !model.preprocessor_json.empty();
    ok = ok && !model.config_yaml.empty();
    ok = ok && !model.hyper_parameters_json.empty();
    ok = ok && !model.pyannote_audio_metadata_json.empty();
    ok = ok && !model.specifications_repr.empty();
    ok = ok && model.inputs.empty();
    ok = ok && model.outputs.empty();
    ok = ok && model.op_counts.empty();
    ok = ok && static_cast<int32_t>(model.tensor_names.size()) == model.tensor_count;
    ok = ok && model.find_tensor("sincnet.conv1d.1.weight") != nullptr;

    q3asr::free_diarization_gguf_model(model);
    return ok;
}

bool expect_embedding_model_pytorch(const std::string & path) {
    q3asr::DiarizationGGUFLoader loader;
    q3asr::diarization_gguf_model model;
    if (!loader.load(path, model)) {
        std::cerr << "Failed to load PyTorch embedding GGUF: " << loader.get_error() << "\n";
        return false;
    }

    bool ok = true;
    ok = ok && model.architecture == "q3asr-diarization";
    ok = ok && model.kind == "speaker-embedding";
    ok = ok && model.serialization_format == "pytorch";
    ok = ok && model.source_repo == "pyannote/wespeaker-voxceleb-resnet34-LM";
    ok = ok && model.source_file == "pytorch_model.bin";
    ok = ok && model.tensor_count == 218;
    ok = ok && model.top_level_key_count == 3;
    ok = ok && model.model_module == "pyannote.audio.models.embedding.wespeaker";
    ok = ok && model.model_class == "WeSpeakerResNet34";
    ok = ok && model.torch_version == "2.0.1";
    ok = ok && model.pyannote_audio_version == "3.1.0";
    ok = ok && !model.config_yaml.empty();
    ok = ok && model.config_json.empty();
    ok = ok && model.preprocessor_json.empty();
    ok = ok && model.hyper_parameters_json == "{}";
    ok = ok && !model.pyannote_audio_metadata_json.empty();
    ok = ok && !model.specifications_repr.empty();
    ok = ok && model.inputs.empty();
    ok = ok && model.outputs.empty();
    ok = ok && model.op_counts.empty();
    ok = ok && static_cast<int32_t>(model.tensor_names.size()) == model.tensor_count;
    ok = ok && model.find_tensor("resnet.conv1.weight") != nullptr;

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
    const std::string segmentation_format =
        get_flag(argc, argv, std::string("--segmentation-format"), std::string("onnx"));
    const std::string embedding_format =
        get_flag(argc, argv, std::string("--embedding-format"), std::string("onnx"));

    if (!std::filesystem::exists(segmentation_path) || !std::filesystem::exists(embedding_path)) {
        std::cerr << "Missing diarization GGUF test artifacts\n";
        return 2;
    }

    const bool ok =
        (segmentation_format == "pytorch"
             ? expect_segmentation_model_pytorch(segmentation_path)
             : expect_segmentation_model(segmentation_path)) &&
        (embedding_format == "pytorch"
             ? expect_embedding_model_pytorch(embedding_path)
             : expect_embedding_model(embedding_path));

    if (!ok) {
        std::cerr << "diarization_gguf_test failed\n";
        return 1;
    }

    std::cout << "diarization_gguf_test passed\n";
    return 0;
}
