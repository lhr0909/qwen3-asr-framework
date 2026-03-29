#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstddef>
#include <string>
#include <vector>

namespace q3asr {

struct diarization_gguf_model {
    std::string architecture;
    std::string name;
    std::string kind;
    std::string serialization_format;
    std::string source_repo;
    std::string source_file;
    std::string config_json;
    std::string preprocessor_json;
    std::string config_yaml;
    std::string hparams_yaml;
    std::string hydra_yaml;
    std::string overrides_yaml;
    std::string hyper_parameters_json;
    std::string pyannote_audio_metadata_json;
    std::string model_module;
    std::string model_class;
    std::string torch_version;
    std::string pyannote_audio_version;
    std::string specifications_repr;
    int32_t tensor_count = 0;
    int32_t top_level_key_count = 0;
    std::vector<std::string> tensor_names;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    void * mmap_addr = nullptr;
    size_t mmap_size = 0;

    ggml_tensor * find_tensor(const std::string & name) const;
};

class DiarizationGGUFLoader {
public:
    bool load(const std::string & path, diarization_gguf_model & model);
    const std::string & get_error() const { return error_msg_; }

private:
    bool parse_metadata(const gguf_context * ctx, diarization_gguf_model & model);
    bool bind_tensor_names(const gguf_context * ctx, diarization_gguf_model & model);
    bool load_tensor_data(const std::string & path, const gguf_context * ctx, diarization_gguf_model & model);

    std::string error_msg_;
};

void free_diarization_gguf_model(diarization_gguf_model & model);

} // namespace q3asr
