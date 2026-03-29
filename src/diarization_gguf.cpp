#include "diarization_gguf.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <sstream>

namespace q3asr {

namespace {

std::string get_string(const gguf_context * ctx, const char * key, const std::string & fallback = {}) {
    const int64_t idx = gguf_find_key(ctx, key);
    return idx >= 0 ? gguf_get_val_str(ctx, idx) : fallback;
}

int32_t get_u32(const gguf_context * ctx, const char * key, int32_t fallback = 0) {
    const int64_t idx = gguf_find_key(ctx, key);
    return idx >= 0 ? static_cast<int32_t>(gguf_get_val_u32(ctx, idx)) : fallback;
}

} // namespace

ggml_tensor * diarization_gguf_model::find_tensor(const std::string & name) const {
    return ctx != nullptr ? ggml_get_tensor(ctx, name.c_str()) : nullptr;
}

bool DiarizationGGUFLoader::load(const std::string & path, diarization_gguf_model & model) {
    free_diarization_gguf_model(model);
    error_msg_.clear();

    ggml_context * meta_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };

    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (gguf_ctx == nullptr) {
        error_msg_ = "Failed to open diarization GGUF: " + path;
        return false;
    }

    model.ctx = meta_ctx;

    const bool ok =
        parse_metadata(gguf_ctx, model) &&
        bind_tensor_names(gguf_ctx, model) &&
        load_tensor_data(path, gguf_ctx, model);

    gguf_free(gguf_ctx);

    if (!ok) {
        free_diarization_gguf_model(model);
        return false;
    }

    return true;
}

bool DiarizationGGUFLoader::parse_metadata(const gguf_context * ctx, diarization_gguf_model & model) {
    model.architecture = get_string(ctx, "general.architecture");
    model.name = get_string(ctx, "general.name");
    model.kind = get_string(ctx, "diarization.kind");
    model.serialization_format = get_string(ctx, "diarization.serialization_format");
    model.source_repo = get_string(ctx, "diarization.source_repo");
    model.source_file = get_string(ctx, "diarization.source_file");
    model.config_json = get_string(ctx, "diarization.config_json");
    model.preprocessor_json = get_string(ctx, "diarization.preprocessor_json");
    model.config_yaml = get_string(ctx, "diarization.config_yaml");
    model.hparams_yaml = get_string(ctx, "diarization.hparams_yaml");
    model.hydra_yaml = get_string(ctx, "diarization.hydra_yaml");
    model.overrides_yaml = get_string(ctx, "diarization.overrides_yaml");
    model.hyper_parameters_json = get_string(ctx, "diarization.pytorch.hyper_parameters_json");
    model.pyannote_audio_metadata_json = get_string(ctx, "diarization.pytorch.pyannote_audio_metadata_json");
    model.model_module = get_string(ctx, "diarization.pytorch.model_module");
    model.model_class = get_string(ctx, "diarization.pytorch.model_class");
    model.torch_version = get_string(ctx, "diarization.pytorch.versions.torch");
    model.pyannote_audio_version = get_string(ctx, "diarization.pytorch.versions.pyannote_audio");
    model.specifications_repr = get_string(ctx, "diarization.pytorch.specifications_repr");
    model.tensor_count = get_u32(ctx, "diarization.tensor_count");
    model.top_level_key_count = get_u32(ctx, "diarization.pytorch.top_level_key_count");

    if (
        model.architecture.empty() ||
        model.kind.empty() ||
        model.serialization_format.empty() ||
        model.tensor_count <= 0
    ) {
        std::ostringstream error;
        error << "Invalid or incomplete diarization GGUF metadata"
              << " architecture='" << model.architecture << "'"
              << " kind='" << model.kind << "'"
              << " serialization_format='" << model.serialization_format << "'"
              << " tensor_count=" << model.tensor_count;
        error_msg_ = error.str();
        return false;
    }

    if (model.serialization_format != "pytorch") {
        error_msg_ = "Unsupported diarization GGUF serialization format: " + model.serialization_format;
        return false;
    }

    return true;
}

bool DiarizationGGUFLoader::bind_tensor_names(const gguf_context * ctx, diarization_gguf_model & model) {
    if (model.ctx == nullptr) {
        error_msg_ = "Diarization GGUF metadata context was not created";
        return false;
    }

    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    model.tensor_names.reserve(static_cast<size_t>(n_tensors));
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        if (name == nullptr) {
            error_msg_ = "Encountered a diarization GGUF tensor without a name";
            return false;
        }
        model.tensor_names.emplace_back(name);
        if (ggml_get_tensor(model.ctx, name) == nullptr) {
            error_msg_ = "Missing tensor in diarization GGUF metadata context: " + model.tensor_names.back();
            return false;
        }
    }

    return static_cast<int32_t>(model.tensor_names.size()) == model.tensor_count;
}

bool DiarizationGGUFLoader::load_tensor_data(const std::string & path, const gguf_context * ctx, diarization_gguf_model & model) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error_msg_ = "Failed to open diarization GGUF for mmap: " + path;
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        error_msg_ = "Failed to stat diarization GGUF: " + path;
        close(fd);
        return false;
    }

    void * mmap_addr = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mmap_addr == MAP_FAILED) {
        error_msg_ = "Failed to mmap diarization GGUF: " + path;
        return false;
    }

    model.mmap_addr = mmap_addr;
    model.mmap_size = static_cast<size_t>(st.st_size);

    const size_t data_offset = gguf_get_data_offset(ctx);
    const size_t total_size = static_cast<size_t>(st.st_size) - data_offset;
    auto * data_base = static_cast<uint8_t *>(mmap_addr) + data_offset;

    size_t max_tensor_size = 0;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        max_tensor_size = std::max(max_tensor_size, gguf_get_tensor_size(ctx, i));
    }

    ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu_dev != nullptr) {
        model.buffer = ggml_backend_dev_buffer_from_host_ptr(gpu_dev, data_base, total_size, max_tensor_size);
    }
    if (model.buffer == nullptr) {
        model.buffer = ggml_backend_cpu_buffer_from_ptr(data_base, total_size);
    }
    if (model.buffer == nullptr) {
        error_msg_ = "Failed to create a backend buffer for diarization GGUF weights";
        return false;
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        ggml_tensor * tensor = ggml_get_tensor(model.ctx, name);
        if (tensor == nullptr) {
            continue;
        }
        tensor->buffer = model.buffer;
        tensor->data = data_base + gguf_get_tensor_offset(ctx, i);
    }

    return true;
}

void free_diarization_gguf_model(diarization_gguf_model & model) {
    if (model.buffer != nullptr) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx != nullptr) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.mmap_addr != nullptr) {
        munmap(model.mmap_addr, model.mmap_size);
        model.mmap_addr = nullptr;
        model.mmap_size = 0;
    }
    model.tensor_names.clear();
    model = {};
}

} // namespace q3asr
