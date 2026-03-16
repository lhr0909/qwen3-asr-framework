#include "gguf_loader.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstdio>

namespace q3asr {

namespace {

int32_t get_u32(const gguf_context * ctx, const char * key, int32_t fallback) {
    const int64_t idx = gguf_find_key(ctx, key);
    return idx >= 0 ? static_cast<int32_t>(gguf_get_val_u32(ctx, idx)) : fallback;
}

float get_f32(const gguf_context * ctx, const char * key, float fallback) {
    const int64_t idx = gguf_find_key(ctx, key);
    return idx >= 0 ? gguf_get_val_f32(ctx, idx) : fallback;
}

ggml_tensor * require_tensor(ggml_context * ctx, const std::string & name, std::string & error_msg) {
    ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
    if (tensor == nullptr) {
        error_msg = "Missing tensor in mmproj GGUF: " + name;
    }
    return tensor;
}

} // namespace

bool GGUFLoader::load(const std::string & path, audio_encoder_model & model) {
    free_model(model);

    ggml_context * meta_ctx = nullptr;
    gguf_init_params params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };

    gguf_context * gguf_ctx = gguf_init_from_file(path.c_str(), params);
    if (gguf_ctx == nullptr) {
        error_msg_ = "Failed to open mmproj GGUF: " + path;
        return false;
    }

    model.ctx = meta_ctx;

    const bool ok =
        parse_config(gguf_ctx, model) &&
        bind_tensors(model) &&
        load_tensor_data(path, gguf_ctx, model);

    gguf_free(gguf_ctx);

    if (!ok) {
        free_model(model);
        return false;
    }

    return true;
}

bool GGUFLoader::parse_config(const gguf_context * ctx, audio_encoder_model & model) {
    auto & cfg = model.config;
    cfg.n_layers = get_u32(ctx, "clip.audio.block_count", 0);
    cfg.d_model = get_u32(ctx, "clip.audio.embedding_length", 0);
    cfg.n_heads = get_u32(ctx, "clip.audio.attention.head_count", 0);
    cfg.n_ff = get_u32(ctx, "clip.audio.feed_forward_length", 0);
    cfg.n_mel_bins = get_u32(ctx, "clip.audio.num_mel_bins", 0);
    cfg.projection_dim = get_u32(ctx, "clip.audio.projection_dim", 0);
    cfg.downsample_hidden_size = get_u32(ctx, "clip.audio.downsample_hidden_size", 0);
    cfg.max_source_positions = get_u32(ctx, "clip.audio.max_source_positions", 0);
    cfg.n_window = get_u32(ctx, "clip.audio.n_window", 0);
    cfg.n_window_infer = get_u32(ctx, "clip.audio.n_window_infer", 0);
    // Qwen3-ASR uses `conv_chunksize` as a batching knob for how many already-split
    // conv chunks to process at once. The actual time-axis conv chunk is always
    // `n_window * 2` mel frames, which is what the encoder graph expects.
    cfg.conv_chunk_frames = cfg.n_window > 0 ? cfg.n_window * 2 : 0;
    cfg.text_hidden_size = get_u32(ctx, "clip.audio.text_hidden_size", cfg.projection_dim);
    cfg.layer_norm_eps = get_f32(ctx, "clip.audio.attention.layer_norm_epsilon", 1e-5f);

    if (
        cfg.n_layers <= 0 ||
        cfg.d_model <= 0 ||
        cfg.n_heads <= 0 ||
        cfg.n_ff <= 0 ||
        cfg.n_mel_bins <= 0 ||
        cfg.projection_dim <= 0 ||
        cfg.downsample_hidden_size <= 0 ||
        cfg.text_hidden_size <= 0
    ) {
        error_msg_ = "Invalid or incomplete audio encoder metadata in mmproj GGUF";
        return false;
    }

    return true;
}

bool GGUFLoader::bind_tensors(audio_encoder_model & model) {
    if (model.ctx == nullptr) {
        error_msg_ = "Audio encoder metadata context was not created";
        return false;
    }

    model.position_embd = require_tensor(model.ctx, "a.position_embd.weight", error_msg_);
    model.conv2d1_w = require_tensor(model.ctx, "a.conv2d.1.weight", error_msg_);
    model.conv2d1_b = require_tensor(model.ctx, "a.conv2d.1.bias", error_msg_);
    model.conv2d2_w = require_tensor(model.ctx, "a.conv2d.2.weight", error_msg_);
    model.conv2d2_b = require_tensor(model.ctx, "a.conv2d.2.bias", error_msg_);
    model.conv2d3_w = require_tensor(model.ctx, "a.conv2d.3.weight", error_msg_);
    model.conv2d3_b = require_tensor(model.ctx, "a.conv2d.3.bias", error_msg_);
    model.conv_out_w = require_tensor(model.ctx, "a.conv_out.weight", error_msg_);
    model.post_ln_w = require_tensor(model.ctx, "a.post_ln.weight", error_msg_);
    model.post_ln_b = require_tensor(model.ctx, "a.post_ln.bias", error_msg_);
    model.proj1_w = require_tensor(model.ctx, "mm.a.mlp.1.weight", error_msg_);
    model.proj1_b = require_tensor(model.ctx, "mm.a.mlp.1.bias", error_msg_);
    model.proj2_w = require_tensor(model.ctx, "mm.a.mlp.2.weight", error_msg_);
    model.proj2_b = require_tensor(model.ctx, "mm.a.mlp.2.bias", error_msg_);

    if (!error_msg_.empty()) {
        return false;
    }

    model.layers.resize(model.config.n_layers);
    for (int32_t i = 0; i < model.config.n_layers; ++i) {
        auto & layer = model.layers[i];
        const std::string prefix = "a.blk." + std::to_string(i) + ".";

        layer.ln1_w = require_tensor(model.ctx, prefix + "ln1.weight", error_msg_);
        layer.ln1_b = require_tensor(model.ctx, prefix + "ln1.bias", error_msg_);
        layer.attn_q_w = require_tensor(model.ctx, prefix + "attn_q.weight", error_msg_);
        layer.attn_q_b = require_tensor(model.ctx, prefix + "attn_q.bias", error_msg_);
        layer.attn_k_w = require_tensor(model.ctx, prefix + "attn_k.weight", error_msg_);
        layer.attn_k_b = require_tensor(model.ctx, prefix + "attn_k.bias", error_msg_);
        layer.attn_v_w = require_tensor(model.ctx, prefix + "attn_v.weight", error_msg_);
        layer.attn_v_b = require_tensor(model.ctx, prefix + "attn_v.bias", error_msg_);
        layer.attn_out_w = require_tensor(model.ctx, prefix + "attn_out.weight", error_msg_);
        layer.attn_out_b = require_tensor(model.ctx, prefix + "attn_out.bias", error_msg_);
        layer.ln2_w = require_tensor(model.ctx, prefix + "ln2.weight", error_msg_);
        layer.ln2_b = require_tensor(model.ctx, prefix + "ln2.bias", error_msg_);
        layer.ffn_up_w = require_tensor(model.ctx, prefix + "ffn_up.weight", error_msg_);
        layer.ffn_up_b = require_tensor(model.ctx, prefix + "ffn_up.bias", error_msg_);
        layer.ffn_down_w = require_tensor(model.ctx, prefix + "ffn_down.weight", error_msg_);
        layer.ffn_down_b = require_tensor(model.ctx, prefix + "ffn_down.bias", error_msg_);

        if (!error_msg_.empty()) {
            return false;
        }
    }

    return true;
}

bool GGUFLoader::load_tensor_data(const std::string & path, const gguf_context * ctx, audio_encoder_model & model) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error_msg_ = "Failed to open mmproj file for mmap: " + path;
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        error_msg_ = "Failed to stat mmproj file: " + path;
        close(fd);
        return false;
    }

    void * mmap_addr = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mmap_addr == MAP_FAILED) {
        error_msg_ = "Failed to mmap mmproj file: " + path;
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
        error_msg_ = "Failed to create a backend buffer for mmproj weights";
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

void free_model(audio_encoder_model & model) {
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

    model.layers.clear();
    model = {};
}

} // namespace q3asr
