#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace q3asr {

struct encoder_layer {
    ggml_tensor * ln1_w = nullptr;
    ggml_tensor * ln1_b = nullptr;
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_q_b = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_k_b = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_v_b = nullptr;
    ggml_tensor * attn_out_w = nullptr;
    ggml_tensor * attn_out_b = nullptr;
    ggml_tensor * ln2_w = nullptr;
    ggml_tensor * ln2_b = nullptr;
    ggml_tensor * ffn_up_w = nullptr;
    ggml_tensor * ffn_up_b = nullptr;
    ggml_tensor * ffn_down_w = nullptr;
    ggml_tensor * ffn_down_b = nullptr;
};

struct audio_encoder_config {
    int32_t n_layers = 0;
    int32_t d_model = 0;
    int32_t n_heads = 0;
    int32_t n_ff = 0;
    int32_t n_mel_bins = 0;
    int32_t projection_dim = 0;
    int32_t downsample_hidden_size = 0;
    int32_t max_source_positions = 0;
    int32_t n_window = 0;
    int32_t n_window_infer = 0;
    int32_t conv_chunk_frames = 0;
    int32_t text_hidden_size = 0;
    float layer_norm_eps = 1e-5f;
};

struct audio_encoder_model {
    audio_encoder_config config;

    ggml_tensor * position_embd = nullptr;
    ggml_tensor * conv2d1_w = nullptr;
    ggml_tensor * conv2d1_b = nullptr;
    ggml_tensor * conv2d2_w = nullptr;
    ggml_tensor * conv2d2_b = nullptr;
    ggml_tensor * conv2d3_w = nullptr;
    ggml_tensor * conv2d3_b = nullptr;
    ggml_tensor * conv_out_w = nullptr;
    ggml_tensor * post_ln_w = nullptr;
    ggml_tensor * post_ln_b = nullptr;
    ggml_tensor * proj1_w = nullptr;
    ggml_tensor * proj1_b = nullptr;
    ggml_tensor * proj2_w = nullptr;
    ggml_tensor * proj2_b = nullptr;

    std::vector<encoder_layer> layers;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    void * mmap_addr = nullptr;
    size_t mmap_size = 0;
};

class GGUFLoader {
public:
    bool load(const std::string & path, audio_encoder_model & model);
    const std::string & get_error() const { return error_msg_; }

private:
    bool parse_config(const gguf_context * ctx, audio_encoder_model & model);
    bool bind_tensors(audio_encoder_model & model);
    bool load_tensor_data(const std::string & path, const gguf_context * ctx, audio_encoder_model & model);

    std::string error_msg_;
};

void free_model(audio_encoder_model & model);

} // namespace q3asr
