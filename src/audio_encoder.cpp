#include "audio_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace q3asr {

namespace {

constexpr int Q3ASR_MAX_NODES = 16384;

int conv2d_output_len(int n) {
    n = (n - 1) / 2 + 1;
    n = (n - 1) / 2 + 1;
    n = (n - 1) / 2 + 1;
    return n;
}

} // namespace

AudioEncoder::~AudioEncoder() {
    if (state_.sched != nullptr) {
        ggml_backend_sched_free(state_.sched);
    }
    if (state_.backend_gpu != nullptr) {
        ggml_backend_free(state_.backend_gpu);
    }
    if (state_.backend_cpu != nullptr) {
        ggml_backend_free(state_.backend_cpu);
    }
    free_model(model_);
}

bool AudioEncoder::load_model(const std::string & model_path, bool use_gpu, int n_threads) {
    error_msg_.clear();
    use_gpu_ = use_gpu;
    n_threads_ = std::max(1, n_threads);

    GGUFLoader loader;
    if (!loader.load(model_path, model_)) {
        error_msg_ = loader.get_error();
        return false;
    }

    state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (state_.backend_cpu == nullptr) {
        error_msg_ = "Failed to initialize the CPU ggml backend for the audio encoder";
        return false;
    }

    if (use_gpu_) {
        state_.backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> bufts;

    if (state_.backend_gpu != nullptr) {
        backends.push_back(state_.backend_gpu);
        bufts.push_back(ggml_backend_get_default_buffer_type(state_.backend_gpu));
    }

    backends.push_back(state_.backend_cpu);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(state_.backend_cpu);
    if (state_.backend_gpu != nullptr) {
        ggml_backend_dev_t gpu_dev = ggml_backend_get_device(state_.backend_gpu);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu_dev);
        if (host_buft != nullptr) {
            cpu_buft = host_buft;
        }
    }
    bufts.push_back(cpu_buft);

    state_.sched = ggml_backend_sched_new(
        backends.data(),
        bufts.data(),
        static_cast<int>(backends.size()),
        Q3ASR_MAX_NODES,
        false,
        true
    );

    if (state_.sched == nullptr) {
        error_msg_ = "Failed to create the audio encoder backend scheduler";
        return false;
    }

    state_.compute_meta.resize(
        ggml_tensor_overhead() * Q3ASR_MAX_NODES +
        ggml_graph_overhead_custom(Q3ASR_MAX_NODES, false)
    );

    return true;
}

bool AudioEncoder::encode(const float * mel_data, int n_mel, int n_frames, std::vector<float> & output) {
    output.clear();

    if (model_.ctx == nullptr) {
        error_msg_ = "Audio encoder model is not loaded";
        return false;
    }

    if (n_mel != model_.config.n_mel_bins) {
        error_msg_ = "Mel bins mismatch: expected " + std::to_string(model_.config.n_mel_bins) +
            ", got " + std::to_string(n_mel);
        return false;
    }

    const int frames_per_window = model_.config.n_window_infer > 0 ? model_.config.n_window_infer : n_frames;

    for (int window_start = 0; window_start < n_frames; window_start += frames_per_window) {
        const int window_len = std::min(frames_per_window, n_frames - window_start);
        std::vector<float> window_mel(static_cast<size_t>(n_mel) * window_len);

        for (int mel = 0; mel < n_mel; ++mel) {
            const float * src = mel_data + static_cast<size_t>(mel) * n_frames + window_start;
            float * dst = window_mel.data() + static_cast<size_t>(mel) * window_len;
            std::memcpy(dst, src, static_cast<size_t>(window_len) * sizeof(float));
        }

        std::vector<float> window_output;
        if (!encode_window(window_mel.data(), window_len, window_output)) {
            return false;
        }

        output.insert(output.end(), window_output.begin(), window_output.end());
    }

    return true;
}

bool AudioEncoder::encode_window(const float * mel_window, int n_frames, std::vector<float> & output) {
    const auto & cfg = model_.config;

    ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx0 = ggml_init(params);
    if (ctx0 == nullptr) {
        error_msg_ = "Failed to initialize the audio encoder graph context";
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, Q3ASR_MAX_NODES, false);
    ggml_tensor * inp_raw = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, static_cast<int64_t>(cfg.n_mel_bins) * n_frames);
    ggml_set_name(inp_raw, "inp_raw");
    ggml_set_input(inp_raw);

    const int flatten_dim = conv2d_output_len(cfg.n_mel_bins) * cfg.downsample_hidden_size;
    const int conv_chunk_frames = cfg.conv_chunk_frames > 0 ? cfg.conv_chunk_frames : n_frames;
    const int n_sub_chunks = (n_frames + conv_chunk_frames - 1) / conv_chunk_frames;

    auto reshape_conv_bias = [&](ggml_tensor * bias) {
        return ggml_reshape_4d(ctx0, bias, 1, 1, bias->ne[2], 1);
    };

    std::vector<ggml_tensor *> sub_embeddings;
    sub_embeddings.reserve(n_sub_chunks);
    int total_tokens = 0;

    for (int sub = 0; sub < n_sub_chunks; ++sub) {
        const int start = sub * conv_chunk_frames;
        const int chunk_len = std::min(conv_chunk_frames, n_frames - start);
        const int tokens = conv2d_output_len(chunk_len);
        total_tokens += tokens;

        ggml_tensor * sub_view = ggml_view_2d(
            ctx0,
            inp_raw,
            chunk_len,
            cfg.n_mel_bins,
            static_cast<size_t>(n_frames) * ggml_type_size(inp_raw->type),
            static_cast<size_t>(start) * ggml_type_size(inp_raw->type)
        );

        ggml_tensor * cur = ggml_reshape_4d(ctx0, ggml_cont(ctx0, sub_view), chunk_len, cfg.n_mel_bins, 1, 1);

        cur = ggml_conv_2d(ctx0, model_.conv2d1_w, cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_add(ctx0, cur, reshape_conv_bias(model_.conv2d1_b));
        cur = ggml_gelu_erf(ctx0, cur);

        cur = ggml_conv_2d(ctx0, model_.conv2d2_w, cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_add(ctx0, cur, reshape_conv_bias(model_.conv2d2_b));
        cur = ggml_gelu_erf(ctx0, cur);

        cur = ggml_conv_2d(ctx0, model_.conv2d3_w, cur, 2, 2, 1, 1, 1, 1);
        cur = ggml_add(ctx0, cur, reshape_conv_bias(model_.conv2d3_b));
        cur = ggml_gelu_erf(ctx0, cur);

        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 2, 0, 1, 3));
        cur = ggml_reshape_2d(ctx0, cur, flatten_dim, tokens);
        cur = ggml_mul_mat(ctx0, model_.conv_out_w, cur);

        ggml_tensor * pos_embd = ggml_view_2d(
            ctx0,
            model_.position_embd,
            model_.position_embd->ne[0],
            tokens,
            model_.position_embd->nb[1],
            0
        );

        cur = ggml_add(ctx0, cur, pos_embd);
        sub_embeddings.push_back(cur);
    }

    ggml_tensor * cur = sub_embeddings.front();
    for (size_t i = 1; i < sub_embeddings.size(); ++i) {
        cur = ggml_concat(ctx0, cur, sub_embeddings[i], 1);
    }

    ggml_tensor * inpL = cur;
    const int head_dim = cfg.d_model / cfg.n_heads;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (const auto & layer : model_.layers) {
        cur = ggml_norm(ctx0, inpL, cfg.layer_norm_eps);
        cur = ggml_mul(ctx0, cur, layer.ln1_w);
        cur = ggml_add(ctx0, cur, layer.ln1_b);

        ggml_tensor * q = ggml_mul_mat(ctx0, layer.attn_q_w, cur);
        q = ggml_add(ctx0, q, layer.attn_q_b);
        ggml_tensor * k = ggml_mul_mat(ctx0, layer.attn_k_w, cur);
        k = ggml_add(ctx0, k, layer.attn_k_b);
        ggml_tensor * v = ggml_mul_mat(ctx0, layer.attn_v_w, cur);
        v = ggml_add(ctx0, v, layer.attn_v_b);

        q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, q, head_dim, cfg.n_heads, total_tokens), 0, 2, 1, 3);
        k = ggml_permute(ctx0, ggml_reshape_3d(ctx0, k, head_dim, cfg.n_heads, total_tokens), 0, 2, 1, 3);
        v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, head_dim, cfg.n_heads, total_tokens), 1, 2, 0, 3));

        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        ggml_tensor * kq_sm = ggml_soft_max_ext(ctx0, kq, nullptr, kq_scale, 0.0f);
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq_sm);
        ggml_tensor * merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        cur = ggml_cont_2d(ctx0, merged, cfg.d_model, total_tokens);
        cur = ggml_mul_mat(ctx0, layer.attn_out_w, cur);
        cur = ggml_add(ctx0, cur, layer.attn_out_b);
        cur = ggml_add(ctx0, cur, inpL);

        ggml_tensor * inp_ff = cur;
        cur = ggml_norm(ctx0, inp_ff, cfg.layer_norm_eps);
        cur = ggml_mul(ctx0, cur, layer.ln2_w);
        cur = ggml_add(ctx0, cur, layer.ln2_b);

        cur = ggml_mul_mat(ctx0, layer.ffn_up_w, cur);
        cur = ggml_add(ctx0, cur, layer.ffn_up_b);
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_mul_mat(ctx0, layer.ffn_down_w, cur);
        cur = ggml_add(ctx0, cur, layer.ffn_down_b);

        inpL = ggml_add(ctx0, cur, inp_ff);
    }

    cur = inpL;
    cur = ggml_norm(ctx0, cur, cfg.layer_norm_eps);
    cur = ggml_mul(ctx0, cur, model_.post_ln_w);
    cur = ggml_add(ctx0, cur, model_.post_ln_b);

    cur = ggml_mul_mat(ctx0, model_.proj1_w, cur);
    cur = ggml_add(ctx0, cur, model_.proj1_b);
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model_.proj2_w, cur);
    cur = ggml_add(ctx0, cur, model_.proj2_b);

    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    bool ok = ggml_backend_sched_alloc_graph(state_.sched, gf);
    if (!ok) {
        error_msg_ = "Failed to allocate the audio encoder graph";
        ggml_free(ctx0);
        return false;
    }

    ggml_tensor * inp = ggml_graph_get_tensor(gf, "inp_raw");
    if (inp == nullptr) {
        error_msg_ = "Failed to look up the audio encoder input tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(
        inp,
        mel_window,
        0,
        static_cast<size_t>(cfg.n_mel_bins) * n_frames * sizeof(float)
    );

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Audio encoder graph execution failed";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    ggml_tensor * out = ggml_graph_get_tensor(gf, "encoder_output");
    if (out == nullptr) {
        error_msg_ = "Failed to look up the audio encoder output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    const int64_t out_dim = out->ne[0];
    const int64_t out_tokens = out->ne[1];
    output.resize(static_cast<size_t>(out_dim * out_tokens));
    ggml_backend_tensor_get(out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);
    return true;
}

} // namespace q3asr
