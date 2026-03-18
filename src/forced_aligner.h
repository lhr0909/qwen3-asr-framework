#ifndef Q3ASR_FORCED_ALIGNER_H
#define Q3ASR_FORCED_ALIGNER_H

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace q3asr {

struct aligner_load_params {
    bool use_gpu = true;
    int n_threads = 4;
    std::string korean_dict_path;
};

struct aligned_item {
    std::string text;
    float start_time = 0.0f;
    float end_time = 0.0f;
};

struct alignment_result {
    std::vector<aligned_item> items;
    bool success = false;
    std::string error_msg;
    int64_t t_mel_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_total_ms = 0;
};

struct align_runtime_params {
    float max_chunk_seconds = 180.0f;
    float chunk_search_expand_seconds = 5.0f;
    float min_chunk_window_ms = 100.0f;
};

struct split_audio_chunk {
    std::vector<float> samples;
    int original_n_samples = 0;
    float offset_sec = 0.0f;
};

struct normalized_word_span {
    std::string text;
    size_t byte_start = 0;
    size_t byte_end = 0;
};

struct forced_aligner_hparams {
    int32_t audio_encoder_layers = 24;
    int32_t audio_d_model = 1024;
    int32_t audio_attention_heads = 16;
    int32_t audio_ffn_dim = 4096;
    int32_t audio_num_mel_bins = 128;
    int32_t audio_conv_channels = 480;
    float audio_layer_norm_eps = 1e-5f;

    int32_t text_decoder_layers = 28;
    int32_t text_hidden_size = 1024;
    int32_t text_attention_heads = 16;
    int32_t text_kv_heads = 8;
    int32_t text_intermediate_size = 3072;
    int32_t text_head_dim = 128;
    float text_rms_norm_eps = 1e-6f;
    float text_rope_theta = 1000000.0f;
    int32_t vocab_size = 152064;

    int32_t classify_num = 5000;

    int32_t timestamp_token_id = 151705;
    int32_t audio_start_token_id = 151669;
    int32_t audio_end_token_id = 151670;
    int32_t audio_pad_token_id = 151676;
    int32_t pad_token_id = 151643;
    int32_t eos_token_id = 151645;

    int32_t timestamp_segment_time_ms = 80;
};

struct fa_encoder_layer {
    ggml_tensor * attn_q_w = nullptr;
    ggml_tensor * attn_q_b = nullptr;
    ggml_tensor * attn_k_w = nullptr;
    ggml_tensor * attn_k_b = nullptr;
    ggml_tensor * attn_v_w = nullptr;
    ggml_tensor * attn_v_b = nullptr;
    ggml_tensor * attn_out_w = nullptr;
    ggml_tensor * attn_out_b = nullptr;
    ggml_tensor * attn_norm_w = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * ffn_up_w = nullptr;
    ggml_tensor * ffn_up_b = nullptr;
    ggml_tensor * ffn_down_w = nullptr;
    ggml_tensor * ffn_down_b = nullptr;
    ggml_tensor * ffn_norm_w = nullptr;
    ggml_tensor * ffn_norm_b = nullptr;
};

struct fa_decoder_layer {
    ggml_tensor * attn_norm = nullptr;
    ggml_tensor * attn_q = nullptr;
    ggml_tensor * attn_k = nullptr;
    ggml_tensor * attn_v = nullptr;
    ggml_tensor * attn_output = nullptr;
    ggml_tensor * attn_q_norm = nullptr;
    ggml_tensor * attn_k_norm = nullptr;
    ggml_tensor * ffn_norm = nullptr;
    ggml_tensor * ffn_gate = nullptr;
    ggml_tensor * ffn_up = nullptr;
    ggml_tensor * ffn_down = nullptr;
};

struct forced_aligner_model {
    forced_aligner_hparams hparams;

    ggml_tensor * conv2d1_w = nullptr;
    ggml_tensor * conv2d1_b = nullptr;
    ggml_tensor * conv2d2_w = nullptr;
    ggml_tensor * conv2d2_b = nullptr;
    ggml_tensor * conv2d3_w = nullptr;
    ggml_tensor * conv2d3_b = nullptr;
    ggml_tensor * conv_out_w = nullptr;
    ggml_tensor * conv_out_b = nullptr;

    ggml_tensor * ln_post_w = nullptr;
    ggml_tensor * ln_post_b = nullptr;
    ggml_tensor * proj1_w = nullptr;
    ggml_tensor * proj1_b = nullptr;
    ggml_tensor * proj2_w = nullptr;
    ggml_tensor * proj2_b = nullptr;

    std::vector<fa_encoder_layer> encoder_layers;

    ggml_tensor * token_embd = nullptr;
    std::vector<fa_decoder_layer> decoder_layers;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * classify_head_w = nullptr;
    ggml_tensor * classify_head_b = nullptr;

    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;

    void * mmap_addr = nullptr;
    size_t mmap_size = 0;

    std::map<std::string, ggml_tensor *> tensors;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> bpe_ranks;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_set<std::string> ko_dict;
};

struct forced_aligner_state {
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

class ForcedAligner {
public:
    ForcedAligner() = default;
    ~ForcedAligner();

    bool load_model(const std::string & model_path, const aligner_load_params & params);

    alignment_result align(const std::string & audio_path, const std::string & text, const std::string & language);
    alignment_result align(
        const std::string & audio_path,
        const std::string & text,
        const std::string & language,
        const align_runtime_params & runtime_params
    );
    alignment_result align(const float * samples, int n_samples, const std::string & text, const std::string & language);
    alignment_result align(
        const float * samples,
        int n_samples,
        const std::string & text,
        const std::string & language,
        const align_runtime_params & runtime_params
    );

    const std::string & error() const { return error_msg_; }
    bool is_loaded() const { return model_loaded_; }
    std::vector<normalized_word_span> normalize_with_spans(const std::string & text, const std::string & language) const;

private:
    bool parse_hparams(gguf_context * ctx);
    bool bind_tensors();
    bool load_tensor_data(const std::string & path, gguf_context * ctx);
    bool load_vocab(gguf_context * ctx);
    bool load_korean_dict(const std::string & dict_path);

    bool encode_audio(const float * mel_data, int n_mel, int n_frames, std::vector<float> & output);
    bool forward_decoder(const int32_t * tokens, int32_t n_tokens, const float * audio_embd, int32_t n_audio, int32_t audio_start_pos, std::vector<float> & output);

    std::vector<int32_t> fix_timestamp_classes(const std::vector<int32_t> & data) const;
    std::vector<float> classes_to_timestamps(const std::vector<int32_t> & classes) const;
    std::vector<int32_t> extract_timestamp_classes(const std::vector<float> & logits, const std::vector<int32_t> & tokens) const;
    std::vector<int32_t> build_input_tokens(const std::vector<int32_t> & text_tokens, int32_t n_audio_frames) const;
    int32_t find_audio_start_pos(const std::vector<int32_t> & tokens) const;
    std::vector<std::string> normalize_alignment_words(const std::string & text, const std::string & language) const;
    std::vector<int32_t> encode_words_with_timestamps(const std::vector<std::string> & words) const;
    std::vector<int32_t> tokenize_with_timestamps(const std::string & text, std::vector<std::string> & words, const std::string & language) const;
    alignment_result align_words(const float * samples, int n_samples, const std::vector<std::string> & words);
    alignment_result align_chunked(
        const float * samples,
        int n_samples,
        const std::vector<std::string> & words,
        const align_runtime_params & runtime_params
    );

    forced_aligner_model model_;
    forced_aligner_state state_;
    aligner_load_params params_;
    bool model_loaded_ = false;
    std::string error_msg_;
};

void free_forced_aligner_model(forced_aligner_model & model);
std::vector<split_audio_chunk> split_audio_into_chunks(
    const float * samples,
    int n_samples,
    int sample_rate,
    const align_runtime_params & params
);

} // namespace q3asr

#endif
