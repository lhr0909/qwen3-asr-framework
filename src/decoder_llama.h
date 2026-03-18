#pragma once

#include "llama.h"

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

namespace q3asr {

struct decoder_load_params {
    bool use_gpu = true;
    int n_threads = 4;
    int n_batch = 512;
    int n_ctx = 4096;
    int n_gpu_layers = -1;
};

struct decoder_transcribe_params {
    int max_tokens = 256;
    float temperature = 0.0f;
    int n_threads = 4;
    int n_batch = 512;
    int n_ctx = 4096;
    std::string language_hint;
    std::function<void(const std::string &)> raw_text_callback;
};

struct decoder_token_span {
    size_t byte_start = 0;
    size_t byte_end = 0;
    float logprob = 0.0f;
};

void configure_llama_logging();

class LlamaDecoder {
public:
    LlamaDecoder() = default;
    ~LlamaDecoder();

    bool load_model(const std::string & path, const decoder_load_params & params);

    bool decode_audio_embeddings(
        const std::vector<float> & embeddings,
        const decoder_transcribe_params & params,
        std::string & raw_text,
        std::vector<decoder_token_span> * out_token_spans = nullptr
    ) const;

    int hidden_size() const;
    const std::string & get_error() const { return error_msg_; }

private:
    std::vector<llama_token> tokenize(const std::string & text, bool parse_special, bool add_special = false) const;
    std::string detokenize(const std::vector<llama_token> & tokens) const;
    std::string token_to_piece(llama_token token) const;

    bool decode_token_batch(
        llama_context * ctx,
        const llama_token * tokens,
        int count,
        bool request_logits,
        llama_pos & n_past
    ) const;

    bool decode_embedding_batch(
        llama_context * ctx,
        const float * embeddings,
        int count,
        llama_pos & n_past
    ) const;

    llama_token sample_token(llama_context * ctx, float temperature, float * out_logprob = nullptr) const;

    llama_model * model_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    decoder_load_params load_params_{};
    mutable std::string error_msg_;
};

} // namespace q3asr
