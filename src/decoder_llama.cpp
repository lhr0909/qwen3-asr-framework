#include "decoder_llama.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>

namespace q3asr {

namespace {

bool llama_verbose_enabled() {
    const char * env = std::getenv("Q3ASR_LLAMA_VERBOSE");
    return env != nullptr && env[0] != '\0' && std::strcmp(env, "0") != 0;
}

bool contains_replacement_char(const std::string & text) {
    return text.find("\xEF\xBF\xBD") != std::string::npos;
}

bool is_valid_utf8(const std::string & text) {
    const unsigned char * s = reinterpret_cast<const unsigned char *>(text.data());
    const size_t n = text.size();
    size_t i = 0;

    while (i < n) {
        const unsigned char c = s[i];
        size_t need = 0;

        if ((c & 0x80u) == 0) {
            ++i;
            continue;
        } else if ((c & 0xE0u) == 0xC0u) {
            need = 1;
            if (c < 0xC2u) {
                return false;
            }
        } else if ((c & 0xF0u) == 0xE0u) {
            need = 2;
        } else if ((c & 0xF8u) == 0xF0u) {
            need = 3;
            if (c > 0xF4u) {
                return false;
            }
        } else {
            return false;
        }

        if (i + need >= n) {
            return false;
        }

        for (size_t j = 1; j <= need; ++j) {
            if ((s[i + j] & 0xC0u) != 0x80u) {
                return false;
            }
        }

        if (need == 2) {
            if (c == 0xE0u && s[i + 1] < 0xA0u) {
                return false;
            }
            if (c == 0xEDu && s[i + 1] >= 0xA0u) {
                return false;
            }
        } else if (need == 3) {
            if (c == 0xF0u && s[i + 1] < 0x90u) {
                return false;
            }
            if (c == 0xF4u && s[i + 1] >= 0x90u) {
                return false;
            }
        }

        i += need + 1;
    }

    return true;
}

void llama_log_quiet(ggml_log_level level, const char * text, void * /* user_data */) {
    (void) level;
    (void) text;
}

void init_llama_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        configure_llama_logging();
        llama_backend_init();
        std::atexit([]() {
            llama_backend_free();
        });
    });
}

std::string build_prefix_prompt(const std::string & context) {
    return "<|im_start|>system\n" + context + "<|im_end|>\n<|im_start|>user\n<|audio_start|>";
}

std::string build_suffix_prompt(const std::string & language_hint) {
    std::string out = "<|audio_end|><|im_end|>\n<|im_start|>assistant\n";
    if (!language_hint.empty()) {
        out += "language ";
        out += language_hint;
        out += "<asr_text>";
    }
    return out;
}

} // namespace

void configure_llama_logging() {
    static std::once_flag once;
    std::call_once(once, []() {
        if (!llama_verbose_enabled()) {
            llama_log_set(llama_log_quiet, nullptr);
        }
    });
}

LlamaDecoder::~LlamaDecoder() {
    if (model_ != nullptr) {
        llama_model_free(model_);
    }
}

bool LlamaDecoder::load_model(const std::string & path, const decoder_load_params & params) {
    init_llama_once();

    error_msg_.clear();
    load_params_ = params;

    if (model_ != nullptr) {
        llama_model_free(model_);
        model_ = nullptr;
        vocab_ = nullptr;
    }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.use_gpu ? (params.n_gpu_layers >= 0 ? params.n_gpu_layers : 999) : 0;
    model_params.use_mmap = true;
    model_params.use_mlock = false;

    model_ = llama_model_load_from_file(path.c_str(), model_params);
    if (model_ == nullptr) {
        error_msg_ = "Failed to load text model: " + path;
        return false;
    }

    vocab_ = llama_model_get_vocab(model_);
    if (vocab_ == nullptr) {
        error_msg_ = "Failed to get the decoder vocabulary";
        return false;
    }

    return true;
}

bool LlamaDecoder::decode_audio_embeddings(
    const std::vector<float> & embeddings,
    const decoder_transcribe_params & params,
    std::string & raw_text,
    std::vector<decoder_token_span> * out_token_spans
) const {
    raw_text.clear();
    if (out_token_spans != nullptr) {
        out_token_spans->clear();
    }

    if (model_ == nullptr || vocab_ == nullptr) {
        error_msg_ = "Decoder model is not loaded";
        return false;
    }

    const int n_embd = hidden_size();
    if (n_embd <= 0 || embeddings.size() % static_cast<size_t>(n_embd) != 0) {
        error_msg_ = "Audio embedding size does not match the decoder hidden size";
        return false;
    }

    const int n_audio = static_cast<int>(embeddings.size() / static_cast<size_t>(n_embd));
    const std::vector<llama_token> prefix_tokens = tokenize(build_prefix_prompt(params.context), true, true);
    const std::vector<llama_token> suffix_tokens = tokenize(build_suffix_prompt(params.language_hint), true, false);

    if (prefix_tokens.empty() || suffix_tokens.empty()) {
        error_msg_ = "Failed to build the decoder prompt";
        return false;
    }

    const int required_ctx = static_cast<int>(prefix_tokens.size() + suffix_tokens.size()) + n_audio + params.max_tokens + 16;

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = std::max(load_params_.n_ctx, std::max(params.n_ctx, required_ctx));
    ctx_params.n_batch = std::max(32, std::max(load_params_.n_batch, params.n_batch));
    ctx_params.n_ubatch = std::min<uint32_t>(ctx_params.n_batch, 512);
    ctx_params.n_seq_max = 1;
    ctx_params.n_threads = std::max(1, params.n_threads);
    ctx_params.n_threads_batch = std::max(1, params.n_threads);

    llama_context * ctx = llama_init_from_model(model_, ctx_params);
    if (ctx == nullptr) {
        error_msg_ = "Failed to create a llama.cpp decoding context";
        return false;
    }

    llama_pos n_past = 0;
    const bool prompt_ok =
        decode_token_batch(ctx, prefix_tokens.data(), static_cast<int>(prefix_tokens.size()), false, n_past) &&
        decode_embedding_batch(ctx, embeddings.data(), n_audio, n_past) &&
        decode_token_batch(ctx, suffix_tokens.data(), static_cast<int>(suffix_tokens.size()), true, n_past);

    if (!prompt_ok) {
        llama_free(ctx);
        return false;
    }

    for (int i = 0; i < params.max_tokens; ++i) {
        float token_logprob = 0.0f;
        const llama_token token = sample_token(ctx, params.temperature, &token_logprob);
        if (llama_vocab_is_eog(vocab_, token)) {
            break;
        }

        const std::string piece = token_to_piece(token);
        const size_t byte_start = raw_text.size();
        raw_text += piece;
        if (out_token_spans != nullptr) {
            out_token_spans->push_back({byte_start, raw_text.size(), token_logprob});
        }
        if (params.raw_text_callback) {
            if (is_valid_utf8(raw_text) && !contains_replacement_char(raw_text)) {
                params.raw_text_callback(raw_text);
            }
        }
        if (!decode_token_batch(ctx, &token, 1, true, n_past)) {
            llama_free(ctx);
            return false;
        }
    }

    llama_free(ctx);
    return true;
}

int LlamaDecoder::hidden_size() const {
    return model_ != nullptr ? llama_model_n_embd_inp(model_) : 0;
}

std::vector<llama_token> LlamaDecoder::tokenize(const std::string & text, bool parse_special, bool add_special) const {
    std::vector<llama_token> tokens(text.size() + 16);
    int32_t rc = llama_tokenize(
        vocab_,
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        add_special,
        parse_special
    );

    if (rc < 0) {
        tokens.resize(static_cast<size_t>(-rc));
        rc = llama_tokenize(
            vocab_,
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            add_special,
            parse_special
        );
    }

    if (rc < 0) {
        return {};
    }

    tokens.resize(static_cast<size_t>(rc));
    return tokens;
}

std::string LlamaDecoder::detokenize(const std::vector<llama_token> & tokens) const {
    if (tokens.empty()) {
        return {};
    }

    std::vector<char> buf(tokens.size() * 8 + 256);
    int32_t rc = llama_detokenize(
        vocab_,
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        buf.data(),
        static_cast<int32_t>(buf.size()),
        false,
        true
    );

    if (rc < 0) {
        buf.resize(static_cast<size_t>(-rc));
        rc = llama_detokenize(
            vocab_,
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            buf.data(),
            static_cast<int32_t>(buf.size()),
            false,
            true
        );
    }

    if (rc < 0) {
        return {};
    }

    return std::string(buf.data(), static_cast<size_t>(rc));
}

std::string LlamaDecoder::token_to_piece(llama_token token) const {
    std::vector<char> buf(32);
    int32_t rc = llama_token_to_piece(
        vocab_,
        token,
        buf.data(),
        static_cast<int32_t>(buf.size()),
        0,
        true
    );

    if (rc < 0) {
        buf.resize(static_cast<size_t>(-rc));
        rc = llama_token_to_piece(
            vocab_,
            token,
            buf.data(),
            static_cast<int32_t>(buf.size()),
            0,
            true
        );
    }

    if (rc < 0) {
        return {};
    }

    return std::string(buf.data(), static_cast<size_t>(rc));
}

bool LlamaDecoder::decode_token_batch(
    llama_context * ctx,
    const llama_token * tokens,
    int count,
    bool request_logits,
    llama_pos & n_past
) const {
    const int chunk_size = std::max(1, std::min(load_params_.n_batch, count));

    for (int offset = 0; offset < count; offset += chunk_size) {
        const int cur = std::min(chunk_size, count - offset);
        llama_batch batch = llama_batch_init(cur, 0, 1);
        batch.n_tokens = cur;

        for (int i = 0; i < cur; ++i) {
            batch.token[i] = tokens[offset + i];
            batch.pos[i] = n_past + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = request_logits && (offset + i == count - 1);
        }

        const int32_t rc = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (rc != 0) {
            error_msg_ = "llama_decode() failed while processing prompt tokens";
            return false;
        }

        n_past += cur;
    }

    return true;
}

bool LlamaDecoder::decode_embedding_batch(
    llama_context * ctx,
    const float * embeddings,
    int count,
    llama_pos & n_past
) const {
    const int n_embd = hidden_size();
    const int chunk_size = std::max(1, std::min(load_params_.n_batch, count));

    for (int offset = 0; offset < count; offset += chunk_size) {
        const int cur = std::min(chunk_size, count - offset);
        llama_batch batch = llama_batch_init(cur, n_embd, 1);
        batch.n_tokens = cur;

        for (int i = 0; i < cur; ++i) {
            std::memcpy(
                batch.embd + static_cast<size_t>(i) * n_embd,
                embeddings + static_cast<size_t>(offset + i) * n_embd,
                static_cast<size_t>(n_embd) * sizeof(float)
            );
            batch.pos[i] = n_past + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 0;
        }

        const int32_t rc = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (rc != 0) {
            error_msg_ = "llama_decode() failed while injecting audio embeddings";
            return false;
        }

        n_past += cur;
    }

    return true;
}

llama_token LlamaDecoder::sample_token(llama_context * ctx, float temperature, float * out_logprob) const {
    const float * logits = llama_get_logits_ith(ctx, -1);
    const int32_t vocab_size = llama_vocab_n_tokens(vocab_);

    if (temperature <= 0.0f) {
        int32_t max_index = 0;
        float max_logit = logits[0];

        for (int32_t i = 1; i < vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                max_index = i;
            }
        }

        if (out_logprob != nullptr) {
            double sum_exp = 0.0;
            for (int32_t i = 0; i < vocab_size; ++i) {
                sum_exp += std::exp(static_cast<double>(logits[i] - max_logit));
            }
            *out_logprob = -static_cast<float>(std::log(sum_exp));
        }

        return max_index;
    }

    const float inv_temperature = 1.0f / temperature;
    float max_scaled_logit = logits[0] * inv_temperature;
    for (int32_t i = 1; i < vocab_size; ++i) {
        max_scaled_logit = std::max(max_scaled_logit, logits[i] * inv_temperature);
    }

    std::vector<double> weights(static_cast<size_t>(vocab_size));
    double sum_exp = 0.0;
    for (int32_t i = 0; i < vocab_size; ++i) {
        const double scaled = static_cast<double>(logits[i] * inv_temperature - max_scaled_logit);
        const double weight = std::exp(scaled);
        weights[static_cast<size_t>(i)] = weight;
        sum_exp += weight;
    }

    thread_local std::mt19937 rng(std::random_device{}());
    std::discrete_distribution<int32_t> dist(weights.begin(), weights.end());
    const int32_t sampled = dist(rng);

    if (out_logprob != nullptr) {
        const double chosen_prob = weights[static_cast<size_t>(sampled)] / sum_exp;
        *out_logprob = static_cast<float>(std::log(std::max(chosen_prob, std::numeric_limits<double>::min())));
    }

    return sampled;
}

} // namespace q3asr
