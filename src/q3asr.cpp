#include "q3asr.h"

#include "audio_encoder.h"
#include "decoder_llama.h"
#include "mel_spectrogram.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace q3asr {

struct transcript_result {
    std::string raw_text;
    std::string language;
    std::string text;
};

class Recognizer {
public:
    bool load(const q3asr_context_params & params) {
        error_msg_.clear();
        params_ = params;

        configure_llama_logging();

        if (!encoder_.load_model(params.mmproj_model_path, params.use_gpu != 0, params.n_threads)) {
            error_msg_ = encoder_.get_error();
            return false;
        }

        decoder_load_params decoder_params;
        decoder_params.use_gpu = params.use_gpu != 0;
        decoder_params.n_threads = params.n_threads;
        decoder_params.n_batch = params.n_batch;
        decoder_params.n_ctx = params.n_ctx;
        decoder_params.n_gpu_layers = params.n_gpu_layers;

        if (!decoder_.load_model(params.text_model_path, decoder_params)) {
            error_msg_ = decoder_.get_error();
            return false;
        }

        if (encoder_.config().text_hidden_size != decoder_.hidden_size()) {
            error_msg_ = "Audio encoder projection dim does not match the decoder hidden size";
            return false;
        }

        generate_mel_filters(mel_filters_, QWEN_N_MELS, QWEN_N_FFT, QWEN_SAMPLE_RATE);
        return true;
    }

    bool transcribe(
        const float * samples,
        int n_samples,
        const q3asr_transcribe_params & params,
        transcript_result & result
    ) {
        result = {};

        MelSpectrogram mel;
        if (!log_mel_spectrogram(samples, n_samples, mel_filters_, mel, params_.n_threads)) {
            error_msg_ = "Failed to compute the Qwen3-ASR mel spectrogram";
            return false;
        }

        std::vector<float> audio_embeddings;
        if (!encoder_.encode(mel.data.data(), mel.n_mel, mel.n_len, audio_embeddings)) {
            error_msg_ = encoder_.get_error();
            return false;
        }

        decoder_transcribe_params decoder_params;
        decoder_params.max_tokens = params.max_tokens;
        decoder_params.n_threads = params_.n_threads;
        decoder_params.n_batch = params_.n_batch;
        decoder_params.n_ctx = params_.n_ctx;
        decoder_params.language_hint = normalize_language(params.language_hint == nullptr ? "" : params.language_hint);

        if (!decoder_.decode_audio_embeddings(audio_embeddings, decoder_params, result.raw_text)) {
            error_msg_ = decoder_.get_error();
            return false;
        }

        const auto parsed = parse_asr_output(result.raw_text, decoder_params.language_hint);
        result.language = parsed.first;
        result.text = parsed.second;
        return true;
    }

    const std::string & error() const { return error_msg_; }

private:
    static std::string normalize_language(const std::string & language) {
        if (language.empty()) {
            return {};
        }

        std::string trimmed = language;
        trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), trimmed.end());

        if (trimmed.empty()) {
            return {};
        }

        trimmed[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(trimmed[0])));
        for (size_t i = 1; i < trimmed.size(); ++i) {
            trimmed[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(trimmed[i])));
        }

        return trimmed;
    }

    static std::pair<std::string, std::string> parse_asr_output(
        const std::string & raw_text,
        const std::string & forced_language
    ) {
        if (raw_text.empty()) {
            return {"", ""};
        }

        if (!forced_language.empty()) {
            return {forced_language, raw_text};
        }

        const std::string tag = "<asr_text>";
        const size_t tag_pos = raw_text.find(tag);
        if (tag_pos == std::string::npos) {
            return {"", raw_text};
        }

        const std::string meta = raw_text.substr(0, tag_pos);
        const std::string text = raw_text.substr(tag_pos + tag.size());

        std::string language;
        const std::string prefix = "language ";

        for (size_t line_start = 0; line_start < meta.size();) {
            const size_t line_end = meta.find('\n', line_start);
            const std::string line = meta.substr(line_start, line_end == std::string::npos ? std::string::npos : line_end - line_start);

            std::string lowered;
            lowered.reserve(line.size());
            for (unsigned char ch : line) {
                lowered.push_back(static_cast<char>(std::tolower(ch)));
            }

            if (lowered.rfind(prefix, 0) == 0) {
                language = normalize_language(line.substr(prefix.size()));
                break;
            }

            if (line_end == std::string::npos) {
                break;
            }
            line_start = line_end + 1;
        }

        if (language == "None") {
            language.clear();
        }

        return {language, text};
    }

    q3asr_context_params params_{};
    AudioEncoder encoder_;
    LlamaDecoder decoder_;
    MelFilters mel_filters_;
    std::string error_msg_;
};

} // namespace q3asr

struct q3asr_context {
    std::unique_ptr<q3asr::Recognizer> recognizer;
    std::string last_error;
};

namespace {

char * q3asr_strdup(const std::string & value) {
    char * out = static_cast<char *>(std::malloc(value.size() + 1));
    if (out == nullptr) {
        return nullptr;
    }
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

int default_thread_count() {
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw == 0 ? 4 : static_cast<int>(hw);
}

} // namespace

q3asr_context_params q3asr_context_default_params(void) {
    q3asr_context_params params = {};
    params.use_gpu = 1;
    params.n_threads = default_thread_count();
    params.n_batch = 512;
    params.n_ctx = 4096;
    params.n_gpu_layers = -1;
    return params;
}

q3asr_transcribe_params q3asr_transcribe_default_params(void) {
    q3asr_transcribe_params params = {};
    params.max_tokens = 256;
    return params;
}

q3asr_context * q3asr_context_create(const q3asr_context_params * params) {
    const q3asr_context_params effective = params != nullptr ? *params : q3asr_context_default_params();

    if (effective.text_model_path == nullptr || effective.mmproj_model_path == nullptr) {
        return nullptr;
    }

    auto ctx = std::make_unique<q3asr_context>();
    ctx->recognizer = std::make_unique<q3asr::Recognizer>();

    if (!ctx->recognizer->load(effective)) {
        ctx->last_error = ctx->recognizer->error();
        return ctx.release();
    }

    return ctx.release();
}

void q3asr_context_destroy(q3asr_context * ctx) {
    delete ctx;
}

const char * q3asr_context_last_error(const q3asr_context * ctx) {
    return ctx == nullptr ? "" : ctx->last_error.c_str();
}

int q3asr_transcribe_pcm_f32(
    q3asr_context * ctx,
    const float * samples,
    int n_samples,
    const q3asr_transcribe_params * params,
    q3asr_transcribe_result * out_result
) {
    if (ctx == nullptr || ctx->recognizer == nullptr || samples == nullptr || n_samples <= 0 || out_result == nullptr) {
        return 0;
    }

    q3asr_transcribe_result_clear(out_result);
    const q3asr_transcribe_params effective = params != nullptr ? *params : q3asr_transcribe_default_params();

    q3asr::transcript_result result;
    if (!ctx->recognizer->transcribe(samples, n_samples, effective, result)) {
        ctx->last_error = ctx->recognizer->error();
        return 0;
    }

    out_result->raw_text = q3asr_strdup(result.raw_text);
    out_result->language = q3asr_strdup(result.language);
    out_result->text = q3asr_strdup(result.text);
    return 1;
}

int q3asr_transcribe_wav_file(
    q3asr_context * ctx,
    const char * wav_path,
    const q3asr_transcribe_params * params,
    q3asr_transcribe_result * out_result
) {
    if (ctx == nullptr || wav_path == nullptr || out_result == nullptr) {
        return 0;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (!load_wav(wav_path, samples, sample_rate)) {
        ctx->last_error = "Failed to load WAV file";
        return 0;
    }

    if (sample_rate != QWEN_SAMPLE_RATE) {
        ctx->last_error = "Only 16 kHz WAV input is supported in the initial library cut";
        return 0;
    }

    return q3asr_transcribe_pcm_f32(ctx, samples.data(), static_cast<int>(samples.size()), params, out_result);
}

void q3asr_transcribe_result_clear(q3asr_transcribe_result * result) {
    if (result == nullptr) {
        return;
    }

    std::free(result->raw_text);
    std::free(result->language);
    std::free(result->text);

    result->raw_text = nullptr;
    result->language = nullptr;
    result->text = nullptr;
}
