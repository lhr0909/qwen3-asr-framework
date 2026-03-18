#include "q3asr.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

struct StreamCapture {
    int call_count = 0;
    std::string last_raw;
};

void capture_stream_callback(const char * raw_text, void * user_data) {
    auto * capture = static_cast<StreamCapture *>(user_data);
    if (capture == nullptr) {
        return;
    }

    ++capture->call_count;
    capture->last_raw = raw_text != nullptr ? raw_text : "";
}

} // namespace

int main(int argc, char ** argv) {
    q3asr_context_params ctx_params = q3asr_context_default_params();
    q3asr_aligner_context_params aligner_params = q3asr_aligner_context_default_params();
    q3asr_transcribe_params tx_params = q3asr_transcribe_default_params();
    std::string audio_path;
    std::string expect_substring;
    std::string expect_language;
    int expect_stream_calls_at_least = 0;
    bool capture_stream = false;
    bool expect_stream_equals_raw = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--text-model") == 0 && i + 1 < argc) {
            ctx_params.text_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--mmproj-model") == 0 && i + 1 < argc) {
            ctx_params.mmproj_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--aligner-model") == 0 && i + 1 < argc) {
            aligner_params.aligner_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (std::strcmp(argv[i], "--audio-chunk-sec") == 0 && i + 1 < argc) {
            // Keep the CLI/test surface aligned with the library surface for chunked long-audio transcription.
            // The default overlap is applied in the same way as the CLI.
            // NOLINTNEXTLINE(cert-err34-c)
            // std::strtof is sufficient here because this is a small argv-only test harness.
            tx_params.max_audio_chunk_seconds = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            // NOLINTNEXTLINE(cert-err34-c)
            tx_params.temperature = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--expect-substring") == 0 && i + 1 < argc) {
            expect_substring = argv[++i];
        } else if (std::strcmp(argv[i], "--expect-language") == 0 && i + 1 < argc) {
            expect_language = argv[++i];
        } else if (std::strcmp(argv[i], "--capture-stream") == 0) {
            capture_stream = true;
        } else if (std::strcmp(argv[i], "--expect-stream-equals-raw") == 0) {
            expect_stream_equals_raw = true;
        } else if (std::strcmp(argv[i], "--expect-stream-calls-at-least") == 0 && i + 1 < argc) {
            expect_stream_calls_at_least = std::atoi(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            return 1;
        }
    }

    if (ctx_params.text_model_path == nullptr || ctx_params.mmproj_model_path == nullptr || audio_path.empty()) {
        std::cerr << "Smoke test requires --text-model, --mmproj-model, and --audio\n";
        return 1;
    }

    q3asr_context * ctx = q3asr_context_create(&ctx_params);
    if (ctx == nullptr) {
        std::cerr << "Failed to create q3asr context\n";
        return 1;
    }

    if (*q3asr_context_last_error(ctx) != '\0') {
        std::cerr << q3asr_context_last_error(ctx) << "\n";
        q3asr_context_destroy(ctx);
        return 1;
    }

    q3asr_transcribe_result result = {};
    StreamCapture stream_capture;
    q3asr_aligner_context * aligner_ctx = nullptr;
    if (aligner_params.aligner_model_path != nullptr) {
        aligner_ctx = q3asr_aligner_context_create(&aligner_params);
        if (aligner_ctx == nullptr) {
            std::cerr << "Failed to create q3asr aligner context\n";
            q3asr_context_destroy(ctx);
            return 1;
        }
        if (*q3asr_aligner_context_last_error(aligner_ctx) != '\0') {
            std::cerr << q3asr_aligner_context_last_error(aligner_ctx) << "\n";
            q3asr_aligner_context_destroy(aligner_ctx);
            q3asr_context_destroy(ctx);
            return 1;
        }

        tx_params.aligner_context = aligner_ctx;
        if (tx_params.max_audio_chunk_seconds <= 0.0f) {
            tx_params.max_audio_chunk_seconds = 180.0f;
        }
        if (tx_params.audio_chunk_overlap_seconds <= 0.0f) {
            tx_params.audio_chunk_overlap_seconds = 5.0f;
        }
    }

    if (capture_stream || expect_stream_calls_at_least > 0) {
        tx_params.raw_text_callback = capture_stream_callback;
        tx_params.raw_text_callback_user_data = &stream_capture;
    }

    if (!q3asr_transcribe_wav_file(ctx, audio_path.c_str(), &tx_params, &result)) {
        std::cerr << q3asr_context_last_error(ctx) << "\n";
        if (aligner_ctx != nullptr) {
            q3asr_aligner_context_destroy(aligner_ctx);
        }
        q3asr_context_destroy(ctx);
        return 1;
    }

    const std::string raw = result.raw_text != nullptr ? result.raw_text : "";
    const std::string language = result.language != nullptr ? result.language : "";
    const std::string text = result.text != nullptr ? result.text : "";

    std::cout << "raw=" << raw << "\n";
    std::cout << "language=" << language << "\n";
    std::cout << "text=" << text << "\n";
    if (tx_params.raw_text_callback != nullptr) {
        std::cout << "stream_calls=" << stream_capture.call_count << "\n";
        std::cout << "stream_raw=" << stream_capture.last_raw << "\n";
    }

    bool ok = !text.empty();
    if (ok && !expect_substring.empty()) {
        ok = raw.find(expect_substring) != std::string::npos || text.find(expect_substring) != std::string::npos;
    }
    if (ok && !expect_language.empty()) {
        ok = language == expect_language;
    }
    if (ok && expect_stream_calls_at_least > 0) {
        ok = stream_capture.call_count >= expect_stream_calls_at_least;
    }
    if (ok && expect_stream_equals_raw) {
        ok = stream_capture.last_raw == raw;
    }

    q3asr_transcribe_result_clear(&result);
    if (aligner_ctx != nullptr) {
        q3asr_aligner_context_destroy(aligner_ctx);
    }
    q3asr_context_destroy(ctx);
    return ok ? 0 : 1;
}
