#include "q3asr.h"

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

struct RawStreamState {
    std::string last;
};

void raw_stream_callback(const char * raw_text, void * user_data) {
    auto * state = static_cast<RawStreamState *>(user_data);
    if (state == nullptr) {
        return;
    }

    const std::string current = raw_text != nullptr ? raw_text : "";
    size_t prefix = 0;
    while (
        prefix < state->last.size() &&
        prefix < current.size() &&
        state->last[prefix] == current[prefix]
    ) {
        ++prefix;
    }

    std::cout << current.substr(prefix);
    std::cout.flush();
    state->last = current;
}

void print_usage(const char * program) {
    std::cerr
        << "Usage:\n"
        << "  " << program << " --text-model <path> --mmproj-model <path> --audio <wav> [options]\n"
        << "  " << program << " --aligner-model <path> --align-text <text> --audio <wav> [options]\n"
        << "Options:\n"
        << "  --language <name>   Force a language hint\n"
        << "  --max-tokens <n>    Maximum number of decoder tokens (default: 256)\n"
        << "  --threads <n>       Thread count for mel + decoder work\n"
        << "  --batch <n>         Decoder batch size (default: 512)\n"
        << "  --ctx <n>           Decoder context size (default: 4096)\n"
        << "  --korean-dict <p>   Optional dictionary path for Korean aligner tokenization\n"
        << "  --no-gpu            Disable GPU offload\n"
        << "  --stream-raw        Stream raw decoder text as it is generated\n"
        << "  --show-raw          Print the raw decoder string as well\n";
}

} // namespace

int main(int argc, char ** argv) {
    q3asr_context_params ctx_params = q3asr_context_default_params();
    q3asr_aligner_context_params aligner_params = q3asr_aligner_context_default_params();
    q3asr_transcribe_params tx_params = q3asr_transcribe_default_params();

    std::string audio_path;
    std::string align_text;
    bool show_raw = false;
    bool stream_raw = false;
    RawStreamState raw_stream_state;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];

        if (std::strcmp(arg, "--text-model") == 0 && i + 1 < argc) {
            ctx_params.text_model_path = argv[++i];
        } else if (std::strcmp(arg, "--aligner-model") == 0 && i + 1 < argc) {
            aligner_params.aligner_model_path = argv[++i];
        } else if (std::strcmp(arg, "--mmproj-model") == 0 && i + 1 < argc) {
            ctx_params.mmproj_model_path = argv[++i];
        } else if (std::strcmp(arg, "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (std::strcmp(arg, "--align-text") == 0 && i + 1 < argc) {
            align_text = argv[++i];
        } else if (std::strcmp(arg, "--language") == 0 && i + 1 < argc) {
            tx_params.language_hint = argv[++i];
        } else if (std::strcmp(arg, "--max-tokens") == 0 && i + 1 < argc) {
            tx_params.max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--threads") == 0 && i + 1 < argc) {
            ctx_params.n_threads = std::atoi(argv[++i]);
            aligner_params.n_threads = ctx_params.n_threads;
        } else if (std::strcmp(arg, "--batch") == 0 && i + 1 < argc) {
            ctx_params.n_batch = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--ctx") == 0 && i + 1 < argc) {
            ctx_params.n_ctx = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--korean-dict") == 0 && i + 1 < argc) {
            aligner_params.korean_dict_path = argv[++i];
        } else if (std::strcmp(arg, "--no-gpu") == 0) {
            ctx_params.use_gpu = 0;
            aligner_params.use_gpu = 0;
        } else if (std::strcmp(arg, "--stream-raw") == 0) {
            stream_raw = true;
        } else if (std::strcmp(arg, "--show-raw") == 0) {
            show_raw = true;
        } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    const bool transcribe_mode =
        ctx_params.text_model_path != nullptr ||
        ctx_params.mmproj_model_path != nullptr;
    const bool align_mode = aligner_params.aligner_model_path != nullptr || !align_text.empty();

    if (audio_path.empty() || (transcribe_mode == align_mode)) {
        print_usage(argv[0]);
        return 1;
    }

    if (transcribe_mode) {
        if (ctx_params.text_model_path == nullptr || ctx_params.mmproj_model_path == nullptr) {
            print_usage(argv[0]);
            return 1;
        }

        q3asr_context * ctx = q3asr_context_create(&ctx_params);
        if (ctx == nullptr) {
            std::cerr << "Failed to create the q3asr context\n";
            return 1;
        }

        if (*q3asr_context_last_error(ctx) != '\0') {
            std::cerr << q3asr_context_last_error(ctx) << "\n";
            q3asr_context_destroy(ctx);
            return 1;
        }

        if (stream_raw) {
            tx_params.raw_text_callback = raw_stream_callback;
            tx_params.raw_text_callback_user_data = &raw_stream_state;
        }

        q3asr_transcribe_result result = {};
        const int ok = q3asr_transcribe_wav_file(ctx, audio_path.c_str(), &tx_params, &result);
        if (!ok) {
            std::cerr << q3asr_context_last_error(ctx) << "\n";
            q3asr_context_destroy(ctx);
            return 1;
        }

        if (stream_raw) {
            std::cout << "\n";
        }

        if (show_raw && !stream_raw) {
            std::cout << "raw: " << (result.raw_text != nullptr ? result.raw_text : "") << "\n";
            std::cout << "language: " << (result.language != nullptr ? result.language : "") << "\n";
            std::cout << "text: ";
        } else if (show_raw && stream_raw) {
            std::cout << "language: " << (result.language != nullptr ? result.language : "") << "\n";
            std::cout << "text: ";
        }

        if (!stream_raw || show_raw) {
            std::cout << ((result.text != nullptr && result.text[0] != '\0') ? result.text : result.raw_text) << "\n";
        }

        q3asr_transcribe_result_clear(&result);
        q3asr_context_destroy(ctx);
        return 0;
    }

    if (aligner_params.aligner_model_path == nullptr || align_text.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    q3asr_aligner_context * ctx = q3asr_aligner_context_create(&aligner_params);
    if (ctx == nullptr) {
        std::cerr << "Failed to create the q3asr aligner context\n";
        return 1;
    }

    if (*q3asr_aligner_context_last_error(ctx) != '\0') {
        std::cerr << q3asr_aligner_context_last_error(ctx) << "\n";
        q3asr_aligner_context_destroy(ctx);
        return 1;
    }

    q3asr_alignment_result result = {};
    const int ok = q3asr_align_wav_file(ctx, audio_path.c_str(), align_text.c_str(), tx_params.language_hint, &result);
    if (!ok) {
        std::cerr << q3asr_aligner_context_last_error(ctx) << "\n";
        q3asr_aligner_context_destroy(ctx);
        return 1;
    }

    std::cout << std::fixed << std::setprecision(3);
    for (size_t i = 0; i < result.n_items; ++i) {
        std::cout
            << result.items[i].start_time
            << '\t'
            << result.items[i].end_time
            << '\t'
            << (result.items[i].text != nullptr ? result.items[i].text : "")
            << "\n";
    }

    q3asr_alignment_result_clear(&result);
    q3asr_aligner_context_destroy(ctx);
    return 0;
}
