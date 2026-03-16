#include "q3asr.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

void print_usage(const char * program) {
    std::cerr
        << "Usage: " << program << " --text-model <path> --mmproj-model <path> --audio <wav> [options]\n"
        << "Options:\n"
        << "  --language <name>   Force a language hint\n"
        << "  --max-tokens <n>    Maximum number of decoder tokens (default: 256)\n"
        << "  --threads <n>       Thread count for mel + decoder work\n"
        << "  --batch <n>         Decoder batch size (default: 512)\n"
        << "  --ctx <n>           Decoder context size (default: 4096)\n"
        << "  --no-gpu            Disable GPU offload\n"
        << "  --show-raw          Print the raw decoder string as well\n";
}

} // namespace

int main(int argc, char ** argv) {
    q3asr_context_params ctx_params = q3asr_context_default_params();
    q3asr_transcribe_params tx_params = q3asr_transcribe_default_params();

    std::string audio_path;
    bool show_raw = false;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];

        if (std::strcmp(arg, "--text-model") == 0 && i + 1 < argc) {
            ctx_params.text_model_path = argv[++i];
        } else if (std::strcmp(arg, "--mmproj-model") == 0 && i + 1 < argc) {
            ctx_params.mmproj_model_path = argv[++i];
        } else if (std::strcmp(arg, "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (std::strcmp(arg, "--language") == 0 && i + 1 < argc) {
            tx_params.language_hint = argv[++i];
        } else if (std::strcmp(arg, "--max-tokens") == 0 && i + 1 < argc) {
            tx_params.max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--threads") == 0 && i + 1 < argc) {
            ctx_params.n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--batch") == 0 && i + 1 < argc) {
            ctx_params.n_batch = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--ctx") == 0 && i + 1 < argc) {
            ctx_params.n_ctx = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--no-gpu") == 0) {
            ctx_params.use_gpu = 0;
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

    if (ctx_params.text_model_path == nullptr || ctx_params.mmproj_model_path == nullptr || audio_path.empty()) {
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

    q3asr_transcribe_result result = {};
    const int ok = q3asr_transcribe_wav_file(ctx, audio_path.c_str(), &tx_params, &result);
    if (!ok) {
        std::cerr << q3asr_context_last_error(ctx) << "\n";
        q3asr_context_destroy(ctx);
        return 1;
    }

    if (show_raw) {
        std::cout << "raw: " << (result.raw_text != nullptr ? result.raw_text : "") << "\n";
        std::cout << "language: " << (result.language != nullptr ? result.language : "") << "\n";
        std::cout << "text: ";
    }

    std::cout << ((result.text != nullptr && result.text[0] != '\0') ? result.text : result.raw_text) << "\n";

    q3asr_transcribe_result_clear(&result);
    q3asr_context_destroy(ctx);
    return 0;
}
