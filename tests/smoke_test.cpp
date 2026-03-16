#include "q3asr.h"

#include <cstring>
#include <iostream>
#include <string>

int main(int argc, char ** argv) {
    q3asr_context_params ctx_params = q3asr_context_default_params();
    std::string audio_path;
    std::string expect_substring;
    std::string expect_language;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--text-model") == 0 && i + 1 < argc) {
            ctx_params.text_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--mmproj-model") == 0 && i + 1 < argc) {
            ctx_params.mmproj_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (std::strcmp(argv[i], "--expect-substring") == 0 && i + 1 < argc) {
            expect_substring = argv[++i];
        } else if (std::strcmp(argv[i], "--expect-language") == 0 && i + 1 < argc) {
            expect_language = argv[++i];
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
    q3asr_transcribe_params tx_params = q3asr_transcribe_default_params();

    if (!q3asr_transcribe_wav_file(ctx, audio_path.c_str(), &tx_params, &result)) {
        std::cerr << q3asr_context_last_error(ctx) << "\n";
        q3asr_context_destroy(ctx);
        return 1;
    }

    const std::string raw = result.raw_text != nullptr ? result.raw_text : "";
    const std::string language = result.language != nullptr ? result.language : "";
    const std::string text = result.text != nullptr ? result.text : "";

    std::cout << "raw=" << raw << "\n";
    std::cout << "language=" << language << "\n";
    std::cout << "text=" << text << "\n";

    bool ok = !text.empty();
    if (ok && !expect_substring.empty()) {
        ok = raw.find(expect_substring) != std::string::npos || text.find(expect_substring) != std::string::npos;
    }
    if (ok && !expect_language.empty()) {
        ok = language == expect_language;
    }

    q3asr_transcribe_result_clear(&result);
    q3asr_context_destroy(ctx);
    return ok ? 0 : 1;
}
