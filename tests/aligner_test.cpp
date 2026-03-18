#include "q3asr.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

namespace {

bool nearly_monotonic(const q3asr_alignment_result & result) {
    float last_start = -1.0f;
    float last_end = -1.0f;

    for (size_t i = 0; i < result.n_items; ++i) {
        const float start = result.items[i].start_time;
        const float end = result.items[i].end_time;
        if (start > end) {
            return false;
        }
        if (i > 0 && start + 1.0e-4f < last_start) {
            return false;
        }
        if (i > 0 && end + 1.0e-4f < last_end) {
            return false;
        }
        last_start = start;
        last_end = end;
    }

    return true;
}

} // namespace

int main(int argc, char ** argv) {
    q3asr_aligner_context_params params = q3asr_aligner_context_default_params();
    q3asr_align_params align_params = q3asr_align_default_params();
    std::string audio_path;
    std::string text;
    std::string language;
    std::string expect_item;
    std::string expect_first;
    std::string expect_last;
    int expect_count = -1;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--aligner-model") == 0 && i + 1 < argc) {
            params.aligner_model_path = argv[++i];
        } else if (std::strcmp(argv[i], "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (std::strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
            text = argv[++i];
        } else if (std::strcmp(argv[i], "--language") == 0 && i + 1 < argc) {
            language = argv[++i];
        } else if (std::strcmp(argv[i], "--max-chunk-sec") == 0 && i + 1 < argc) {
            align_params.max_chunk_seconds = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(argv[i], "--expect-count") == 0 && i + 1 < argc) {
            expect_count = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--expect-item") == 0 && i + 1 < argc) {
            expect_item = argv[++i];
        } else if (std::strcmp(argv[i], "--expect-first") == 0 && i + 1 < argc) {
            expect_first = argv[++i];
        } else if (std::strcmp(argv[i], "--expect-last") == 0 && i + 1 < argc) {
            expect_last = argv[++i];
        } else {
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            return 1;
        }
    }

    if (params.aligner_model_path == nullptr || audio_path.empty() || text.empty() || language.empty()) {
        std::cerr << "Aligner test requires --aligner-model, --audio, --text, and --language\n";
        return 1;
    }

    q3asr_aligner_context * ctx = q3asr_aligner_context_create(&params);
    if (ctx == nullptr) {
        std::cerr << "Failed to create q3asr aligner context\n";
        return 1;
    }

    if (*q3asr_aligner_context_last_error(ctx) != '\0') {
        std::cerr << q3asr_aligner_context_last_error(ctx) << "\n";
        q3asr_aligner_context_destroy(ctx);
        return 1;
    }

    q3asr_alignment_result result = {};
    if (!q3asr_align_wav_file_ex(ctx, audio_path.c_str(), text.c_str(), language.c_str(), &align_params, &result)) {
        std::cerr << q3asr_aligner_context_last_error(ctx) << "\n";
        q3asr_aligner_context_destroy(ctx);
        return 1;
    }

    bool ok = result.n_items > 0 && nearly_monotonic(result);
    if (ok && expect_count >= 0) {
        ok = static_cast<int>(result.n_items) == expect_count;
    }
    if (ok && !expect_first.empty()) {
        ok = result.items[0].text != nullptr && expect_first == result.items[0].text;
    }
    if (ok && !expect_last.empty()) {
        ok = result.items[result.n_items - 1].text != nullptr && expect_last == result.items[result.n_items - 1].text;
    }
    if (ok && !expect_item.empty()) {
        ok = false;
        for (size_t i = 0; i < result.n_items; ++i) {
            if (result.items[i].text != nullptr && expect_item == result.items[i].text) {
                ok = true;
                break;
            }
        }
    }

    std::cout << "count=" << result.n_items << "\n";
    for (size_t i = 0; i < result.n_items; ++i) {
        std::cout
            << i
            << ":"
            << (result.items[i].text != nullptr ? result.items[i].text : "")
            << "@"
            << result.items[i].start_time
            << ","
            << result.items[i].end_time
            << "\n";
    }

    q3asr_alignment_result_clear(&result);
    q3asr_aligner_context_destroy(ctx);
    return ok ? 0 : 1;
}
