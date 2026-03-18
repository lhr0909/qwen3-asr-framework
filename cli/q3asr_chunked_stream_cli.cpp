#include "q3asr.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

bool read_text_file(const std::string & path, std::string & out, std::string & error) {
    std::ifstream file(path);
    if (!file.is_open()) {
        error = "Failed to open text file: " + path;
        return false;
    }

    out.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    return true;
}

struct ConsoleState {
    bool use_ansi = false;
    bool cursor_hidden = false;
    bool rendered_once = false;
    std::chrono::steady_clock::time_point last_render = std::chrono::steady_clock::time_point::min();
    std::string language;
    std::string committed_text;
    std::string partial_text;
    int chunk_index = 0;
    int chunk_count = 0;
};

std::string tail_view(const std::string & text, size_t max_chars) {
    if (text.empty()) {
        return "(empty)";
    }
    if (text.size() <= max_chars) {
        return text;
    }
    return "..." + text.substr(text.size() - max_chars);
}

void render_console(ConsoleState & state, bool force) {
    const auto now = std::chrono::steady_clock::now();
    if (!force && state.rendered_once && now - state.last_render < std::chrono::milliseconds(50)) {
        return;
    }

    state.last_render = now;

    if (state.use_ansi) {
        if (!state.cursor_hidden) {
            std::cerr << "\x1b[?25l";
            state.cursor_hidden = true;
        }
        std::cerr << "\x1b[2J\x1b[H";
    }

    std::cerr << "q3asr chunked streaming\n";
    if (state.chunk_count > 0) {
        std::cerr << "chunk: " << state.chunk_index << "/" << state.chunk_count << "\n";
    }
    if (!state.language.empty()) {
        std::cerr << "language: " << state.language << "\n";
    }
    std::cerr << "\ncommitted:\n" << tail_view(state.committed_text, 4000) << "\n";
    std::cerr << "\npartial:\n" << tail_view(state.partial_text, 1600) << "\n";
    std::cerr.flush();

    state.rendered_once = true;
}

void progress_callback(
    const char * language,
    const char * committed_text,
    const char * partial_text,
    int chunk_index,
    int chunk_count,
    void * user_data
) {
    auto * state = static_cast<ConsoleState *>(user_data);
    if (state == nullptr) {
        return;
    }

    state->language = language != nullptr ? language : "";
    state->committed_text = committed_text != nullptr ? committed_text : "";
    state->partial_text = partial_text != nullptr ? partial_text : "";
    state->chunk_index = chunk_index;
    state->chunk_count = chunk_count;
    render_console(*state, false);
}

void restore_console(ConsoleState & state) {
    if (state.use_ansi && state.cursor_hidden) {
        std::cerr << "\x1b[?25h";
        state.cursor_hidden = false;
    }
}

void print_usage(const char * program) {
    std::cerr
        << "Usage:\n"
        << "  " << program << " --text-model <path> --mmproj-model <path> --aligner-model <path> --audio <wav> [options]\n"
        << "Options:\n"
        << "  --language <name>       Force a language hint\n"
        << "  --context <text>        Prompt context / hotword hint text\n"
        << "  --context-file <p>      Read prompt context from a text file\n"
        << "  --audio-chunk-sec <sec> Max audio seconds per transcription chunk (default: 180)\n"
        << "  --audio-overlap-sec <sec> Overlap seconds between chunk decode windows (default: 5)\n"
        << "  --max-tokens <n>        Maximum number of decoder tokens (default: 256)\n"
        << "  --temp <value>          Decoder temperature (default: 0, greedy)\n"
        << "  --threads <n>           Thread count for mel + decoder work\n"
        << "  --batch <n>             Decoder batch size (default: 512)\n"
        << "  --ctx <n>               Decoder context size (default: 4096)\n"
        << "  --no-gpu                Disable GPU offload\n"
        << "  --show-raw              Print the final raw decoder string as well\n";
}

} // namespace

int main(int argc, char ** argv) {
    q3asr_context_params ctx_params = q3asr_context_default_params();
    q3asr_aligner_context_params aligner_params = q3asr_aligner_context_default_params();
    q3asr_transcribe_params tx_params = q3asr_transcribe_default_params();
    std::string audio_path;
    std::string context;
    std::string context_file;
    bool has_context_arg = false;
    bool has_context_file_arg = false;
    bool show_raw = false;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];
        if (std::strcmp(arg, "--text-model") == 0 && i + 1 < argc) {
            ctx_params.text_model_path = argv[++i];
        } else if (std::strcmp(arg, "--mmproj-model") == 0 && i + 1 < argc) {
            ctx_params.mmproj_model_path = argv[++i];
        } else if (std::strcmp(arg, "--aligner-model") == 0 && i + 1 < argc) {
            aligner_params.aligner_model_path = argv[++i];
        } else if (std::strcmp(arg, "--audio") == 0 && i + 1 < argc) {
            audio_path = argv[++i];
        } else if (std::strcmp(arg, "--language") == 0 && i + 1 < argc) {
            tx_params.language_hint = argv[++i];
        } else if (std::strcmp(arg, "--context") == 0 && i + 1 < argc) {
            context = argv[++i];
            has_context_arg = true;
        } else if (std::strcmp(arg, "--context-file") == 0 && i + 1 < argc) {
            context_file = argv[++i];
            has_context_file_arg = true;
        } else if (std::strcmp(arg, "--audio-chunk-sec") == 0 && i + 1 < argc) {
            tx_params.max_audio_chunk_seconds = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(arg, "--audio-overlap-sec") == 0 && i + 1 < argc) {
            tx_params.audio_chunk_overlap_seconds = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(arg, "--max-tokens") == 0 && i + 1 < argc) {
            tx_params.max_tokens = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--temp") == 0 && i + 1 < argc) {
            tx_params.temperature = std::strtof(argv[++i], nullptr);
        } else if (std::strcmp(arg, "--threads") == 0 && i + 1 < argc) {
            ctx_params.n_threads = std::atoi(argv[++i]);
            aligner_params.n_threads = ctx_params.n_threads;
        } else if (std::strcmp(arg, "--batch") == 0 && i + 1 < argc) {
            ctx_params.n_batch = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--ctx") == 0 && i + 1 < argc) {
            ctx_params.n_ctx = std::atoi(argv[++i]);
        } else if (std::strcmp(arg, "--no-gpu") == 0) {
            ctx_params.use_gpu = 0;
            aligner_params.use_gpu = 0;
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

    if (has_context_arg && has_context_file_arg) {
        std::cerr << "Use only one of --context or --context-file\n";
        return 1;
    }

    std::string text_file_error;
    if (has_context_file_arg && !read_text_file(context_file, context, text_file_error)) {
        std::cerr << text_file_error << "\n";
        return 1;
    }

    if (
        ctx_params.text_model_path == nullptr ||
        ctx_params.mmproj_model_path == nullptr ||
        aligner_params.aligner_model_path == nullptr ||
        audio_path.empty()
    ) {
        print_usage(argv[0]);
        return 1;
    }

    if (tx_params.max_audio_chunk_seconds <= 0.0f) {
        tx_params.max_audio_chunk_seconds = 180.0f;
    }
    if (tx_params.audio_chunk_overlap_seconds <= 0.0f) {
        tx_params.audio_chunk_overlap_seconds = 5.0f;
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

    q3asr_aligner_context * aligner_ctx = q3asr_aligner_context_create(&aligner_params);
    if (aligner_ctx == nullptr) {
        std::cerr << "Failed to create the q3asr aligner context\n";
        q3asr_context_destroy(ctx);
        return 1;
    }
    if (*q3asr_aligner_context_last_error(aligner_ctx) != '\0') {
        std::cerr << q3asr_aligner_context_last_error(aligner_ctx) << "\n";
        q3asr_aligner_context_destroy(aligner_ctx);
        q3asr_context_destroy(ctx);
        return 1;
    }

    ConsoleState console_state;
#if !defined(_WIN32)
    console_state.use_ansi = ::isatty(fileno(stderr)) != 0;
#endif

    tx_params.aligner_context = aligner_ctx;
    tx_params.context = context.c_str();
    tx_params.progress_callback = progress_callback;
    tx_params.progress_callback_user_data = &console_state;

    q3asr_transcribe_result result = {};
    const int ok = q3asr_transcribe_wav_file(ctx, audio_path.c_str(), &tx_params, &result);

    if (ok) {
        console_state.language = result.language != nullptr ? result.language : "";
        console_state.committed_text = result.text != nullptr ? result.text : "";
        console_state.partial_text.clear();
        if (console_state.chunk_count == 0) {
            console_state.chunk_index = 1;
            console_state.chunk_count = 1;
        }
        render_console(console_state, true);
        std::cerr << "\n";
    } else {
        restore_console(console_state);
        std::cerr << q3asr_context_last_error(ctx) << "\n";
        q3asr_alignment_result dummy = {};
        (void) dummy;
    }

    restore_console(console_state);

    if (!ok) {
        q3asr_aligner_context_destroy(aligner_ctx);
        q3asr_context_destroy(ctx);
        return 1;
    }

    if (show_raw && result.raw_text != nullptr) {
        std::cout << "raw: " << result.raw_text << "\n";
    }
    if (result.language != nullptr && result.language[0] != '\0') {
        std::cout << "language: " << result.language << "\n";
    }
    std::cout << "text: " << (result.text != nullptr ? result.text : "") << "\n";

    q3asr_transcribe_result_clear(&result);
    q3asr_aligner_context_destroy(aligner_ctx);
    q3asr_context_destroy(ctx);
    return 0;
}
