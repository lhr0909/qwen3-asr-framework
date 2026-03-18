#include "q3asr.h"

#include "audio_encoder.h"
#include "decoder_llama.h"
#include "forced_aligner.h"
#include "mel_spectrogram.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct q3asr_aligner_context {
    std::unique_ptr<q3asr::ForcedAligner> aligner;
    std::string last_error;
};

namespace q3asr {

namespace {

} // namespace

struct transcript_result {
    std::string raw_text;
    std::string language;
    std::string text;
    std::vector<decoder_token_span> text_token_spans;
};

struct parsed_asr_output {
    std::string language;
    std::string text;
    size_t text_byte_start = 0;
};

struct scored_fragment {
    std::string text;
    float avg_logprob = 0.0f;
    bool has_confidence = false;
};

struct transcript_chunk_view {
    transcript_result transcript;
    alignment_result alignment;
    std::vector<normalized_word_span> spans;
    std::string language;
    float window_start_sec = 0.0f;
    float core_start_sec = 0.0f;
    float core_end_sec = 0.0f;
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

        const std::string language_hint = normalize_language(params.language_hint == nullptr ? "" : params.language_hint);
        const std::string context = params.context != nullptr ? params.context : "";
        std::function<void(const std::string &)> raw_text_callback;
        if (params.raw_text_callback != nullptr) {
            raw_text_callback = [&](const std::string & raw_text) {
                params.raw_text_callback(raw_text.c_str(), params.raw_text_callback_user_data);
            };
        }

        progress_callback_fn progress_callback;
        if (params.progress_callback != nullptr) {
            progress_callback = [&](const std::string & language,
                                    const std::string & committed_text,
                                    const std::string & partial_text,
                                    int chunk_index,
                                    int chunk_count) {
                params.progress_callback(
                    language.c_str(),
                    committed_text.c_str(),
                    partial_text.c_str(),
                    chunk_index,
                    chunk_count,
                    params.progress_callback_user_data
                );
            };
        }

        if (
            params.aligner_context != nullptr &&
            params.aligner_context->aligner != nullptr &&
            params.max_audio_chunk_seconds > 0.0f &&
            static_cast<float>(n_samples) / QWEN_SAMPLE_RATE > params.max_audio_chunk_seconds
        ) {
            return transcribe_long(samples, n_samples, params, language_hint, context, raw_text_callback, progress_callback, result);
        }

        std::function<void(const std::string &)> chunk_raw_callback = raw_text_callback;
        if (progress_callback) {
            chunk_raw_callback = [&, raw_text_callback](const std::string & raw_text) {
                if (raw_text_callback) {
                    raw_text_callback(raw_text);
                }
                progress_callback(language_hint, {}, extract_partial_text(raw_text, language_hint), 1, 1);
            };
        }

        const bool ok = transcribe_chunk(samples, n_samples, params, language_hint, context, chunk_raw_callback, result);
        if (ok && progress_callback) {
            progress_callback(
                !result.language.empty() ? result.language : language_hint,
                result.text,
                {},
                1,
                1
            );
        }
        return ok;
    }

    const std::string & error() const { return error_msg_; }

private:
    using progress_callback_fn = std::function<void(
        const std::string & language,
        const std::string & committed_text,
        const std::string & partial_text,
        int chunk_index,
        int chunk_count
    )>;

    bool transcribe_chunk(
        const float * samples,
        int n_samples,
        const q3asr_transcribe_params & params,
        const std::string & language_hint,
        const std::string & context,
        const std::function<void(const std::string &)> & raw_text_callback,
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
        decoder_params.temperature = params.temperature;
        decoder_params.n_threads = params_.n_threads;
        decoder_params.n_batch = params_.n_batch;
        decoder_params.n_ctx = params_.n_ctx;
        decoder_params.language_hint = language_hint;
        decoder_params.context = context;
        decoder_params.raw_text_callback = raw_text_callback;

        std::vector<decoder_token_span> raw_token_spans;
        if (!decoder_.decode_audio_embeddings(audio_embeddings, decoder_params, result.raw_text, &raw_token_spans)) {
            error_msg_ = decoder_.get_error();
            return false;
        }

        const auto parsed = parse_asr_output(result.raw_text, decoder_params.language_hint);
        result.language = parsed.language;
        result.text = parsed.text;
        result.text_token_spans.clear();
        result.text_token_spans.reserve(raw_token_spans.size());
        for (const decoder_token_span & span : raw_token_spans) {
            if (span.byte_end <= parsed.text_byte_start) {
                continue;
            }

            const size_t clipped_start = std::max(span.byte_start, parsed.text_byte_start) - parsed.text_byte_start;
            const size_t clipped_end = span.byte_end - parsed.text_byte_start;
            if (clipped_end <= clipped_start) {
                continue;
            }

            result.text_token_spans.push_back({clipped_start, clipped_end, span.logprob});
        }
        return true;
    }

    bool transcribe_long(
        const float * samples,
        int n_samples,
        const q3asr_transcribe_params & params,
        const std::string & requested_language,
        const std::string & context,
        const std::function<void(const std::string &)> & raw_text_callback,
        const progress_callback_fn & progress_callback,
        transcript_result & result
    ) {
        result = {};

        if (params.aligner_context == nullptr || params.aligner_context->aligner == nullptr) {
            error_msg_ = "Chunked long-audio transcription requires a loaded forced aligner context";
            return false;
        }

        align_runtime_params split_params;
        split_params.max_chunk_seconds = params.max_audio_chunk_seconds;
        const std::vector<split_audio_chunk> core_chunks =
            split_audio_into_chunks(samples, n_samples, QWEN_SAMPLE_RATE, split_params);
        if (core_chunks.empty()) {
            error_msg_ = "Failed to split long audio for chunked transcription";
            return false;
        }

        q3asr_transcribe_params chunk_params = params;
        chunk_params.aligner_context = nullptr;
        chunk_params.max_audio_chunk_seconds = 0.0f;
        chunk_params.audio_chunk_overlap_seconds = 0.0f;
        chunk_params.raw_text_callback = nullptr;
        chunk_params.raw_text_callback_user_data = nullptr;
        chunk_params.progress_callback = nullptr;
        chunk_params.progress_callback_user_data = nullptr;

        align_runtime_params align_params;
        align_params.max_chunk_seconds = std::max(
            params.max_audio_chunk_seconds,
            params.max_audio_chunk_seconds + 2.0f * std::max(0.0f, params.audio_chunk_overlap_seconds)
        );

        const float overlap_sec = std::max(0.0f, params.audio_chunk_overlap_seconds);
        const int overlap_samples = static_cast<int>(std::lround(overlap_sec * QWEN_SAMPLE_RATE));
        const float boundary_band_sec = overlap_sec > 0.0f
            ? std::min(overlap_sec, std::max(0.75f, overlap_sec * 0.5f))
            : 0.0f;
        const float total_audio_sec = static_cast<float>(n_samples) / QWEN_SAMPLE_RATE;

        std::string merged_text;
        std::string resolved_language = requested_language;
        std::string last_emitted_raw;
        transcript_chunk_view previous_chunk;
        bool have_previous_chunk = false;

        const auto emit_progress = [&]() {
            if (!raw_text_callback) {
                return;
            }

            const std::string raw = build_raw_output(merged_text, resolved_language, requested_language);
            if (raw != last_emitted_raw) {
                raw_text_callback(raw);
                last_emitted_raw = raw;
            }
        };

        for (size_t i = 0; i < core_chunks.size(); ++i) {
            const int chunk_index = static_cast<int>(i + 1);
            const int chunk_count = static_cast<int>(core_chunks.size());
            const int core_start = static_cast<int>(std::lround(core_chunks[i].offset_sec * QWEN_SAMPLE_RATE));
            const int core_end = std::min(n_samples, core_start + core_chunks[i].original_n_samples);
            const int window_start = std::max(0, core_start - overlap_samples);
            const int window_end = std::min(n_samples, core_end + overlap_samples);
            const int window_len = std::max(0, window_end - window_start);
            if (window_len <= 0) {
                continue;
            }

            transcript_result chunk_result;
            std::function<void(const std::string &)> chunk_raw_callback;
            if (progress_callback) {
                chunk_raw_callback = [&](const std::string & raw_text) {
                    progress_callback(
                        !resolved_language.empty() ? resolved_language : requested_language,
                        merged_text,
                        provisional_partial_text(
                            merged_text,
                            raw_text,
                            !resolved_language.empty() ? resolved_language : requested_language
                        ),
                        chunk_index,
                        chunk_count
                    );
                };
            }
            if (!transcribe_chunk(
                    samples + window_start,
                    window_len,
                    chunk_params,
                    resolved_language,
                    context,
                    chunk_raw_callback,
                    chunk_result)) {
                error_msg_ = "Chunked transcription failed for chunk " +
                             std::to_string(i + 1) + "/" + std::to_string(core_chunks.size()) +
                             ": " + error_msg_;
                return false;
            }

            const std::string chunk_language =
                !chunk_result.language.empty() ? chunk_result.language : resolved_language;
            if (resolved_language.empty() && !chunk_language.empty()) {
                resolved_language = chunk_language;
            }

            const alignment_result align_result = params.aligner_context->aligner->align(
                samples + window_start,
                window_len,
                chunk_result.text,
                chunk_language,
                align_params
            );
            if (!align_result.success) {
                error_msg_ = "Forced alignment failed for transcription chunk " +
                             std::to_string(i + 1) + "/" + std::to_string(core_chunks.size()) +
                             ": " + align_result.error_msg;
                return false;
            }

            transcript_chunk_view current_chunk;
            current_chunk.transcript = std::move(chunk_result);
            current_chunk.alignment = std::move(align_result);
            current_chunk.spans = params.aligner_context->aligner->normalize_with_spans(
                current_chunk.transcript.text,
                chunk_language
            );
            current_chunk.language = chunk_language;
            current_chunk.window_start_sec = static_cast<float>(window_start) / QWEN_SAMPLE_RATE;
            current_chunk.core_start_sec = static_cast<float>(core_start) / QWEN_SAMPLE_RATE;
            current_chunk.core_end_sec = static_cast<float>(core_end) / QWEN_SAMPLE_RATE;

            if (!have_previous_chunk) {
                const bool is_last_chunk = i + 1 == core_chunks.size();
                const float stable_end = is_last_chunk
                    ? current_chunk.core_end_sec
                    : std::max(current_chunk.core_start_sec, current_chunk.core_end_sec - boundary_band_sec);
                const scored_fragment leading_fragment = extract_scored_fragment(
                    current_chunk,
                    current_chunk.core_start_sec,
                    stable_end,
                    is_last_chunk
                );
                append_text_fragment(merged_text, leading_fragment.text);
                emit_progress();
                if (progress_callback) {
                    progress_callback(resolved_language, merged_text, {}, chunk_index, chunk_count);
                }

                if (is_last_chunk) {
                    break;
                }

                previous_chunk = std::move(current_chunk);
                have_previous_chunk = true;
                continue;
            }

            const float boundary_time = previous_chunk.core_end_sec;
            const float band_start = std::max(0.0f, boundary_time - boundary_band_sec);
            const float band_end = std::min(total_audio_sec, boundary_time + boundary_band_sec);
            if (band_end > band_start) {
                const scored_fragment left_boundary = extract_scored_fragment(
                    previous_chunk,
                    band_start,
                    band_end,
                    false
                );
                const scored_fragment right_boundary = extract_scored_fragment(
                    current_chunk,
                    band_start,
                    band_end,
                    false
                );
                append_text_fragment(merged_text, choose_boundary_fragment(left_boundary, right_boundary).text);
            }

            const bool is_last_chunk = i + 1 == core_chunks.size();
            const float stable_start = std::min(current_chunk.core_end_sec, current_chunk.core_start_sec + boundary_band_sec);
            const float stable_end = is_last_chunk
                ? current_chunk.core_end_sec
                : std::max(current_chunk.core_start_sec, current_chunk.core_end_sec - boundary_band_sec);
            const scored_fragment stable_fragment = extract_scored_fragment(
                current_chunk,
                stable_start,
                stable_end,
                is_last_chunk
            );
            append_text_fragment(merged_text, stable_fragment.text);
            emit_progress();
            if (progress_callback) {
                progress_callback(resolved_language, merged_text, {}, chunk_index, chunk_count);
            }

            previous_chunk = std::move(current_chunk);
        }

        result.language = resolved_language;
        result.text = merged_text;
        result.raw_text = build_raw_output(merged_text, resolved_language, requested_language);
        return true;
    }

    static std::string trim_ascii(const std::string & value) {
        size_t start = 0;
        while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
            ++start;
        }

        size_t end = value.size();
        while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
            --end;
        }

        return value.substr(start, end - start);
    }

    static bool is_ascii_punct_no_space_before(char ch) {
        switch (ch) {
            case ',':
            case '.':
            case ';':
            case ':':
            case '!':
            case '?':
            case ')':
            case ']':
            case '}':
            case '%':
            case '"':
            case '\'':
                return true;
            default:
                return false;
        }
    }

    static bool is_ascii_punct_no_space_after(char ch) {
        switch (ch) {
            case '(':
            case '[':
            case '{':
            case '"':
            case '\'':
                return true;
            default:
                return false;
        }
    }

    static void append_text_fragment(std::string & merged_text, const std::string & fragment) {
        const std::string trimmed = trim_ascii(fragment);
        if (trimmed.empty()) {
            return;
        }

        if (merged_text.empty()) {
            merged_text = trimmed;
            return;
        }

        const char last = merged_text.back();
        const char first = trimmed.front();
        const bool last_space = std::isspace(static_cast<unsigned char>(last)) != 0;
        const bool first_space = std::isspace(static_cast<unsigned char>(first)) != 0;

        if (!last_space && !first_space && !is_ascii_punct_no_space_before(first) && !is_ascii_punct_no_space_after(last)) {
            merged_text.push_back(' ');
        }

        merged_text += trimmed;
    }

    static std::string build_raw_output(
        const std::string & merged_text,
        const std::string & resolved_language,
        const std::string & requested_language
    ) {
        if (!requested_language.empty() || resolved_language.empty()) {
            return merged_text;
        }

        return "language " + resolved_language + "<asr_text>" + merged_text;
    }

    static std::string extract_partial_text(const std::string & raw_text, const std::string & forced_language) {
        if (!forced_language.empty()) {
            return raw_text;
        }

        const std::string tag = "<asr_text>";
        const size_t tag_pos = raw_text.find(tag);
        if (tag_pos != std::string::npos) {
            return raw_text.substr(tag_pos + tag.size());
        }

        std::string lowered;
        lowered.reserve(raw_text.size());
        for (unsigned char ch : raw_text) {
            lowered.push_back(static_cast<char>(std::tolower(ch)));
        }

        if (lowered.rfind("language", 0) == 0) {
            return {};
        }

        return raw_text;
    }

    static std::string strip_committed_overlap(const std::string & committed_text, const std::string & partial_text) {
        if (committed_text.empty() || partial_text.empty()) {
            return partial_text;
        }

        const size_t max_overlap = std::min<size_t>({committed_text.size(), partial_text.size(), 512});
        size_t best = 0;
        for (size_t n = max_overlap; n > 0; --n) {
            if (committed_text.compare(committed_text.size() - n, n, partial_text, 0, n) == 0) {
                best = n;
                break;
            }
        }

        if (best >= 12) {
            return partial_text.substr(best);
        }
        return partial_text;
    }

    static std::string provisional_partial_text(
        const std::string & committed_text,
        const std::string & raw_text,
        const std::string & forced_language
    ) {
        return strip_committed_overlap(committed_text, extract_partial_text(raw_text, forced_language));
    }

    static std::pair<size_t, size_t> trim_ascii_bounds(
        const std::string & value,
        size_t start,
        size_t end
    ) {
        start = std::min(start, value.size());
        end = std::min(end, value.size());
        while (start < end && std::isspace(static_cast<unsigned char>(value[start])) != 0) {
            ++start;
        }
        while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
            --end;
        }
        return {start, end};
    }

    static std::string join_normalized_units(const std::vector<std::string> & units, const std::string & language) {
        const std::string normalized_language = normalize_language(language);
        const bool cjk_join = normalized_language == "Chinese" || normalized_language == "Japanese";

        std::string out;
        for (size_t i = 0; i < units.size(); ++i) {
            if (i > 0 && !cjk_join) {
                out.push_back(' ');
            }
            out += units[i];
        }
        return out;
    }

    static float average_logprob_for_range(
        const std::vector<decoder_token_span> & spans,
        size_t byte_start,
        size_t byte_end,
        bool & has_confidence
    ) {
        has_confidence = false;
        if (byte_end <= byte_start) {
            return 0.0f;
        }

        double weighted_sum = 0.0;
        size_t weighted_bytes = 0;
        for (const decoder_token_span & span : spans) {
            const size_t overlap_start = std::max(byte_start, span.byte_start);
            const size_t overlap_end = std::min(byte_end, span.byte_end);
            if (overlap_end <= overlap_start) {
                continue;
            }

            const size_t overlap_bytes = overlap_end - overlap_start;
            weighted_sum += static_cast<double>(overlap_bytes) * span.logprob;
            weighted_bytes += overlap_bytes;
        }

        if (weighted_bytes == 0) {
            return 0.0f;
        }

        has_confidence = true;
        return static_cast<float>(weighted_sum / static_cast<double>(weighted_bytes));
    }

    static scored_fragment extract_scored_fragment(
        const transcript_chunk_view & chunk,
        float segment_start_sec,
        float segment_end_sec,
        bool include_end
    ) {
        const std::string & chunk_text = chunk.transcript.text;
        if (chunk_text.empty()) {
            return {};
        }

        if (segment_end_sec <= segment_start_sec + 1.0e-6f) {
            return {};
        }

        if (chunk.alignment.items.empty()) {
            scored_fragment out;
            const auto [trimmed_start, trimmed_end] = trim_ascii_bounds(chunk_text, 0, chunk_text.size());
            if (trimmed_end > trimmed_start) {
                out.text = chunk_text.substr(trimmed_start, trimmed_end - trimmed_start);
                out.avg_logprob = average_logprob_for_range(
                    chunk.transcript.text_token_spans,
                    trimmed_start,
                    trimmed_end,
                    out.has_confidence
                );
            }
            return out;
        }

        size_t first_keep = chunk.alignment.items.size();
        size_t last_keep = chunk.alignment.items.size();
        for (size_t i = 0; i < chunk.alignment.items.size(); ++i) {
            const float midpoint =
                chunk.window_start_sec + 0.5f * (chunk.alignment.items[i].start_time + chunk.alignment.items[i].end_time);
            const bool keep = include_end
                ? (midpoint >= segment_start_sec && midpoint <= segment_end_sec + 1.0e-3f)
                : (midpoint >= segment_start_sec && midpoint < segment_end_sec);
            if (!keep) {
                continue;
            }

            if (first_keep == chunk.alignment.items.size()) {
                first_keep = i;
            }
            last_keep = i;
        }

        scored_fragment out;
        if (first_keep == chunk.alignment.items.size()) {
            return out;
        }

        if (chunk.spans.size() == chunk.alignment.items.size()) {
            const size_t raw_start = std::min(chunk.spans[first_keep].byte_start, chunk_text.size());
            const size_t raw_end = std::min(chunk.spans[last_keep].byte_end, chunk_text.size());
            const auto [trimmed_start, trimmed_end] = trim_ascii_bounds(chunk_text, raw_start, raw_end);
            if (trimmed_end > trimmed_start) {
                out.text = chunk_text.substr(trimmed_start, trimmed_end - trimmed_start);
                out.avg_logprob = average_logprob_for_range(
                    chunk.transcript.text_token_spans,
                    trimmed_start,
                    trimmed_end,
                    out.has_confidence
                );
                return out;
            }
        }

        std::vector<std::string> fallback_units;
        fallback_units.reserve(last_keep - first_keep + 1);
        for (size_t i = first_keep; i <= last_keep; ++i) {
            fallback_units.push_back(chunk.alignment.items[i].text);
        }
        out.text = join_normalized_units(fallback_units, chunk.language);
        return out;
    }

    static bool starts_with(const std::string & value, const std::string & prefix) {
        return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
    }

    static const scored_fragment & choose_boundary_fragment(
        const scored_fragment & left,
        const scored_fragment & right
    ) {
        if (left.text.empty()) {
            return right;
        }
        if (right.text.empty()) {
            return left;
        }

        const std::string left_trimmed = trim_ascii(left.text);
        const std::string right_trimmed = trim_ascii(right.text);

        if (left.has_confidence && right.has_confidence) {
            const float delta = left.avg_logprob - right.avg_logprob;
            if (left_trimmed == right_trimmed) {
                return delta >= 0.0f ? left : right;
            }
            if (starts_with(right_trimmed, left_trimmed) && delta > -0.35f) {
                return right;
            }
            if (starts_with(left_trimmed, right_trimmed) && delta < 0.35f) {
                return left;
            }
            if (std::fabs(delta) > 0.05f) {
                return delta > 0.0f ? left : right;
            }
        } else if (left.has_confidence != right.has_confidence) {
            return left.has_confidence ? left : right;
        }

        return left_trimmed.size() >= right_trimmed.size() ? left : right;
    }

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

    static parsed_asr_output parse_asr_output(
        const std::string & raw_text,
        const std::string & forced_language
    ) {
        if (raw_text.empty()) {
            return {};
        }

        if (!forced_language.empty()) {
            parsed_asr_output out;
            out.language = forced_language;
            out.text = raw_text;
            return out;
        }

        const std::string tag = "<asr_text>";
        const size_t tag_pos = raw_text.find(tag);
        if (tag_pos == std::string::npos) {
            parsed_asr_output out;
            out.text = raw_text;
            return out;
        }

        const std::string meta = raw_text.substr(0, tag_pos);
        parsed_asr_output out;
        out.text_byte_start = tag_pos + tag.size();
        out.text = raw_text.substr(out.text_byte_start);

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

        out.language = language;
        return out;
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

q3asr_aligner_context_params q3asr_aligner_context_default_params(void) {
    q3asr_aligner_context_params params = {};
    params.use_gpu = 1;
    params.n_threads = default_thread_count();
    return params;
}

q3asr_transcribe_params q3asr_transcribe_default_params(void) {
    q3asr_transcribe_params params = {};
    params.context = nullptr;
    params.max_tokens = 256;
    params.temperature = 0.0f;
    params.aligner_context = nullptr;
    params.max_audio_chunk_seconds = 0.0f;
    params.audio_chunk_overlap_seconds = 0.0f;
    params.raw_text_callback = nullptr;
    params.raw_text_callback_user_data = nullptr;
    params.progress_callback = nullptr;
    params.progress_callback_user_data = nullptr;
    return params;
}

q3asr_align_params q3asr_align_default_params(void) {
    q3asr_align_params params = {};
    params.max_chunk_seconds = 180.0f;
    params.chunk_search_expand_seconds = 5.0f;
    params.min_chunk_window_ms = 100.0f;
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

q3asr_aligner_context * q3asr_aligner_context_create(const q3asr_aligner_context_params * params) {
    const q3asr_aligner_context_params effective =
        params != nullptr ? *params : q3asr_aligner_context_default_params();

    if (effective.aligner_model_path == nullptr) {
        return nullptr;
    }

    q3asr::configure_llama_logging();

    auto ctx = std::make_unique<q3asr_aligner_context>();
    ctx->aligner = std::make_unique<q3asr::ForcedAligner>();

    q3asr::aligner_load_params load_params;
    load_params.use_gpu = effective.use_gpu != 0;
    load_params.n_threads = effective.n_threads;
    load_params.korean_dict_path = effective.korean_dict_path != nullptr ? effective.korean_dict_path : "";

    if (!ctx->aligner->load_model(effective.aligner_model_path, load_params)) {
        ctx->last_error = ctx->aligner->error();
        return ctx.release();
    }

    return ctx.release();
}

void q3asr_context_destroy(q3asr_context * ctx) {
    delete ctx;
}

void q3asr_aligner_context_destroy(q3asr_aligner_context * ctx) {
    delete ctx;
}

const char * q3asr_context_last_error(const q3asr_context * ctx) {
    return ctx == nullptr ? "" : ctx->last_error.c_str();
}

const char * q3asr_aligner_context_last_error(const q3asr_aligner_context * ctx) {
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

int q3asr_align_pcm_f32(
    q3asr_aligner_context * ctx,
    const float * samples,
    int n_samples,
    const char * text,
    const char * language,
    q3asr_alignment_result * out_result
) {
    return q3asr_align_pcm_f32_ex(ctx, samples, n_samples, text, language, nullptr, out_result);
}

int q3asr_align_pcm_f32_ex(
    q3asr_aligner_context * ctx,
    const float * samples,
    int n_samples,
    const char * text,
    const char * language,
    const q3asr_align_params * params,
    q3asr_alignment_result * out_result
) {
    if (
        ctx == nullptr ||
        ctx->aligner == nullptr ||
        samples == nullptr ||
        n_samples <= 0 ||
        text == nullptr ||
        out_result == nullptr
    ) {
        return 0;
    }

    q3asr_alignment_result_clear(out_result);

    const q3asr_align_params effective =
        params != nullptr ? *params : q3asr_align_default_params();
    q3asr::align_runtime_params runtime_params;
    runtime_params.max_chunk_seconds = effective.max_chunk_seconds;
    runtime_params.chunk_search_expand_seconds = effective.chunk_search_expand_seconds;
    runtime_params.min_chunk_window_ms = effective.min_chunk_window_ms;

    const q3asr::alignment_result result = ctx->aligner->align(
        samples,
        n_samples,
        text,
        language != nullptr ? language : "",
        runtime_params
    );

    if (!result.success) {
        ctx->last_error = result.error_msg.empty() ? ctx->aligner->error() : result.error_msg;
        return 0;
    }

    out_result->items = static_cast<q3asr_aligned_item *>(
        std::calloc(result.items.size(), sizeof(q3asr_aligned_item))
    );
    if (!result.items.empty() && out_result->items == nullptr) {
        ctx->last_error = "Failed to allocate alignment result items";
        return 0;
    }

    out_result->n_items = result.items.size();
    for (size_t i = 0; i < result.items.size(); ++i) {
        out_result->items[i].text = q3asr_strdup(result.items[i].text);
        if (out_result->items[i].text == nullptr && !result.items[i].text.empty()) {
            q3asr_alignment_result_clear(out_result);
            ctx->last_error = "Failed to allocate alignment result text";
            return 0;
        }
        out_result->items[i].start_time = result.items[i].start_time;
        out_result->items[i].end_time = result.items[i].end_time;
    }

    return 1;
}

int q3asr_align_wav_file(
    q3asr_aligner_context * ctx,
    const char * wav_path,
    const char * text,
    const char * language,
    q3asr_alignment_result * out_result
) {
    return q3asr_align_wav_file_ex(ctx, wav_path, text, language, nullptr, out_result);
}

int q3asr_align_wav_file_ex(
    q3asr_aligner_context * ctx,
    const char * wav_path,
    const char * text,
    const char * language,
    const q3asr_align_params * params,
    q3asr_alignment_result * out_result
) {
    if (ctx == nullptr || wav_path == nullptr || text == nullptr || out_result == nullptr) {
        return 0;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (!load_wav(wav_path, samples, sample_rate)) {
        ctx->last_error = "Failed to load WAV file";
        return 0;
    }

    if (sample_rate != QWEN_SAMPLE_RATE) {
        ctx->last_error = "Only 16 kHz WAV input is supported";
        return 0;
    }

    return q3asr_align_pcm_f32_ex(
        ctx,
        samples.data(),
        static_cast<int>(samples.size()),
        text,
        language,
        params,
        out_result
    );
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

void q3asr_alignment_result_clear(q3asr_alignment_result * result) {
    if (result == nullptr) {
        return;
    }

    if (result->items != nullptr) {
        for (size_t i = 0; i < result->n_items; ++i) {
            std::free(result->items[i].text);
        }
    }

    std::free(result->items);
    result->items = nullptr;
    result->n_items = 0;
}
