#include "forced_aligner.h"

#include "ggml-cpu.h"
#include "mel_spectrogram.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace q3asr {

namespace {

constexpr int Q3ASR_FA_MAX_NODES = 16384;
constexpr int Q3ASR_FA_AUDIO_N_WINDOW = 50;
constexpr int Q3ASR_FA_AUDIO_N_WINDOW_INFER = 800;
constexpr float Q3ASR_FA_DEFAULT_MAX_CHUNK_SECONDS = 180.0f;
constexpr float Q3ASR_FA_DEFAULT_SEARCH_EXPAND_SECONDS = 5.0f;
constexpr float Q3ASR_FA_DEFAULT_MIN_WINDOW_MS = 100.0f;
constexpr float Q3ASR_FA_MIN_INPUT_SECONDS = 0.5f;

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
}

int32_t get_u32(const gguf_context * ctx, const char * key, int32_t fallback) {
    const int64_t idx = gguf_find_key(ctx, key);
    return idx >= 0 ? static_cast<int32_t>(gguf_get_val_u32(ctx, idx)) : fallback;
}

float get_f32(const gguf_context * ctx, const char * key, float fallback) {
    const int64_t idx = gguf_find_key(ctx, key);
    return idx >= 0 ? gguf_get_val_f32(ctx, idx) : fallback;
}

std::string to_lower_ascii(const std::string & value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

ggml_tensor * require_tensor(ggml_context * ctx, const std::string & name, std::string & error_msg) {
    ggml_tensor * tensor = ggml_get_tensor(ctx, name.c_str());
    if (tensor == nullptr) {
        error_msg = "Missing tensor in aligner GGUF: " + name;
    }
    return tensor;
}

ggml_tensor * optional_tensor(ggml_context * ctx, const std::string & name) {
    return ggml_get_tensor(ctx, name.c_str());
}

void compute_sinusoidal_pe(float * pe, int n_ctx, int d_model) {
    const int half_dim = d_model / 2;
    for (int pos = 0; pos < n_ctx; ++pos) {
        for (int i = 0; i < half_dim; ++i) {
            const float div_term = std::exp(-std::log(10000.0f) * i / (half_dim - 1));
            const float angle = pos * div_term;
            pe[pos * d_model + i] = std::sin(angle);
            pe[pos * d_model + half_dim + i] = std::cos(angle);
        }
    }
}

int32_t chunk_output_len(int32_t chunk_frames) {
    int32_t len = chunk_frames;
    for (int i = 0; i < 3; ++i) {
        len = (len - 1) / 2 + 1;
    }
    return len;
}

int32_t get_feat_extract_output_lengths(int32_t input_lengths) {
    const int32_t input_lengths_leave = input_lengths % 100;
    const int32_t feat_lengths = (input_lengths_leave - 1) / 2 + 1;
    return ((feat_lengths - 1) / 2 + 1 - 1) / 2 + 1 + (input_lengths / 100) * 13;
}

float round_millis(float value) {
    return std::round(value * 1000.0f) / 1000.0f;
}

std::vector<split_audio_chunk> split_audio_into_chunks_impl(
    const float * samples,
    int n_samples,
    int sample_rate,
    const align_runtime_params & params
) {
    std::vector<split_audio_chunk> chunks;
    if (samples == nullptr || n_samples <= 0 || sample_rate <= 0) {
        return chunks;
    }

    const float max_chunk_sec =
        params.max_chunk_seconds > 0.0f ? params.max_chunk_seconds : Q3ASR_FA_DEFAULT_MAX_CHUNK_SECONDS;
    const float search_expand_sec =
        params.chunk_search_expand_seconds >= 0.0f ? params.chunk_search_expand_seconds : Q3ASR_FA_DEFAULT_SEARCH_EXPAND_SECONDS;
    const float min_window_ms =
        params.min_chunk_window_ms > 0.0f ? params.min_chunk_window_ms : Q3ASR_FA_DEFAULT_MIN_WINDOW_MS;

    const float total_sec = static_cast<float>(n_samples) / static_cast<float>(sample_rate);
    if (max_chunk_sec <= 0.0f || total_sec <= max_chunk_sec) {
        split_audio_chunk chunk;
        chunk.samples.assign(samples, samples + n_samples);
        chunk.original_n_samples = n_samples;
        chunks.push_back(std::move(chunk));
        return chunks;
    }

    const int max_len = std::max(1, static_cast<int>(std::floor(max_chunk_sec * sample_rate)));
    const int expand = std::max(0, static_cast<int>(std::floor(search_expand_sec * sample_rate)));
    const int win = std::max(4, static_cast<int>(std::floor((min_window_ms / 1000.0f) * sample_rate)));
    const int min_len = std::max(1, static_cast<int>(std::floor(Q3ASR_FA_MIN_INPUT_SECONDS * sample_rate)));

    int start = 0;
    float offset_sec = 0.0f;

    while ((n_samples - start) > max_len) {
        const int cut = start + max_len;
        const int left = std::max(start, cut - expand);
        const int right = std::min(n_samples, cut + expand);

        int boundary = cut;
        if (right - left > win) {
            const int seg_len = right - left;
            std::vector<float> prefix(static_cast<size_t>(seg_len) + 1, 0.0f);
            for (int i = 0; i < seg_len; ++i) {
                prefix[static_cast<size_t>(i + 1)] =
                    prefix[static_cast<size_t>(i)] + std::fabs(samples[left + i]);
            }

            float best_sum = std::numeric_limits<float>::infinity();
            int best_pos = 0;
            for (int i = 0; i <= seg_len - win; ++i) {
                const float sum = prefix[static_cast<size_t>(i + win)] - prefix[static_cast<size_t>(i)];
                if (sum < best_sum) {
                    best_sum = sum;
                    best_pos = i;
                }
            }

            float best_inner = std::numeric_limits<float>::infinity();
            int best_inner_pos = 0;
            for (int i = 0; i < win; ++i) {
                const float value = std::fabs(samples[left + best_pos + i]);
                if (value < best_inner) {
                    best_inner = value;
                    best_inner_pos = i;
                }
            }

            boundary = left + best_pos + best_inner_pos;
        }

        boundary = std::max(boundary, start + 1);
        boundary = std::min(boundary, n_samples);

        split_audio_chunk chunk;
        chunk.samples.assign(samples + start, samples + boundary);
        chunk.original_n_samples = boundary - start;
        chunk.offset_sec = offset_sec;

        if (static_cast<int>(chunk.samples.size()) < min_len) {
            chunk.samples.resize(static_cast<size_t>(min_len), 0.0f);
        }

        chunks.push_back(std::move(chunk));
        offset_sec += static_cast<float>(boundary - start) / static_cast<float>(sample_rate);
        start = boundary;
    }

    split_audio_chunk tail;
    tail.samples.assign(samples + start, samples + n_samples);
    tail.original_n_samples = n_samples - start;
    tail.offset_sec = offset_sec;
    if (static_cast<int>(tail.samples.size()) < min_len) {
        tail.samples.resize(static_cast<size_t>(min_len), 0.0f);
    }
    chunks.push_back(std::move(tail));

    return chunks;
}

std::vector<std::pair<size_t, size_t>> assign_word_ranges(
    const std::vector<split_audio_chunk> & chunks,
    size_t n_words,
    int total_n_samples
) {
    std::vector<std::pair<size_t, size_t>> spans;
    spans.reserve(chunks.size());

    if (chunks.empty()) {
        return spans;
    }

    size_t begin = 0;
    int consumed_samples = 0;

    for (size_t i = 0; i < chunks.size(); ++i) {
        const bool last_chunk = i + 1 == chunks.size();
        size_t end = n_words;

        if (!last_chunk && total_n_samples > 0 && n_words > 0) {
            consumed_samples += chunks[i].original_n_samples;
            const double ratio = static_cast<double>(consumed_samples) / static_cast<double>(total_n_samples);
            const double raw_target = ratio * static_cast<double>(n_words);
            end = static_cast<size_t>(std::llround(raw_target));
            end = std::max(end, begin);
            end = std::min(end, n_words);
        }

        spans.emplace_back(begin, end);
        begin = end;
    }

    if (!spans.empty()) {
        spans.back().second = n_words;
    }

    return spans;
}

size_t utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0) {
        return 1;
    }
    if ((c & 0xE0) == 0xC0) {
        return 2;
    }
    if ((c & 0xF0) == 0xE0) {
        return 3;
    }
    if ((c & 0xF8) == 0xF0) {
        return 4;
    }
    return 1;
}

std::vector<std::string> split_utf8_chars(const std::string & value) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < value.size()) {
        size_t len = utf8_char_len(static_cast<unsigned char>(value[i]));
        if (i + len > value.size()) {
            len = 1;
        }
        chars.push_back(value.substr(i, len));
        i += len;
    }
    return chars;
}

bool decode_utf8_codepoint(const std::string & value, uint32_t & codepoint) {
    if (value.empty()) {
        return false;
    }

    const unsigned char c0 = static_cast<unsigned char>(value[0]);
    if ((c0 & 0x80) == 0) {
        codepoint = c0;
        return true;
    }
    if ((c0 & 0xE0) == 0xC0 && value.size() >= 2) {
        codepoint = ((c0 & 0x1F) << 6) |
                    (static_cast<unsigned char>(value[1]) & 0x3F);
        return true;
    }
    if ((c0 & 0xF0) == 0xE0 && value.size() >= 3) {
        codepoint = ((c0 & 0x0F) << 12) |
                    ((static_cast<unsigned char>(value[1]) & 0x3F) << 6) |
                    (static_cast<unsigned char>(value[2]) & 0x3F);
        return true;
    }
    if ((c0 & 0xF8) == 0xF0 && value.size() >= 4) {
        codepoint = ((c0 & 0x07) << 18) |
                    ((static_cast<unsigned char>(value[1]) & 0x3F) << 12) |
                    ((static_cast<unsigned char>(value[2]) & 0x3F) << 6) |
                    (static_cast<unsigned char>(value[3]) & 0x3F);
        return true;
    }

    codepoint = c0;
    return true;
}

bool is_cjk_char(uint32_t code) {
    return
        (0x4E00 <= code && code <= 0x9FFF) ||
        (0x3400 <= code && code <= 0x4DBF) ||
        (0x20000 <= code && code <= 0x2A6DF) ||
        (0x2A700 <= code && code <= 0x2B73F) ||
        (0x2B740 <= code && code <= 0x2B81F) ||
        (0x2B820 <= code && code <= 0x2CEAF) ||
        (0xF900 <= code && code <= 0xFAFF);
}

bool is_hangul_char(uint32_t code) {
    return
        (0xAC00 <= code && code <= 0xD7AF) ||
        (0x1100 <= code && code <= 0x11FF) ||
        (0x3130 <= code && code <= 0x318F) ||
        (0xA960 <= code && code <= 0xA97F) ||
        (0xD7B0 <= code && code <= 0xD7FF);
}

bool is_kana_char(uint32_t code) {
    return
        (0x3040 <= code && code <= 0x309F) ||
        (0x30A0 <= code && code <= 0x30FF) ||
        (0x31F0 <= code && code <= 0x31FF) ||
        (0xFF66 <= code && code <= 0xFF9F);
}

bool is_fullwidth_letter_or_digit(uint32_t code) {
    return
        (0xFF10 <= code && code <= 0xFF19) ||
        (0xFF21 <= code && code <= 0xFF3A) ||
        (0xFF41 <= code && code <= 0xFF5A);
}

bool is_kept_char(const std::string & ch) {
    if (ch == "'") {
        return true;
    }

    uint32_t code = 0;
    if (!decode_utf8_codepoint(ch, code)) {
        return false;
    }

    if (code < 128) {
        return std::isalnum(static_cast<unsigned char>(code)) != 0;
    }

    return is_cjk_char(code) || is_hangul_char(code) || is_kana_char(code) || is_fullwidth_letter_or_digit(code);
}

std::string clean_token(const std::string & token) {
    std::string cleaned;
    for (const std::string & ch : split_utf8_chars(token)) {
        if (is_kept_char(ch)) {
            cleaned += ch;
        }
    }
    return cleaned;
}

std::vector<std::string> split_segment_with_chinese(const std::string & seg) {
    std::vector<std::string> tokens;
    std::string buffer;

    for (const std::string & ch : split_utf8_chars(seg)) {
        uint32_t code = 0;
        decode_utf8_codepoint(ch, code);
        if (is_cjk_char(code)) {
            if (!buffer.empty()) {
                tokens.push_back(buffer);
                buffer.clear();
            }
            tokens.push_back(ch);
        } else {
            buffer += ch;
        }
    }

    if (!buffer.empty()) {
        tokens.push_back(buffer);
    }

    return tokens;
}

std::vector<std::string> tokenize_space_lang(const std::string & text) {
    std::vector<std::string> tokens;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            ++i;
        }
        if (i >= text.size()) {
            break;
        }
        const size_t start = i;
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) == 0) {
            ++i;
        }
        const std::string cleaned = clean_token(text.substr(start, i - start));
        if (cleaned.empty()) {
            continue;
        }
        const auto seg_tokens = split_segment_with_chinese(cleaned);
        tokens.insert(tokens.end(), seg_tokens.begin(), seg_tokens.end());
    }
    return tokens;
}

std::vector<std::string> tokenize_japanese_fallback(const std::string & text) {
    std::vector<std::string> tokens;
    std::string buffer;

    auto flush = [&]() {
        if (!buffer.empty()) {
            tokens.push_back(buffer);
            buffer.clear();
        }
    };

    for (const std::string & ch : split_utf8_chars(text)) {
        uint32_t code = 0;
        decode_utf8_codepoint(ch, code);

        if (!is_kept_char(ch)) {
            flush();
            continue;
        }

        if (is_cjk_char(code)) {
            flush();
            tokens.push_back(ch);
            continue;
        }

        buffer += ch;
    }

    flush();
    return tokens;
}

size_t utf8_strlen(const std::string & value) {
    size_t count = 0;
    size_t i = 0;
    while (i < value.size()) {
        i += utf8_char_len(static_cast<unsigned char>(value[i]));
        ++count;
    }
    return count;
}

std::string utf8_substr(const std::string & value, size_t char_start, size_t char_count) {
    size_t byte_start = 0;
    for (size_t c = 0; c < char_start && byte_start < value.size(); ++c) {
        byte_start += utf8_char_len(static_cast<unsigned char>(value[byte_start]));
    }

    size_t byte_end = byte_start;
    for (size_t c = 0; c < char_count && byte_end < value.size(); ++c) {
        byte_end += utf8_char_len(static_cast<unsigned char>(value[byte_end]));
    }

    return value.substr(byte_start, byte_end - byte_start);
}

std::vector<std::string> tokenize_korean_dict(
    const std::string & text,
    const std::unordered_set<std::string> & ko_dict
) {
    std::vector<std::string> whitespace_words;
    size_t i = 0;
    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            ++i;
        }
        if (i >= text.size()) {
            break;
        }
        const size_t start = i;
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) == 0) {
            ++i;
        }
        const std::string cleaned = clean_token(text.substr(start, i - start));
        if (!cleaned.empty()) {
            whitespace_words.push_back(cleaned);
        }
    }

    std::vector<std::string> result;
    for (const std::string & word : whitespace_words) {
        const size_t length = utf8_strlen(word);
        if (length <= 2) {
            result.push_back(word);
            continue;
        }

        float best_score = -1.0e9f;
        size_t best_left_len = 0;
        std::string best_left;
        std::string best_right;

        for (size_t e = 2; e <= length; ++e) {
            const std::string left = utf8_substr(word, 0, e);
            const std::string right = utf8_substr(word, e, length - e);
            const float score = ko_dict.count(left) != 0 ? 1.0f : 0.0f;

            if (score > best_score || (score == best_score && e > best_left_len)) {
                best_score = score;
                best_left_len = e;
                best_left = left;
                best_right = right;
            }
        }

        if (!best_left.empty()) {
            result.push_back(best_left);
        }
        if (!best_right.empty()) {
            result.push_back(best_right);
        }
    }

    return result;
}

struct provisional_span {
    std::string text;
    size_t byte_start = 0;
    size_t byte_end = 0;
};

void flush_provisional_buffer(
    std::vector<provisional_span> & spans,
    std::string & buffer,
    size_t & buffer_start,
    size_t & buffer_end
) {
    if (buffer.empty()) {
        return;
    }

    spans.push_back({buffer, buffer_start, buffer_end});
    buffer.clear();
    buffer_start = 0;
    buffer_end = 0;
}

std::vector<normalized_word_span> normalize_space_lang_with_spans(const std::string & text) {
    std::vector<normalized_word_span> result;
    size_t i = 0;

    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            ++i;
        }
        if (i >= text.size()) {
            break;
        }

        const size_t seg_start = i;
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) == 0) {
            i += utf8_char_len(static_cast<unsigned char>(text[i]));
        }
        const size_t seg_end = i;

        std::vector<provisional_span> spans;
        std::string buffer;
        size_t buffer_start = 0;
        size_t buffer_end = 0;
        bool saw_cjk = false;

        for (size_t pos = seg_start; pos < seg_end;) {
            const size_t len = utf8_char_len(static_cast<unsigned char>(text[pos]));
            const size_t next = std::min(pos + len, seg_end);
            const std::string ch = text.substr(pos, next - pos);

            if (is_kept_char(ch)) {
                uint32_t code = 0;
                decode_utf8_codepoint(ch, code);
                if (is_cjk_char(code)) {
                    flush_provisional_buffer(spans, buffer, buffer_start, buffer_end);
                    spans.push_back({ch, pos, next});
                    saw_cjk = true;
                } else {
                    if (buffer.empty()) {
                        buffer_start = pos;
                    }
                    buffer += ch;
                    buffer_end = next;
                }
            }

            pos = next;
        }

        flush_provisional_buffer(spans, buffer, buffer_start, buffer_end);
        if (spans.empty()) {
            continue;
        }

        if (spans.size() == 1 && !saw_cjk) {
            spans[0].byte_start = seg_start;
            spans[0].byte_end = seg_end;
        }

        for (const provisional_span & span : spans) {
            result.push_back({span.text, span.byte_start, span.byte_end});
        }
    }

    return result;
}

std::vector<normalized_word_span> normalize_japanese_with_spans(const std::string & text) {
    std::vector<normalized_word_span> result;
    std::string buffer;
    size_t buffer_start = 0;
    size_t buffer_end = 0;

    auto flush = [&]() {
        if (!buffer.empty()) {
            result.push_back({buffer, buffer_start, buffer_end});
            buffer.clear();
            buffer_start = 0;
            buffer_end = 0;
        }
    };

    for (size_t pos = 0; pos < text.size();) {
        const size_t len = utf8_char_len(static_cast<unsigned char>(text[pos]));
        const size_t next = std::min(pos + len, text.size());
        const std::string ch = text.substr(pos, next - pos);

        if (!is_kept_char(ch)) {
            flush();
            pos = next;
            continue;
        }

        uint32_t code = 0;
        decode_utf8_codepoint(ch, code);
        if (is_cjk_char(code)) {
            flush();
            result.push_back({ch, pos, next});
        } else {
            if (buffer.empty()) {
                buffer_start = pos;
            }
            buffer += ch;
            buffer_end = next;
        }

        pos = next;
    }

    flush();
    return result;
}

std::vector<normalized_word_span> normalize_korean_with_spans(
    const std::string & text,
    const std::unordered_set<std::string> & ko_dict
) {
    std::vector<normalized_word_span> result;
    size_t i = 0;

    while (i < text.size()) {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) != 0) {
            ++i;
        }
        if (i >= text.size()) {
            break;
        }

        const size_t seg_start = i;
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i])) == 0) {
            i += utf8_char_len(static_cast<unsigned char>(text[i]));
        }
        const size_t seg_end = i;

        const std::string cleaned = clean_token(text.substr(seg_start, seg_end - seg_start));
        if (cleaned.empty()) {
            continue;
        }

        const size_t length = utf8_strlen(cleaned);
        if (length <= 2 || ko_dict.empty()) {
            result.push_back({cleaned, seg_start, seg_end});
            continue;
        }

        float best_score = -1.0e9f;
        size_t best_left_len = 0;
        std::string best_left;
        std::string best_right;

        for (size_t e = 2; e <= length; ++e) {
            const std::string left = utf8_substr(cleaned, 0, e);
            const std::string right = utf8_substr(cleaned, e, length - e);
            const float score = ko_dict.count(left) != 0 ? 1.0f : 0.0f;

            if (score > best_score || (score == best_score && e > best_left_len)) {
                best_score = score;
                best_left_len = e;
                best_left = left;
                best_right = right;
            }
        }

        if (best_right.empty()) {
            result.push_back({best_left.empty() ? cleaned : best_left, seg_start, seg_end});
            continue;
        }

        const size_t left_bytes = utf8_substr(cleaned, 0, best_left_len).size();
        const size_t split_pos = std::min(seg_start + left_bytes, seg_end);
        result.push_back({best_left, seg_start, split_pos});
        result.push_back({best_right, split_pos, seg_end});
    }

    return result;
}

const std::vector<std::string> & get_byte_to_unicode_table() {
    static std::vector<std::string> table;
    if (!table.empty()) {
        return table;
    }

    table.resize(256);
    std::vector<int> byte_to_cp(256, 0);
    std::vector<bool> assigned(256, false);

    auto mark = [&](int lo, int hi) {
        for (int b = lo; b <= hi; ++b) {
            byte_to_cp[b] = b;
            assigned[b] = true;
        }
    };

    mark(0x21, 0x7E);
    mark(0xA1, 0xAC);
    mark(0xAE, 0xFF);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (!assigned[b]) {
            byte_to_cp[b] = 256 + n;
            ++n;
        }
    }

    auto cp_to_utf8 = [](int cp) {
        std::string out;
        if (cp < 0x80) {
            out += static_cast<char>(cp);
        } else if (cp < 0x800) {
            out += static_cast<char>(0xC0 | (cp >> 6));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            out += static_cast<char>(0xE0 | (cp >> 12));
            out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            out += static_cast<char>(0x80 | (cp & 0x3F));
        }
        return out;
    };

    for (int b = 0; b < 256; ++b) {
        table[b] = cp_to_utf8(byte_to_cp[b]);
    }

    return table;
}

std::string bytes_to_bpe_string(const std::string & text) {
    const auto & table = get_byte_to_unicode_table();
    std::string result;
    result.reserve(text.size() * 2);
    for (unsigned char c : text) {
        result += table[c];
    }
    return result;
}

std::vector<std::string> bpe_encode_word(
    const std::string & word_bpe,
    const std::unordered_map<std::string, int> & bpe_ranks
) {
    std::vector<std::string> symbols = split_utf8_chars(word_bpe);
    if (symbols.size() <= 1) {
        return symbols;
    }

    while (true) {
        int best_rank = INT_MAX;
        size_t best_pos = 0;

        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const std::string key = symbols[i] + " " + symbols[i + 1];
            const auto it = bpe_ranks.find(key);
            if (it != bpe_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_pos = i;
            }
        }

        if (best_rank == INT_MAX) {
            break;
        }

        std::vector<std::string> merged;
        merged.reserve(symbols.size() - 1);
        for (size_t i = 0; i < symbols.size(); ++i) {
            if (i == best_pos) {
                merged.push_back(symbols[i] + symbols[i + 1]);
                ++i;
            } else {
                merged.push_back(symbols[i]);
            }
        }
        symbols = std::move(merged);
        if (symbols.size() == 1) {
            break;
        }
    }

    return symbols;
}

} // namespace

std::vector<split_audio_chunk> split_audio_into_chunks(
    const float * samples,
    int n_samples,
    int sample_rate,
    const align_runtime_params & params
) {
    return split_audio_into_chunks_impl(samples, n_samples, sample_rate, params);
}

ForcedAligner::~ForcedAligner() {
    if (state_.sched != nullptr) {
        ggml_backend_sched_free(state_.sched);
        state_.sched = nullptr;
    }
    if (state_.backend_gpu != nullptr) {
        ggml_backend_free(state_.backend_gpu);
        state_.backend_gpu = nullptr;
    }
    if (state_.backend_cpu != nullptr) {
        ggml_backend_free(state_.backend_cpu);
        state_.backend_cpu = nullptr;
    }
    free_forced_aligner_model(model_);
}

bool ForcedAligner::load_model(const std::string & model_path, const aligner_load_params & params) {
    error_msg_.clear();
    params_ = params;
    free_forced_aligner_model(model_);
    model_loaded_ = false;

    ggml_context * meta_ctx = nullptr;
    gguf_init_params gguf_params = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };

    gguf_context * gguf_ctx = gguf_init_from_file(model_path.c_str(), gguf_params);
    if (gguf_ctx == nullptr) {
        error_msg_ = "Failed to open aligner GGUF: " + model_path;
        return false;
    }

    model_.ctx = meta_ctx;

    const bool ok =
        parse_hparams(gguf_ctx) &&
        bind_tensors() &&
        load_tensor_data(model_path, gguf_ctx) &&
        load_vocab(gguf_ctx);

    gguf_free(gguf_ctx);

    if (!ok) {
        free_forced_aligner_model(model_);
        return false;
    }

    if (!params_.korean_dict_path.empty() && !load_korean_dict(params_.korean_dict_path)) {
        free_forced_aligner_model(model_);
        error_msg_ = "Failed to load Korean dictionary: " + params_.korean_dict_path;
        return false;
    }

    state_.backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (state_.backend_cpu == nullptr) {
        error_msg_ = "Failed to initialize CPU ggml backend for forced alignment";
        free_forced_aligner_model(model_);
        return false;
    }
    ggml_backend_cpu_set_n_threads(state_.backend_cpu, std::max(1, params_.n_threads));

    if (params_.use_gpu) {
        state_.backend_gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    }

    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_type_t> bufts;

    if (state_.backend_gpu != nullptr) {
        backends.push_back(state_.backend_gpu);
        bufts.push_back(ggml_backend_get_default_buffer_type(state_.backend_gpu));
    }

    backends.push_back(state_.backend_cpu);
    ggml_backend_buffer_type_t cpu_buft = ggml_backend_get_default_buffer_type(state_.backend_cpu);
    if (state_.backend_gpu != nullptr) {
        ggml_backend_dev_t gpu_dev = ggml_backend_get_device(state_.backend_gpu);
        ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(gpu_dev);
        if (host_buft != nullptr) {
            cpu_buft = host_buft;
        }
    }
    bufts.push_back(cpu_buft);

    state_.sched = ggml_backend_sched_new(
        backends.data(),
        bufts.data(),
        static_cast<int>(backends.size()),
        Q3ASR_FA_MAX_NODES,
        false,
        true
    );

    if (state_.sched == nullptr) {
        error_msg_ = "Failed to create forced aligner backend scheduler";
        free_forced_aligner_model(model_);
        return false;
    }

    state_.compute_meta.resize(
        ggml_tensor_overhead() * Q3ASR_FA_MAX_NODES +
        ggml_graph_overhead_custom(Q3ASR_FA_MAX_NODES, false)
    );

    model_loaded_ = true;
    return true;
}

bool ForcedAligner::parse_hparams(gguf_context * ctx) {
    auto & hp = model_.hparams;

    hp.audio_encoder_layers = get_u32(ctx, "qwen3-asr.audio.encoder.layer_count", hp.audio_encoder_layers);
    hp.audio_d_model = get_u32(ctx, "qwen3-asr.audio.encoder.embedding_length", hp.audio_d_model);
    hp.audio_attention_heads = get_u32(ctx, "qwen3-asr.audio.encoder.attention.head_count", hp.audio_attention_heads);
    hp.audio_ffn_dim = get_u32(ctx, "qwen3-asr.audio.encoder.feed_forward_length", hp.audio_ffn_dim);
    hp.audio_num_mel_bins = get_u32(ctx, "qwen3-asr.audio.num_mel_bins", hp.audio_num_mel_bins);
    hp.audio_conv_channels = get_u32(ctx, "qwen3-asr.audio.conv_channels", hp.audio_conv_channels);

    hp.text_decoder_layers = get_u32(ctx, "qwen3-asr.block_count", hp.text_decoder_layers);
    hp.text_hidden_size = get_u32(ctx, "qwen3-asr.embedding_length", hp.text_hidden_size);
    hp.text_attention_heads = get_u32(ctx, "qwen3-asr.attention.head_count", hp.text_attention_heads);
    hp.text_kv_heads = get_u32(ctx, "qwen3-asr.attention.head_count_kv", hp.text_kv_heads);
    hp.text_intermediate_size = get_u32(ctx, "qwen3-asr.feed_forward_length", hp.text_intermediate_size);
    hp.text_head_dim = get_u32(ctx, "qwen3-asr.attention.key_length", hp.text_head_dim);
    hp.text_rms_norm_eps = get_f32(ctx, "qwen3-asr.attention.layer_norm_rms_epsilon", hp.text_rms_norm_eps);
    hp.text_rope_theta = get_f32(ctx, "qwen3-asr.rope.freq_base", hp.text_rope_theta);
    hp.vocab_size = get_u32(ctx, "qwen3-asr.vocab_size", hp.vocab_size);

    hp.classify_num = get_u32(ctx, "qwen3-asr.classify_num", hp.classify_num);
    hp.timestamp_token_id = get_u32(ctx, "qwen3-asr.timestamp_token_id", hp.timestamp_token_id);
    hp.audio_start_token_id = get_u32(ctx, "qwen3-asr.audio.start_token_id", hp.audio_start_token_id);
    hp.audio_end_token_id = get_u32(ctx, "qwen3-asr.audio.end_token_id", hp.audio_end_token_id);
    hp.audio_pad_token_id = get_u32(ctx, "qwen3-asr.audio.pad_token_id", hp.audio_pad_token_id);
    hp.timestamp_segment_time_ms = get_u32(ctx, "qwen3-asr.timestamp_segment_time", hp.timestamp_segment_time_ms);

    if (
        hp.audio_encoder_layers <= 0 ||
        hp.audio_d_model <= 0 ||
        hp.audio_attention_heads <= 0 ||
        hp.audio_ffn_dim <= 0 ||
        hp.text_decoder_layers <= 0 ||
        hp.text_hidden_size <= 0 ||
        hp.text_attention_heads <= 0 ||
        hp.text_kv_heads <= 0 ||
        hp.text_intermediate_size <= 0 ||
        hp.classify_num <= 0
    ) {
        error_msg_ = "Invalid or incomplete forced aligner metadata in GGUF";
        return false;
    }

    return true;
}

bool ForcedAligner::bind_tensors() {
    if (model_.ctx == nullptr) {
        error_msg_ = "Aligner GGUF metadata context was not created";
        return false;
    }

    auto & hp = model_.hparams;

    model_.conv2d1_w = require_tensor(model_.ctx, "audio.encoder.conv1.weight", error_msg_);
    model_.conv2d1_b = require_tensor(model_.ctx, "audio.encoder.conv1.bias", error_msg_);
    model_.conv2d2_w = require_tensor(model_.ctx, "audio.encoder.conv2.weight", error_msg_);
    model_.conv2d2_b = require_tensor(model_.ctx, "audio.encoder.conv2.bias", error_msg_);
    model_.conv2d3_w = require_tensor(model_.ctx, "audio.encoder.conv3.weight", error_msg_);
    model_.conv2d3_b = require_tensor(model_.ctx, "audio.encoder.conv3.bias", error_msg_);
    model_.conv_out_w = require_tensor(model_.ctx, "audio.encoder.conv_out.weight", error_msg_);
    model_.conv_out_b = optional_tensor(model_.ctx, "audio.encoder.conv_out.bias");
    model_.ln_post_w = require_tensor(model_.ctx, "audio.encoder.ln_post.weight", error_msg_);
    model_.ln_post_b = optional_tensor(model_.ctx, "audio.encoder.ln_post.bias");
    model_.proj1_w = require_tensor(model_.ctx, "audio.encoder.proj1.weight", error_msg_);
    model_.proj1_b = optional_tensor(model_.ctx, "audio.encoder.proj1.bias");
    model_.proj2_w = require_tensor(model_.ctx, "audio.encoder.proj2.weight", error_msg_);
    model_.proj2_b = optional_tensor(model_.ctx, "audio.encoder.proj2.bias");
    model_.token_embd = require_tensor(model_.ctx, "token_embd.weight", error_msg_);
    model_.output_norm = require_tensor(model_.ctx, "output_norm.weight", error_msg_);
    model_.classify_head_w = optional_tensor(model_.ctx, "classify_head.weight");
    if (model_.classify_head_w == nullptr) {
        model_.classify_head_w = optional_tensor(model_.ctx, "output.weight");
    }
    if (model_.classify_head_w == nullptr && error_msg_.empty()) {
        error_msg_ = "Missing tensor in aligner GGUF: classify_head.weight";
    }
    model_.classify_head_b = optional_tensor(model_.ctx, "classify_head.bias");
    if (model_.classify_head_b == nullptr) {
        model_.classify_head_b = optional_tensor(model_.ctx, "output.bias");
    }

    if (!error_msg_.empty()) {
        return false;
    }

    model_.encoder_layers.resize(hp.audio_encoder_layers);
    for (int32_t i = 0; i < hp.audio_encoder_layers; ++i) {
        auto & layer = model_.encoder_layers[i];
        const std::string prefix = "audio.encoder.blk." + std::to_string(i) + ".";

        layer.attn_q_w = require_tensor(model_.ctx, prefix + "attn_q.weight", error_msg_);
        layer.attn_q_b = optional_tensor(model_.ctx, prefix + "attn_q.bias");
        layer.attn_k_w = require_tensor(model_.ctx, prefix + "attn_k.weight", error_msg_);
        layer.attn_k_b = optional_tensor(model_.ctx, prefix + "attn_k.bias");
        layer.attn_v_w = require_tensor(model_.ctx, prefix + "attn_v.weight", error_msg_);
        layer.attn_v_b = optional_tensor(model_.ctx, prefix + "attn_v.bias");
        layer.attn_out_w = require_tensor(model_.ctx, prefix + "attn_out.weight", error_msg_);
        layer.attn_out_b = optional_tensor(model_.ctx, prefix + "attn_out.bias");
        layer.attn_norm_w = require_tensor(model_.ctx, prefix + "attn_norm.weight", error_msg_);
        layer.attn_norm_b = optional_tensor(model_.ctx, prefix + "attn_norm.bias");
        layer.ffn_up_w = require_tensor(model_.ctx, prefix + "ffn_up.weight", error_msg_);
        layer.ffn_up_b = optional_tensor(model_.ctx, prefix + "ffn_up.bias");
        layer.ffn_down_w = require_tensor(model_.ctx, prefix + "ffn_down.weight", error_msg_);
        layer.ffn_down_b = optional_tensor(model_.ctx, prefix + "ffn_down.bias");
        layer.ffn_norm_w = require_tensor(model_.ctx, prefix + "ffn_norm.weight", error_msg_);
        layer.ffn_norm_b = optional_tensor(model_.ctx, prefix + "ffn_norm.bias");

        if (!error_msg_.empty()) {
            return false;
        }
    }

    model_.decoder_layers.resize(hp.text_decoder_layers);
    for (int32_t i = 0; i < hp.text_decoder_layers; ++i) {
        auto & layer = model_.decoder_layers[i];
        const std::string prefix = "blk." + std::to_string(i) + ".";

        layer.attn_norm = require_tensor(model_.ctx, prefix + "attn_norm.weight", error_msg_);
        layer.attn_q = require_tensor(model_.ctx, prefix + "attn_q.weight", error_msg_);
        layer.attn_k = require_tensor(model_.ctx, prefix + "attn_k.weight", error_msg_);
        layer.attn_v = require_tensor(model_.ctx, prefix + "attn_v.weight", error_msg_);
        layer.attn_output = require_tensor(model_.ctx, prefix + "attn_output.weight", error_msg_);
        layer.attn_q_norm = require_tensor(model_.ctx, prefix + "attn_q_norm.weight", error_msg_);
        layer.attn_k_norm = require_tensor(model_.ctx, prefix + "attn_k_norm.weight", error_msg_);
        layer.ffn_norm = require_tensor(model_.ctx, prefix + "ffn_norm.weight", error_msg_);
        layer.ffn_gate = require_tensor(model_.ctx, prefix + "ffn_gate.weight", error_msg_);
        layer.ffn_up = require_tensor(model_.ctx, prefix + "ffn_up.weight", error_msg_);
        layer.ffn_down = require_tensor(model_.ctx, prefix + "ffn_down.weight", error_msg_);

        if (!error_msg_.empty()) {
            return false;
        }
    }

    return true;
}

bool ForcedAligner::load_tensor_data(const std::string & path, gguf_context * ctx) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        error_msg_ = "Failed to open aligner GGUF for mmap: " + path;
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        error_msg_ = "Failed to stat aligner GGUF: " + path;
        close(fd);
        return false;
    }

    void * mmap_addr = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    if (mmap_addr == MAP_FAILED) {
        error_msg_ = "Failed to mmap aligner GGUF: " + path;
        return false;
    }

    model_.mmap_addr = mmap_addr;
    model_.mmap_size = static_cast<size_t>(st.st_size);

    const size_t data_offset = gguf_get_data_offset(ctx);
    const size_t total_size = static_cast<size_t>(st.st_size) - data_offset;
    auto * data_base = static_cast<uint8_t *>(mmap_addr) + data_offset;

    size_t max_tensor_size = 0;
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        max_tensor_size = std::max(max_tensor_size, gguf_get_tensor_size(ctx, i));
    }

    if (params_.use_gpu) {
        ggml_backend_dev_t gpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (gpu_dev != nullptr) {
            model_.buffer = ggml_backend_dev_buffer_from_host_ptr(gpu_dev, data_base, total_size, max_tensor_size);
        }
    }
    if (model_.buffer == nullptr) {
        model_.buffer = ggml_backend_cpu_buffer_from_ptr(data_base, total_size);
    }
    if (model_.buffer == nullptr) {
        error_msg_ = "Failed to create a backend buffer for aligner weights";
        return false;
    }

    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        ggml_tensor * tensor = ggml_get_tensor(model_.ctx, name);
        if (tensor == nullptr) {
            continue;
        }
        tensor->buffer = model_.buffer;
        tensor->data = data_base + gguf_get_tensor_offset(ctx, i);
        model_.tensors[name] = tensor;
    }

    return true;
}

bool ForcedAligner::load_vocab(gguf_context * ctx) {
    const int64_t tokens_idx = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tokens_idx < 0) {
        error_msg_ = "Tokenizer vocabulary not found in aligner GGUF";
        return false;
    }

    const int64_t n_vocab = gguf_get_arr_n(ctx, tokens_idx);
    if (n_vocab <= 0) {
        error_msg_ = "Tokenizer vocabulary is empty in aligner GGUF";
        return false;
    }

    model_.vocab.resize(static_cast<size_t>(n_vocab));
    for (int64_t i = 0; i < n_vocab; ++i) {
        model_.vocab[static_cast<size_t>(i)] = gguf_get_arr_str(ctx, tokens_idx, i);
        model_.token_to_id[model_.vocab[static_cast<size_t>(i)]] = static_cast<int32_t>(i);
    }

    const int64_t merges_idx = gguf_find_key(ctx, "tokenizer.ggml.merges");
    if (merges_idx >= 0) {
        const int64_t n_merges = gguf_get_arr_n(ctx, merges_idx);
        for (int64_t i = 0; i < n_merges; ++i) {
            model_.bpe_ranks[gguf_get_arr_str(ctx, merges_idx, i)] = static_cast<int>(i);
        }
    }

    return true;
}

bool ForcedAligner::load_korean_dict(const std::string & dict_path) {
    std::ifstream file(dict_path);
    if (!file.is_open()) {
        return false;
    }

    model_.ko_dict.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        const size_t pos = line.find(' ');
        const std::string word = pos == std::string::npos ? line : line.substr(0, pos);
        if (!word.empty()) {
            model_.ko_dict.insert(word);
        }
    }

    return true;
}

bool ForcedAligner::encode_audio(const float * mel_data, int n_mel, int n_frames, std::vector<float> & output) {
    const auto & hp = model_.hparams;
    const int n_state = hp.audio_d_model;
    const int n_head = hp.audio_attention_heads;
    const int n_layer = hp.audio_encoder_layers;
    const int n_state_head = n_state / n_head;
    const float eps = hp.audio_layer_norm_eps;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(n_state_head));

    const int32_t chunk_mel_size = Q3ASR_FA_AUDIO_N_WINDOW * 2;
    const int32_t n_chunks = (n_frames + chunk_mel_size - 1) / chunk_mel_size;

    std::vector<int32_t> chunk_lengths(static_cast<size_t>(n_chunks));
    std::vector<int32_t> chunk_out_lens(static_cast<size_t>(n_chunks));
    int32_t max_chunk_len = chunk_mel_size;
    int32_t total_out_frames = 0;

    for (int32_t c = 0; c < n_chunks; ++c) {
        if (c < n_chunks - 1) {
            chunk_lengths[static_cast<size_t>(c)] = chunk_mel_size;
        } else {
            chunk_lengths[static_cast<size_t>(c)] = n_frames - c * chunk_mel_size;
            if (chunk_lengths[static_cast<size_t>(c)] == 0) {
                chunk_lengths[static_cast<size_t>(c)] = chunk_mel_size;
            }
        }
        chunk_out_lens[static_cast<size_t>(c)] = chunk_output_len(chunk_lengths[static_cast<size_t>(c)]);
        total_out_frames += chunk_out_lens[static_cast<size_t>(c)];
    }

    const int32_t max_out_w = chunk_output_len(max_chunk_len);

    ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx0 = ggml_init(params);
    if (ctx0 == nullptr) {
        error_msg_ = "Failed to initialize the aligner conv graph context";
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, Q3ASR_FA_MAX_NODES, false);

    ggml_tensor * mel_batch = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, max_chunk_len, n_mel, 1, n_chunks);
    ggml_set_name(mel_batch, "mel_batch");
    ggml_set_input(mel_batch);

    ggml_tensor * cur = ggml_conv_2d(ctx0, model_.conv2d1_w, mel_batch, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d1_b != nullptr) {
        ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d1_b, 1, 1, hp.audio_conv_channels, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu_erf(ctx0, cur);

    cur = ggml_conv_2d(ctx0, model_.conv2d2_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d2_b != nullptr) {
        ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d2_b, 1, 1, hp.audio_conv_channels, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu_erf(ctx0, cur);

    cur = ggml_conv_2d(ctx0, model_.conv2d3_w, cur, 2, 2, 1, 1, 1, 1);
    if (model_.conv2d3_b != nullptr) {
        ggml_tensor * bias = ggml_reshape_4d(ctx0, model_.conv2d3_b, 1, 1, hp.audio_conv_channels, 1);
        cur = ggml_add(ctx0, cur, bias);
    }
    cur = ggml_gelu_erf(ctx0, cur);

    const int64_t conv_out_w = cur->ne[0];
    const int64_t conv_out_h = cur->ne[1];
    const int64_t conv_out_c = cur->ne[2];
    const int64_t feat_dim = conv_out_c * conv_out_h;

    cur = ggml_reshape_3d(ctx0, cur, conv_out_w, feat_dim, n_chunks);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    cur = ggml_reshape_2d(ctx0, cur, feat_dim, conv_out_w * n_chunks);
    cur = ggml_mul_mat(ctx0, model_.conv_out_w, cur);
    if (model_.conv_out_b != nullptr) {
        cur = ggml_add(ctx0, cur, model_.conv_out_b);
    }
    cur = ggml_reshape_3d(ctx0, cur, n_state, conv_out_w, n_chunks);

    ggml_set_name(cur, "conv_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate aligner conv graph";
        ggml_free(ctx0);
        return false;
    }

    {
        const size_t batch_size = static_cast<size_t>(max_chunk_len) * n_mel * n_chunks;
        std::vector<float> mel_batch_data(batch_size, 0.0f);

        for (int32_t c = 0; c < n_chunks; ++c) {
            const int32_t chunk_len = chunk_lengths[static_cast<size_t>(c)];
            const int32_t start_frame = c * chunk_mel_size;
            for (int m = 0; m < n_mel; ++m) {
                for (int f = 0; f < chunk_len; ++f) {
                    const size_t idx =
                        static_cast<size_t>(f) +
                        static_cast<size_t>(m) * max_chunk_len +
                        static_cast<size_t>(c) * max_chunk_len * n_mel;
                    mel_batch_data[idx] = mel_data[static_cast<size_t>(m) * n_frames + start_frame + f];
                }
            }
        }

        ggml_tensor * mel_tensor = ggml_graph_get_tensor(gf, "mel_batch");
        ggml_backend_tensor_set(mel_tensor, mel_batch_data.data(), 0, batch_size * sizeof(float));
    }

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute aligner conv graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    ggml_tensor * conv_out_tensor = ggml_graph_get_tensor(gf, "conv_out");
    std::vector<float> conv_all(static_cast<size_t>(n_state) * conv_out_w * n_chunks);
    ggml_backend_tensor_get(conv_out_tensor, conv_all.data(), 0, conv_all.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);

    std::vector<float> pos_emb_data(static_cast<size_t>(max_out_w) * n_state);
    compute_sinusoidal_pe(pos_emb_data.data(), max_out_w, n_state);

    std::vector<float> hidden_flat(static_cast<size_t>(total_out_frames) * n_state);
    int32_t dst_offset = 0;
    for (int32_t c = 0; c < n_chunks; ++c) {
        const int32_t valid = chunk_out_lens[static_cast<size_t>(c)];
        for (int32_t t = 0; t < valid; ++t) {
            for (int32_t d = 0; d < n_state; ++d) {
                const float val = conv_all[static_cast<size_t>(d) + static_cast<size_t>(t) * n_state + static_cast<size_t>(c) * n_state * conv_out_w];
                hidden_flat[static_cast<size_t>(dst_offset + t) * n_state + d] = val + pos_emb_data[static_cast<size_t>(t) * n_state + d];
            }
        }
        dst_offset += valid;
    }

    const int32_t n_ctx = total_out_frames;
    const int32_t window_aftercnn = max_out_w * (Q3ASR_FA_AUDIO_N_WINDOW_INFER / chunk_mel_size);

    std::vector<int32_t> cu_seqlens;
    cu_seqlens.push_back(0);
    int32_t remaining = total_out_frames;
    while (remaining > 0) {
        if (remaining >= window_aftercnn) {
            cu_seqlens.push_back(cu_seqlens.back() + window_aftercnn);
            remaining -= window_aftercnn;
        } else {
            cu_seqlens.push_back(cu_seqlens.back() + remaining);
            remaining = 0;
        }
    }

    std::vector<float> attn_mask(static_cast<size_t>(n_ctx) * n_ctx, -INFINITY);
    for (size_t seg = 1; seg < cu_seqlens.size(); ++seg) {
        const int32_t seg_start = cu_seqlens[seg - 1];
        const int32_t seg_end = cu_seqlens[seg];
        for (int32_t r = seg_start; r < seg_end; ++r) {
            for (int32_t c = seg_start; c < seg_end; ++c) {
                attn_mask[static_cast<size_t>(r) * n_ctx + c] = 0.0f;
            }
        }
    }

    ctx0 = ggml_init(params);
    if (ctx0 == nullptr) {
        error_msg_ = "Failed to initialize the aligner encoder graph context";
        return false;
    }

    gf = ggml_new_graph_custom(ctx0, Q3ASR_FA_MAX_NODES, false);

    ggml_tensor * inp_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_state, n_ctx);
    ggml_set_name(inp_hidden, "inp_hidden");
    ggml_set_input(inp_hidden);

    ggml_tensor * mask_tensor = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_ctx, n_ctx);
    ggml_set_name(mask_tensor, "attn_mask");
    ggml_set_input(mask_tensor);

    ggml_tensor * inpL = inp_hidden;
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model_.encoder_layers[static_cast<size_t>(il)];

        cur = ggml_norm(ctx0, inpL, eps);
        if (layer.attn_norm_w != nullptr) {
            cur = ggml_mul(ctx0, cur, layer.attn_norm_w);
        }
        if (layer.attn_norm_b != nullptr) {
            cur = ggml_add(ctx0, cur, layer.attn_norm_b);
        }

        ggml_tensor * qcur = ggml_mul_mat(ctx0, layer.attn_q_w, cur);
        if (layer.attn_q_b != nullptr) {
            qcur = ggml_add(ctx0, qcur, layer.attn_q_b);
        }
        ggml_tensor * kcur = ggml_mul_mat(ctx0, layer.attn_k_w, cur);
        if (layer.attn_k_b != nullptr) {
            kcur = ggml_add(ctx0, kcur, layer.attn_k_b);
        }
        ggml_tensor * vcur = ggml_mul_mat(ctx0, layer.attn_v_w, cur);
        if (layer.attn_v_b != nullptr) {
            vcur = ggml_add(ctx0, vcur, layer.attn_v_b);
        }

        ggml_tensor * q = ggml_permute(ctx0, ggml_reshape_3d(ctx0, qcur, n_state_head, n_head, n_ctx), 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(ctx0, ggml_reshape_3d(ctx0, kcur, n_state_head, n_head, n_ctx), 0, 2, 1, 3);
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        ggml_tensor * kq_sm = ggml_soft_max_ext(ctx0, kq, mask_tensor, kq_scale, 0.0f);
        ggml_tensor * v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, vcur, n_state_head, n_head, n_ctx), 1, 2, 0, 3));
        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq_sm);
        ggml_tensor * merged = ggml_permute(ctx0, kqv, 0, 2, 1, 3);

        cur = ggml_cont_2d(ctx0, merged, n_state, n_ctx);
        cur = ggml_mul_mat(ctx0, layer.attn_out_w, cur);
        if (layer.attn_out_b != nullptr) {
            cur = ggml_add(ctx0, cur, layer.attn_out_b);
        }
        cur = ggml_add(ctx0, cur, inpL);

        ggml_tensor * inp_ff = cur;
        cur = ggml_norm(ctx0, inp_ff, eps);
        if (layer.ffn_norm_w != nullptr) {
            cur = ggml_mul(ctx0, cur, layer.ffn_norm_w);
        }
        if (layer.ffn_norm_b != nullptr) {
            cur = ggml_add(ctx0, cur, layer.ffn_norm_b);
        }

        cur = ggml_mul_mat(ctx0, layer.ffn_up_w, cur);
        if (layer.ffn_up_b != nullptr) {
            cur = ggml_add(ctx0, cur, layer.ffn_up_b);
        }
        cur = ggml_gelu_erf(ctx0, cur);
        cur = ggml_mul_mat(ctx0, layer.ffn_down_w, cur);
        if (layer.ffn_down_b != nullptr) {
            cur = ggml_add(ctx0, cur, layer.ffn_down_b);
        }

        inpL = ggml_add(ctx0, cur, inp_ff);
    }

    cur = inpL;
    cur = ggml_norm(ctx0, cur, eps);
    if (model_.ln_post_w != nullptr) {
        cur = ggml_mul(ctx0, cur, model_.ln_post_w);
    }
    if (model_.ln_post_b != nullptr) {
        cur = ggml_add(ctx0, cur, model_.ln_post_b);
    }

    cur = ggml_mul_mat(ctx0, model_.proj1_w, cur);
    if (model_.proj1_b != nullptr) {
        cur = ggml_add(ctx0, cur, model_.proj1_b);
    }
    cur = ggml_gelu_erf(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model_.proj2_w, cur);
    if (model_.proj2_b != nullptr) {
        cur = ggml_add(ctx0, cur, model_.proj2_b);
    }

    ggml_set_name(cur, "audio_enc_out");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate aligner encoder graph";
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inp_hidden"), hidden_flat.data(), 0, hidden_flat.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "attn_mask"), attn_mask.data(), 0, attn_mask.size() * sizeof(float));

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute aligner encoder graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    ggml_tensor * audio_out = ggml_graph_get_tensor(gf, "audio_enc_out");
    if (audio_out == nullptr) {
        error_msg_ = "Failed to find aligner encoder output tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    output.resize(static_cast<size_t>(audio_out->ne[0]) * audio_out->ne[1]);
    ggml_backend_tensor_get(audio_out, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);

    return true;
}

bool ForcedAligner::forward_decoder(
    const int32_t * tokens,
    int32_t n_tokens,
    const float * audio_embd,
    int32_t n_audio,
    int32_t audio_start_pos,
    std::vector<float> & output
) {
    if (model_.ctx == nullptr) {
        error_msg_ = "Forced aligner model is not loaded";
        return false;
    }

    const auto & hp = model_.hparams;
    const int n_head = hp.text_attention_heads;
    const int n_kv_head = hp.text_kv_heads;
    const int head_dim = hp.text_head_dim;
    const int hidden_size = hp.text_hidden_size;
    const float eps = hp.text_rms_norm_eps;
    const float rope_theta = hp.text_rope_theta;
    const float kq_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    ggml_init_params params = {
        /*.mem_size   =*/ state_.compute_meta.size(),
        /*.mem_buffer =*/ state_.compute_meta.data(),
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx0 = ggml_init(params);
    if (ctx0 == nullptr) {
        error_msg_ = "Failed to initialize the aligner decoder graph context";
        return false;
    }

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, Q3ASR_FA_MAX_NODES, false);

    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_pos, "inp_pos");
    ggml_set_input(inp_pos);

    ggml_tensor * causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F16, n_tokens, n_tokens);
    ggml_set_name(causal_mask, "causal_mask");
    ggml_set_input(causal_mask);

    ggml_tensor * inp_audio = nullptr;
    if (audio_embd != nullptr && n_audio > 0) {
        inp_audio = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hidden_size, n_audio);
        ggml_set_name(inp_audio, "inp_audio");
        ggml_set_input(inp_audio);
    }

    ggml_tensor * cur = ggml_get_rows(ctx0, model_.token_embd, inp_tokens);

    if (inp_audio != nullptr && audio_start_pos >= 0 && audio_start_pos + n_audio <= n_tokens) {
        ggml_tensor * before = nullptr;
        ggml_tensor * after = nullptr;

        if (audio_start_pos > 0) {
            before = ggml_view_2d(ctx0, cur, hidden_size, audio_start_pos, cur->nb[1], 0);
        }
        if (audio_start_pos + n_audio < n_tokens) {
            const int after_start = audio_start_pos + n_audio;
            const int after_len = n_tokens - after_start;
            after = ggml_view_2d(ctx0, cur, hidden_size, after_len, cur->nb[1], static_cast<size_t>(after_start) * cur->nb[1]);
        }

        if (before != nullptr && after != nullptr) {
            cur = ggml_concat(ctx0, ggml_concat(ctx0, before, inp_audio, 1), after, 1);
        } else if (before != nullptr) {
            cur = ggml_concat(ctx0, before, inp_audio, 1);
        } else if (after != nullptr) {
            cur = ggml_concat(ctx0, inp_audio, after, 1);
        } else {
            cur = inp_audio;
        }
    }

    ggml_tensor * inpL = cur;
    for (int il = 0; il < hp.text_decoder_layers; ++il) {
        const auto & layer = model_.decoder_layers[static_cast<size_t>(il)];

        cur = ggml_rms_norm(ctx0, inpL, eps);
        cur = ggml_mul(ctx0, cur, layer.attn_norm);

        ggml_tensor * qcur = ggml_mul_mat(ctx0, layer.attn_q, cur);
        ggml_tensor * kcur = ggml_mul_mat(ctx0, layer.attn_k, cur);
        ggml_tensor * vcur = ggml_mul_mat(ctx0, layer.attn_v, cur);

        qcur = ggml_reshape_3d(ctx0, qcur, head_dim, n_head, n_tokens);
        kcur = ggml_reshape_3d(ctx0, kcur, head_dim, n_kv_head, n_tokens);
        vcur = ggml_reshape_3d(ctx0, vcur, head_dim, n_kv_head, n_tokens);

        qcur = ggml_rms_norm(ctx0, qcur, eps);
        qcur = ggml_mul(ctx0, qcur, layer.attn_q_norm);
        kcur = ggml_rms_norm(ctx0, kcur, eps);
        kcur = ggml_mul(ctx0, kcur, layer.attn_k_norm);

        qcur = ggml_rope_ext(
            ctx0, qcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
            rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        );
        kcur = ggml_rope_ext(
            ctx0, kcur, inp_pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, 0,
            rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f
        );

        ggml_tensor * qfa = ggml_permute(ctx0, qcur, 0, 2, 1, 3);
        ggml_tensor * kfa = ggml_permute(ctx0, kcur, 0, 2, 1, 3);
        ggml_tensor * vfa = ggml_cast(ctx0, ggml_permute(ctx0, vcur, 0, 2, 1, 3), GGML_TYPE_F16);

        cur = ggml_flash_attn_ext(ctx0, qfa, kfa, vfa, causal_mask, kq_scale, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);
        cur = ggml_reshape_2d(ctx0, cur, n_head * head_dim, n_tokens);

        cur = ggml_mul_mat(ctx0, layer.attn_output, cur);
        cur = ggml_add(ctx0, cur, inpL);

        ggml_tensor * inp_ff = cur;
        cur = ggml_rms_norm(ctx0, inp_ff, eps);
        cur = ggml_mul(ctx0, cur, layer.ffn_norm);

        ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ffn_gate, cur);
        ggml_tensor * up = ggml_mul_mat(ctx0, layer.ffn_up, cur);
        gate = ggml_silu(ctx0, gate);
        cur = ggml_mul(ctx0, gate, up);
        cur = ggml_mul_mat(ctx0, layer.ffn_down, cur);

        inpL = ggml_add(ctx0, cur, inp_ff);
    }

    cur = ggml_rms_norm(ctx0, inpL, eps);
    cur = ggml_mul(ctx0, cur, model_.output_norm);
    cur = ggml_mul_mat(ctx0, model_.classify_head_w, cur);
    if (model_.classify_head_b != nullptr) {
        cur = ggml_add(ctx0, cur, model_.classify_head_b);
    }

    ggml_set_name(cur, "logits");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    if (!ggml_backend_sched_alloc_graph(state_.sched, gf)) {
        error_msg_ = "Failed to allocate aligner decoder graph";
        ggml_free(ctx0);
        return false;
    }

    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inp_tokens"), tokens, 0, static_cast<size_t>(n_tokens) * sizeof(int32_t));

    std::vector<int32_t> positions(static_cast<size_t>(n_tokens));
    for (int i = 0; i < n_tokens; ++i) {
        positions[static_cast<size_t>(i)] = i;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "inp_pos"), positions.data(), 0, positions.size() * sizeof(int32_t));

    {
        std::vector<ggml_fp16_t> mask(static_cast<size_t>(n_tokens) * n_tokens);
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
        for (int q = 0; q < n_tokens; ++q) {
            for (int k = 0; k < n_tokens; ++k) {
                mask[static_cast<size_t>(k) + static_cast<size_t>(q) * n_tokens] = k <= q ? zero : neg_inf;
            }
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "causal_mask"), mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    }

    if (audio_embd != nullptr && n_audio > 0) {
        ggml_backend_tensor_set(
            ggml_graph_get_tensor(gf, "inp_audio"),
            audio_embd,
            0,
            static_cast<size_t>(n_audio) * hidden_size * sizeof(float)
        );
    }

    if (ggml_backend_sched_graph_compute(state_.sched, gf) != GGML_STATUS_SUCCESS) {
        error_msg_ = "Failed to compute aligner decoder graph";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    ggml_tensor * logits = ggml_graph_get_tensor(gf, "logits");
    if (logits == nullptr) {
        error_msg_ = "Failed to find aligner logits tensor";
        ggml_backend_sched_reset(state_.sched);
        ggml_free(ctx0);
        return false;
    }

    const int64_t n_classes = logits->ne[0];
    const int64_t n_out_tokens = logits->ne[1];
    output.resize(static_cast<size_t>(n_classes) * n_out_tokens);
    ggml_backend_tensor_get(logits, output.data(), 0, output.size() * sizeof(float));

    ggml_backend_sched_reset(state_.sched);
    ggml_free(ctx0);

    return true;
}

std::vector<int32_t> ForcedAligner::fix_timestamp_classes(const std::vector<int32_t> & data) const {
    const int n = static_cast<int>(data.size());
    if (n == 0) {
        return {};
    }

    std::vector<int> dp(static_cast<size_t>(n), 1);
    std::vector<int> parent(static_cast<size_t>(n), -1);

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (data[static_cast<size_t>(j)] <= data[static_cast<size_t>(i)] &&
                dp[static_cast<size_t>(j)] + 1 > dp[static_cast<size_t>(i)]) {
                dp[static_cast<size_t>(i)] = dp[static_cast<size_t>(j)] + 1;
                parent[static_cast<size_t>(i)] = j;
            }
        }
    }

    int max_len = 0;
    int max_idx = 0;
    for (int i = 0; i < n; ++i) {
        if (dp[static_cast<size_t>(i)] > max_len) {
            max_len = dp[static_cast<size_t>(i)];
            max_idx = i;
        }
    }

    std::vector<bool> is_normal(static_cast<size_t>(n), false);
    for (int idx = max_idx; idx != -1; idx = parent[static_cast<size_t>(idx)]) {
        is_normal[static_cast<size_t>(idx)] = true;
    }

    std::vector<int32_t> result = data;
    int i = 0;
    while (i < n) {
        if (!is_normal[static_cast<size_t>(i)]) {
            int j = i;
            while (j < n && !is_normal[static_cast<size_t>(j)]) {
                ++j;
            }

            const int anomaly_count = j - i;
            int32_t left_val = -1;
            int32_t right_val = -1;

            for (int k = i - 1; k >= 0; --k) {
                if (is_normal[static_cast<size_t>(k)]) {
                    left_val = result[static_cast<size_t>(k)];
                    break;
                }
            }
            for (int k = j; k < n; ++k) {
                if (is_normal[static_cast<size_t>(k)]) {
                    right_val = result[static_cast<size_t>(k)];
                    break;
                }
            }

            if (anomaly_count <= 2) {
                for (int k = i; k < j; ++k) {
                    if (left_val < 0) {
                        result[static_cast<size_t>(k)] = right_val;
                    } else if (right_val < 0) {
                        result[static_cast<size_t>(k)] = left_val;
                    } else {
                        result[static_cast<size_t>(k)] = ((k - (i - 1)) <= (j - k)) ? left_val : right_val;
                    }
                }
            } else {
                if (left_val >= 0 && right_val >= 0) {
                    const float step = static_cast<float>(right_val - left_val) / (anomaly_count + 1);
                    for (int k = i; k < j; ++k) {
                        result[static_cast<size_t>(k)] = static_cast<int32_t>(left_val + step * (k - i + 1));
                    }
                } else if (left_val >= 0) {
                    for (int k = i; k < j; ++k) {
                        result[static_cast<size_t>(k)] = left_val;
                    }
                } else if (right_val >= 0) {
                    for (int k = i; k < j; ++k) {
                        result[static_cast<size_t>(k)] = right_val;
                    }
                }
            }

            i = j;
        } else {
            ++i;
        }
    }

    return result;
}

std::vector<float> ForcedAligner::classes_to_timestamps(const std::vector<int32_t> & classes) const {
    std::vector<float> timestamps;
    timestamps.reserve(classes.size());
    const float segment_time_sec = static_cast<float>(model_.hparams.timestamp_segment_time_ms) / 1000.0f;
    for (int32_t cls : classes) {
        timestamps.push_back(cls * segment_time_sec);
    }
    return timestamps;
}

std::vector<int32_t> ForcedAligner::extract_timestamp_classes(
    const std::vector<float> & logits,
    const std::vector<int32_t> & tokens
) const {
    const int32_t n_classes = model_.hparams.classify_num;
    std::vector<int32_t> classes;

    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] != model_.hparams.timestamp_token_id) {
            continue;
        }

        const float * logit_ptr = logits.data() + i * static_cast<size_t>(n_classes);
        int32_t best_class = 0;
        float best_score = logit_ptr[0];
        for (int32_t c = 1; c < n_classes; ++c) {
            if (logit_ptr[c] > best_score) {
                best_score = logit_ptr[c];
                best_class = c;
            }
        }
        classes.push_back(best_class);
    }

    return classes;
}

std::vector<int32_t> ForcedAligner::build_input_tokens(const std::vector<int32_t> & text_tokens, int32_t n_audio_frames) const {
    const auto & hp = model_.hparams;
    std::vector<int32_t> tokens;
    tokens.reserve(static_cast<size_t>(n_audio_frames) + text_tokens.size() + 2);

    tokens.push_back(hp.audio_start_token_id);
    tokens.insert(tokens.end(), static_cast<size_t>(n_audio_frames), hp.audio_pad_token_id);
    tokens.push_back(hp.audio_end_token_id);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());

    return tokens;
}

int32_t ForcedAligner::find_audio_start_pos(const std::vector<int32_t> & tokens) const {
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (tokens[i] == model_.hparams.audio_start_token_id) {
            return static_cast<int32_t>(i + 1);
        }
    }
    return -1;
}

std::vector<std::string> ForcedAligner::normalize_alignment_words(
    const std::string & text,
    const std::string & language
) const {
    const std::string lang = to_lower_ascii(language);
    if (lang == "japanese") {
        return tokenize_japanese_fallback(text);
    }
    if (lang == "korean" && !model_.ko_dict.empty()) {
        return tokenize_korean_dict(text, model_.ko_dict);
    }
    return tokenize_space_lang(text);
}

std::vector<normalized_word_span> ForcedAligner::normalize_with_spans(
    const std::string & text,
    const std::string & language
) const {
    const std::string lang = to_lower_ascii(language);
    if (lang == "japanese") {
        return normalize_japanese_with_spans(text);
    }
    if (lang == "korean" && !model_.ko_dict.empty()) {
        return normalize_korean_with_spans(text, model_.ko_dict);
    }
    return normalize_space_lang_with_spans(text);
}

std::vector<int32_t> ForcedAligner::encode_words_with_timestamps(const std::vector<std::string> & words) const {
    std::vector<int32_t> tokens;

    for (const std::string & word : words) {
        if (word.empty()) {
            continue;
        }

        const std::string bpe_str = bytes_to_bpe_string(word);
        const auto subwords = bpe_encode_word(bpe_str, model_.bpe_ranks);

        for (const std::string & subword : subwords) {
            const auto it = model_.token_to_id.find(subword);
            if (it == model_.token_to_id.end()) {
                continue;
            }
            tokens.push_back(it->second);
        }

        tokens.push_back(model_.hparams.timestamp_token_id);
        tokens.push_back(model_.hparams.timestamp_token_id);
    }

    return tokens;
}

std::vector<int32_t> ForcedAligner::tokenize_with_timestamps(
    const std::string & text,
    std::vector<std::string> & words,
    const std::string & language
) const {
    words = normalize_alignment_words(text, language);
    return encode_words_with_timestamps(words);
}

alignment_result ForcedAligner::align(const std::string & audio_path, const std::string & text, const std::string & language) {
    return align(audio_path, text, language, align_runtime_params{});
}

alignment_result ForcedAligner::align(
    const std::string & audio_path,
    const std::string & text,
    const std::string & language,
    const align_runtime_params & runtime_params
) {
    alignment_result result;

    if (!model_loaded_) {
        result.error_msg = "Forced aligner model is not loaded";
        return result;
    }

    std::vector<float> samples;
    int sample_rate = 0;
    if (!load_wav(audio_path, samples, sample_rate)) {
        result.error_msg = "Failed to load WAV file";
        return result;
    }
    if (sample_rate != QWEN_SAMPLE_RATE) {
        result.error_msg = "Only 16 kHz WAV input is supported";
        return result;
    }

    return align(samples.data(), static_cast<int>(samples.size()), text, language, runtime_params);
}

alignment_result ForcedAligner::align(const float * samples, int n_samples, const std::string & text, const std::string & language) {
    return align(samples, n_samples, text, language, align_runtime_params{});
}

alignment_result ForcedAligner::align(
    const float * samples,
    int n_samples,
    const std::string & text,
    const std::string & language,
    const align_runtime_params & runtime_params
) {
    alignment_result result;

    if (!model_loaded_) {
        result.error_msg = "Forced aligner model is not loaded";
        return result;
    }
    if (samples == nullptr || n_samples <= 0) {
        result.error_msg = "No audio samples provided for alignment";
        return result;
    }

    const std::vector<std::string> words = normalize_alignment_words(text, language);
    if (words.empty()) {
        result.error_msg = "Forced aligner input text produced no alignable units";
        return result;
    }

    const float audio_duration = static_cast<float>(n_samples) / QWEN_SAMPLE_RATE;
    const float max_chunk_seconds = runtime_params.max_chunk_seconds;
    if (max_chunk_seconds > 0.0f && audio_duration > max_chunk_seconds) {
        return align_chunked(samples, n_samples, words, runtime_params);
    }

    return align_words(samples, n_samples, words);
}

alignment_result ForcedAligner::align_words(const float * samples, int n_samples, const std::vector<std::string> & words) {
    alignment_result result;
    const int64_t t_total_start = now_ms();
    const float audio_duration = static_cast<float>(n_samples) / QWEN_SAMPLE_RATE;

    MelFilters mel_filters;
    const int64_t t_mel_start = now_ms();
    generate_mel_filters(mel_filters, QWEN_N_MELS, QWEN_N_FFT, QWEN_SAMPLE_RATE);

    MelSpectrogram mel;
    if (!log_mel_spectrogram(samples, n_samples, mel_filters, mel, std::max(1, params_.n_threads))) {
        result.error_msg = "Failed to compute the Qwen3-ASR mel spectrogram";
        return result;
    }
    result.t_mel_ms = now_ms() - t_mel_start;

    std::vector<float> audio_features;
    const int64_t t_encode_start = now_ms();
    if (!encode_audio(mel.data.data(), mel.n_mel, mel.n_len, audio_features)) {
        result.error_msg = error_msg_;
        return result;
    }
    result.t_encode_ms = now_ms() - t_encode_start;

    const int32_t n_audio_frames = static_cast<int32_t>(audio_features.size() / static_cast<size_t>(model_.hparams.text_hidden_size));
    const int32_t n_audio_pads = get_feat_extract_output_lengths(mel.n_len);

    std::vector<int32_t> text_tokens = encode_words_with_timestamps(words);
    std::vector<int32_t> input_tokens = build_input_tokens(text_tokens, n_audio_pads);
    const int32_t audio_start_pos = find_audio_start_pos(input_tokens);

    std::vector<float> logits;
    const int64_t t_decode_start = now_ms();
    if (!forward_decoder(input_tokens.data(), static_cast<int32_t>(input_tokens.size()), audio_features.data(), n_audio_frames, audio_start_pos, logits)) {
        result.error_msg = error_msg_;
        return result;
    }
    result.t_decode_ms = now_ms() - t_decode_start;

    std::vector<int32_t> timestamp_classes = extract_timestamp_classes(logits, input_tokens);
    timestamp_classes = fix_timestamp_classes(timestamp_classes);
    std::vector<float> timestamps = classes_to_timestamps(timestamp_classes);

    for (float & ts : timestamps) {
        if (ts > audio_duration) {
            ts = audio_duration;
        }
    }

    result.items.reserve(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
        aligned_item item;
        item.text = words[i];

        const size_t start_idx = i * 2;
        const size_t end_idx = i * 2 + 1;
        item.start_time = start_idx < timestamps.size() ? timestamps[start_idx] : 0.0f;
        item.end_time = end_idx < timestamps.size() ? timestamps[end_idx] : audio_duration;

        result.items.push_back(std::move(item));
    }

    result.success = true;
    result.t_total_ms = now_ms() - t_total_start;
    return result;
}

alignment_result ForcedAligner::align_chunked(
    const float * samples,
    int n_samples,
    const std::vector<std::string> & words,
    const align_runtime_params & runtime_params
) {
    alignment_result result;
    const int64_t t_total_start = now_ms();

    const std::vector<split_audio_chunk> chunks = split_audio_into_chunks(samples, n_samples, QWEN_SAMPLE_RATE, runtime_params);
    if (chunks.empty()) {
        result.error_msg = "Failed to split audio for chunked alignment";
        return result;
    }

    const std::vector<std::pair<size_t, size_t>> word_ranges = assign_word_ranges(chunks, words.size(), n_samples);
    result.items.reserve(words.size());

    for (size_t i = 0; i < chunks.size(); ++i) {
        const auto [begin, end] = word_ranges[i];
        if (begin >= end || begin >= words.size()) {
            continue;
        }

        const std::vector<std::string> chunk_words(words.begin() + static_cast<std::ptrdiff_t>(begin), words.begin() + static_cast<std::ptrdiff_t>(end));
        alignment_result chunk_result = align_words(
            chunks[i].samples.data(),
            static_cast<int>(chunks[i].samples.size()),
            chunk_words
        );
        if (!chunk_result.success) {
            chunk_result.error_msg =
                "Chunked forced alignment failed for chunk " + std::to_string(i + 1) +
                "/" + std::to_string(chunks.size()) + ": " + chunk_result.error_msg;
            return chunk_result;
        }

        result.t_mel_ms += chunk_result.t_mel_ms;
        result.t_encode_ms += chunk_result.t_encode_ms;
        result.t_decode_ms += chunk_result.t_decode_ms;

        for (aligned_item & item : chunk_result.items) {
            item.start_time = round_millis(std::min(
                item.start_time + chunks[i].offset_sec,
                static_cast<float>(n_samples) / QWEN_SAMPLE_RATE
            ));
            item.end_time = round_millis(std::min(
                item.end_time + chunks[i].offset_sec,
                static_cast<float>(n_samples) / QWEN_SAMPLE_RATE
            ));
            result.items.push_back(std::move(item));
        }
    }

    result.success = true;
    result.t_total_ms = now_ms() - t_total_start;
    return result;
}

void free_forced_aligner_model(forced_aligner_model & model) {
    if (model.buffer != nullptr) {
        ggml_backend_buffer_free(model.buffer);
        model.buffer = nullptr;
    }
    if (model.ctx != nullptr) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.mmap_addr != nullptr) {
        munmap(model.mmap_addr, model.mmap_size);
        model.mmap_addr = nullptr;
        model.mmap_size = 0;
    }

    model.encoder_layers.clear();
    model.decoder_layers.clear();
    model.tensors.clear();
    model.vocab.clear();
    model.bpe_ranks.clear();
    model.token_to_id.clear();
    model.ko_dict.clear();
    model = {};
}

} // namespace q3asr
