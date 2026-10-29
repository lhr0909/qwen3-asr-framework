#ifndef Q3ASR_REPETITION_H
#define Q3ASR_REPETITION_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace q3asr {

// Port of Qwen3-ASR's detect_and_fix_repetitions (qwen_asr/inference/utils.py @c17a131), which the Python reference
// and transformers' Qwen3ASRProcessor apply to every decoded output. It works on code points like Python's str.

inline std::u32string utf8_to_u32(const std::string & s) {
    std::u32string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        int len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : (c >> 3) == 0x1E ? 4 : 0;
        if (len == 0 || i + len > s.size()) {
            out.push_back(0xFFFD);
            ++i;
            continue;
        }
        char32_t cp = len == 1 ? c : c & (0xFF >> (len + 1));
        for (int j = 1; j < len; ++j) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[i + j]) & 0x3F);
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

inline std::string u32_to_utf8(const std::u32string & s) {
    std::string out;
    out.reserve(s.size() * 3);
    for (char32_t cp : s) {
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

inline std::u32string fix_char_repeats(const std::u32string & s, int thresh) {
    std::u32string res;
    res.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        size_t count = 1;
        while (i + count < s.size() && s[i + count] == s[i]) {
            ++count;
        }
        res.append(s, i, count > static_cast<size_t>(thresh) ? 1 : count);
        i += count;
    }
    return res;
}

// The reference recurses on the remainder after each collapsed pattern; this loops instead.
inline std::u32string fix_pattern_repeats(const std::u32string & s, int thresh, int max_len = 20) {
    const size_t t = static_cast<size_t>(thresh);
    std::u32string result;
    size_t pos = 0;
    while (true) {
        const size_t n = s.size() - pos;
        if (n < t * 2) {
            result.append(s, pos, std::u32string::npos);
            return result;
        }
        size_t i = 0;
        bool found = false;
        while (i <= n - t * 2) {
            for (size_t k = 1; k <= static_cast<size_t>(max_len); ++k) {
                if (i + k * t > n) {
                    break;
                }
                const size_t base = pos + i;
                bool valid = true;
                for (size_t rep = 1; rep < t && valid; ++rep) {
                    valid = s.compare(base + rep * k, k, s, base, k) == 0;
                }
                if (valid) {
                    size_t end_index = i + t * k;
                    while (end_index + k <= n && s.compare(pos + end_index, k, s, base, k) == 0) {
                        end_index += k;
                    }
                    result.append(s, base, k);
                    pos += end_index;
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
            result.push_back(s[pos + i]);
            ++i;
        }
        if (!found) {
            result.append(s, pos + i, std::u32string::npos);
            return result;
        }
    }
}

inline std::string detect_and_fix_repetitions(const std::string & text, int threshold = 20) {
    if (threshold <= 0) {
        return text;
    }
    return u32_to_utf8(fix_pattern_repeats(fix_char_repeats(utf8_to_u32(text), threshold), threshold));
}

// Tokens to keep when `tokens` ends in a block of up to 64 tokens repeated at least min_repeats times over at least
// min_tokens tokens: everything up to and including the first copy. Otherwise tokens.size().
inline size_t loop_stop_keep(const std::vector<int32_t> & tokens, int min_repeats, int min_tokens) {
    const size_t n = tokens.size();
    if (min_repeats < 2 || n < static_cast<size_t>(min_repeats)) {
        return n;
    }
    for (size_t period = 1; period <= 64 && period * static_cast<size_t>(min_repeats) <= n; ++period) {
        size_t repeats = 1;
        while ((repeats + 1) * period <= n &&
               std::equal(tokens.end() - static_cast<std::ptrdiff_t>(period), tokens.end(),
                          tokens.end() - static_cast<std::ptrdiff_t>((repeats + 1) * period))) {
            ++repeats;
        }
        if (repeats >= static_cast<size_t>(min_repeats) && repeats * period >= static_cast<size_t>(min_tokens)) {
            return n - (repeats - 1) * period;
        }
    }
    return n;
}

} // namespace q3asr

#endif
