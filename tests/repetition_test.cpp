#include "repetition.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect_fix(const std::string & in, const std::string & want) {
    const std::string got = q3asr::detect_and_fix_repetitions(in, 20);
    if (got != want) {
        std::cerr << "detect_and_fix_repetitions mismatch for: " << in << "\n  got:  " << got << "\n  want: " << want << "\n";
        ++failures;
    }
}

std::vector<int32_t> repeat(const std::vector<int32_t> & block, int times, std::vector<int32_t> head = {}) {
    for (int i = 0; i < times; ++i) {
        head.insert(head.end(), block.begin(), block.end());
    }
    return head;
}

void expect_keep(const std::vector<int32_t> & tokens, int repeats, int min_tokens, size_t want) {
    const size_t got = q3asr::loop_stop_keep(tokens, repeats, min_tokens);
    if (got != want) {
        std::cerr << "loop_stop_keep: got " << got << ", want " << want << " (n=" << tokens.size() << ")\n";
        ++failures;
    }
}

} // namespace

int main() {
    // Expected values come from transformers' Qwen3ASRProcessor _detect_and_fix_repetitions (threshold 20).
    expect_fix(std::string(25, 'a'), "a");
    expect_fix(std::string(20, 'a'), std::string(20, 'a'));
    std::string ab21;
    for (int i = 0; i < 21; ++i) ab21 += "ab";
    expect_fix(ab21 + "c", "abc");
    expect_fix(ab21.substr(4), ab21.substr(4));
    std::string thai = "x";
    for (int i = 0; i < 22; ++i) thai += "ครับ ";
    expect_fix(thai + "ค่ะ", "xครับ ค่ะ");
    std::string zh;
    for (int i = 0; i < 30; ++i) zh += "你好";
    expect_fix(zh + "吗", "你好吗");
    expect_fix("hello world", "hello world");
    if (q3asr::detect_and_fix_repetitions(std::string(25, 'a'), 0) != std::string(25, 'a')) {
        std::cerr << "threshold 0 must leave the text unchanged\n";
        ++failures;
    }

    const std::vector<int32_t> six = {1, 2, 3, 4, 5, 6};
    expect_keep(repeat(six, 4), 4, 24, 6);
    expect_keep(repeat(six, 4, {7, 8, 9}), 4, 24, 9);
    expect_keep(repeat(six, 3, {7, 8, 9}), 4, 24, 21);
    expect_keep(repeat({1, 2}, 4), 4, 24, 8);
    expect_keep(repeat({1, 2}, 12), 4, 24, 2);
    expect_keep(repeat(six, 4), 0, 0, 24);

    if (failures == 0) {
        std::cout << "repetition tests passed\n";
    }
    return failures == 0 ? 0 : 1;
}
