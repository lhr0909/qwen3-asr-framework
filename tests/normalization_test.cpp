#include "forced_aligner.h"

#include <iostream>
#include <string>
#include <vector>

namespace {

bool spans_monotonic(const std::vector<q3asr::normalized_word_span> & spans) {
    size_t last_end = 0;
    for (const auto & span : spans) {
        if (span.text.empty()) {
            return false;
        }
        if (span.byte_start > span.byte_end) {
            return false;
        }
        if (!spans.empty() && span.byte_start < last_end) {
            return false;
        }
        last_end = span.byte_end;
    }
    return true;
}

bool expect_tokens(
    const std::vector<q3asr::normalized_word_span> & spans,
    const std::vector<std::string> & expected
) {
    if (spans.size() != expected.size()) {
        return false;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        if (spans[i].text != expected[i]) {
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    q3asr::ForcedAligner aligner;
    bool ok = true;

    const auto english = aligner.normalize_with_spans(
        "You can apparently promote on Sundays on /r/apple on Reddit.",
        "English"
    );
    ok = ok && spans_monotonic(english) && expect_tokens(
        english,
        {"You", "can", "apparently", "promote", "on", "Sundays", "on", "rapple", "on", "Reddit"}
    );

    const auto chinese = aligner.normalize_with_spans(
        "甚至出现交易几乎停滞的情况。",
        "Chinese"
    );
    ok = ok && spans_monotonic(chinese) && expect_tokens(
        chinese,
        {"甚", "至", "出", "现", "交", "易", "几", "乎", "停", "滞", "的", "情", "况"}
    );

    const auto thai = aligner.normalize_with_spans(
        "ปีมะรุมเนี่ยผ่านวันดีวันศุกร์ก็เป็นวันที่แฮปปี้แอนเจรแทนที่สุดในวันทั้งเจ็ดถ้าคิดว่าชีวิตน้องก็มีอุปสรรคเยอะแยะมากมายเครียดอยู่นะ",
        "Thai"
    );
    ok = ok && spans_monotonic(thai);
    ok = ok && thai.size() > 1;
    ok = ok && !thai.empty() && thai.front().text == "ปี";

    if (!ok) {
        std::cerr << "English span count: " << english.size() << "\n";
        std::cerr << "Chinese span count: " << chinese.size() << "\n";
        std::cerr << "Thai span count: " << thai.size() << "\n";
        if (!thai.empty()) {
            std::cerr << "Thai first span: " << thai.front().text << "\n";
        }
        return 1;
    }

    std::cout << "english=" << english.size() << "\n";
    std::cout << "chinese=" << chinese.size() << "\n";
    std::cout << "thai=" << thai.size() << "\n";
    return 0;
}
