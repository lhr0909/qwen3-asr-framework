#pragma once

#include "gguf_loader.h"

#include <string>
#include <vector>

namespace q3asr {

struct audio_encoder_state {
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> compute_meta;
};

class AudioEncoder {
public:
    AudioEncoder() = default;
    ~AudioEncoder();

    bool load_model(const std::string & model_path, bool use_gpu, int n_threads);

    bool encode(
        const float * mel_data,
        int n_mel,
        int n_frames,
        std::vector<float> & output
    );

    const audio_encoder_config & config() const { return model_.config; }
    const std::string & get_error() const { return error_msg_; }

private:
    bool encode_window(
        const float * mel_window,
        int n_frames,
        std::vector<float> & output
    );

    audio_encoder_model model_;
    audio_encoder_state state_;
    bool use_gpu_ = true;
    int n_threads_ = 4;
    std::string error_msg_;
};

} // namespace q3asr
