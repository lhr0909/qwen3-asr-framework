#ifndef Q3ASR_H
#define Q3ASR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q3asr_context q3asr_context;

typedef struct {
    const char * text_model_path;
    const char * mmproj_model_path;
    int use_gpu;
    int n_threads;
    int n_batch;
    int n_ctx;
    int n_gpu_layers;
} q3asr_context_params;

typedef struct {
    const char * language_hint;
    int max_tokens;
} q3asr_transcribe_params;

typedef struct {
    char * raw_text;
    char * language;
    char * text;
} q3asr_transcribe_result;

q3asr_context_params q3asr_context_default_params(void);
q3asr_transcribe_params q3asr_transcribe_default_params(void);

q3asr_context * q3asr_context_create(const q3asr_context_params * params);
void q3asr_context_destroy(q3asr_context * ctx);
const char * q3asr_context_last_error(const q3asr_context * ctx);

int q3asr_transcribe_wav_file(
    q3asr_context * ctx,
    const char * wav_path,
    const q3asr_transcribe_params * params,
    q3asr_transcribe_result * out_result
);

int q3asr_transcribe_pcm_f32(
    q3asr_context * ctx,
    const float * samples,
    int n_samples,
    const q3asr_transcribe_params * params,
    q3asr_transcribe_result * out_result
);

void q3asr_transcribe_result_clear(q3asr_transcribe_result * result);

#ifdef __cplusplus
}
#endif

#endif
