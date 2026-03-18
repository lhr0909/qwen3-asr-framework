#ifndef Q3ASR_H
#define Q3ASR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct q3asr_context q3asr_context;
typedef struct q3asr_aligner_context q3asr_aligner_context;
typedef void (*q3asr_raw_stream_callback)(const char * raw_text, void * user_data);
typedef void (*q3asr_progress_callback)(
    const char * language,
    const char * committed_text,
    const char * partial_text,
    int chunk_index,
    int chunk_count,
    void * user_data
);

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
    const char * aligner_model_path;
    const char * korean_dict_path;
    int use_gpu;
    int n_threads;
} q3asr_aligner_context_params;

typedef struct {
    const char * language_hint;
    const char * context;
    int max_tokens;
    float temperature;
    q3asr_aligner_context * aligner_context;
    float max_audio_chunk_seconds;
    float audio_chunk_overlap_seconds;
    q3asr_raw_stream_callback raw_text_callback;
    void * raw_text_callback_user_data;
    q3asr_progress_callback progress_callback;
    void * progress_callback_user_data;
} q3asr_transcribe_params;

typedef struct {
    char * raw_text;
    char * language;
    char * text;
} q3asr_transcribe_result;

typedef struct {
    char * text;
    float start_time;
    float end_time;
} q3asr_aligned_item;

typedef struct {
    q3asr_aligned_item * items;
    size_t n_items;
} q3asr_alignment_result;

typedef struct {
    float max_chunk_seconds;
    float chunk_search_expand_seconds;
    float min_chunk_window_ms;
} q3asr_align_params;

q3asr_context_params q3asr_context_default_params(void);
q3asr_aligner_context_params q3asr_aligner_context_default_params(void);
q3asr_transcribe_params q3asr_transcribe_default_params(void);
q3asr_align_params q3asr_align_default_params(void);

q3asr_context * q3asr_context_create(const q3asr_context_params * params);
void q3asr_context_destroy(q3asr_context * ctx);
const char * q3asr_context_last_error(const q3asr_context * ctx);

q3asr_aligner_context * q3asr_aligner_context_create(const q3asr_aligner_context_params * params);
void q3asr_aligner_context_destroy(q3asr_aligner_context * ctx);
const char * q3asr_aligner_context_last_error(const q3asr_aligner_context * ctx);

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

int q3asr_align_wav_file(
    q3asr_aligner_context * ctx,
    const char * wav_path,
    const char * text,
    const char * language,
    q3asr_alignment_result * out_result
);

int q3asr_align_wav_file_ex(
    q3asr_aligner_context * ctx,
    const char * wav_path,
    const char * text,
    const char * language,
    const q3asr_align_params * params,
    q3asr_alignment_result * out_result
);

int q3asr_align_pcm_f32(
    q3asr_aligner_context * ctx,
    const float * samples,
    int n_samples,
    const char * text,
    const char * language,
    q3asr_alignment_result * out_result
);

int q3asr_align_pcm_f32_ex(
    q3asr_aligner_context * ctx,
    const float * samples,
    int n_samples,
    const char * text,
    const char * language,
    const q3asr_align_params * params,
    q3asr_alignment_result * out_result
);

void q3asr_transcribe_result_clear(q3asr_transcribe_result * result);
void q3asr_alignment_result_clear(q3asr_alignment_result * result);

#ifdef __cplusplus
}
#endif

#endif
