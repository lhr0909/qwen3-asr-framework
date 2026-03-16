cmake_minimum_required(VERSION 3.20)

project(q3asr VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

option(Q3ASR_BUILD_CLI "Build the q3asr wav-file CLI" ON)
option(Q3ASR_BUILD_TESTS "Build q3asr tests" ON)
option(Q3ASR_ENABLE_METAL "Enable Metal backend when building llama.cpp on Apple platforms" ON)

set(
    Q3ASR_LLAMA_CPP_SOURCE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp"
    CACHE PATH
    "Path to an upstream llama.cpp checkout used as the decoder and ggml dependency"
)

if(NOT EXISTS "${Q3ASR_LLAMA_CPP_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Q3ASR_LLAMA_CPP_SOURCE_DIR does not point at a llama.cpp checkout")
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)

if(APPLE AND Q3ASR_ENABLE_METAL)
    set(GGML_METAL ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${Q3ASR_LLAMA_CPP_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/llama.cpp-upstream" EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)

set(Q3ASR_SOURCES
    src/audio_encoder.cpp
    src/decoder_llama.cpp
    src/gguf_loader.cpp
    src/mel_spectrogram.cpp
    src/q3asr.cpp
)

add_library(q3asr STATIC ${Q3ASR_SOURCES})

target_include_directories(q3asr
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${Q3ASR_LLAMA_CPP_SOURCE_DIR}/include"
        "${Q3ASR_LLAMA_CPP_SOURCE_DIR}/ggml/include"
)

target_link_libraries(q3asr
    PUBLIC
        llama
    PRIVATE
        ggml
        ggml-base
        ggml-cpu
        Threads::Threads
)

if(APPLE)
    target_link_libraries(q3asr PRIVATE "-framework Accelerate")
endif()

if(Q3ASR_BUILD_CLI)
    add_executable(q3asr-cli cli/q3asr_cli.cpp)
    target_link_libraries(q3asr-cli PRIVATE q3asr)
endif()

if(Q3ASR_BUILD_TESTS)
    enable_testing()

    add_executable(q3asr-smoke-test tests/smoke_test.cpp)
    target_link_libraries(q3asr-smoke-test PRIVATE q3asr)

    set(
        Q3ASR_TEST_TEXT_MODEL
        "${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/Qwen3-ASR-1.7B-text-Q8_0.gguf"
        CACHE FILEPATH
        "Default text model for the smoke test"
    )
    set(
        Q3ASR_TEST_MMPROJ_MODEL
        "${CMAKE_CURRENT_SOURCE_DIR}/models/gguf/Qwen3-ASR-1.7B-mmproj.gguf"
        CACHE FILEPATH
        "Default mmproj model for the smoke test"
    )
    set(
        Q3ASR_TEST_AUDIO
        "${CMAKE_CURRENT_SOURCE_DIR}/testdata/asr_zh.wav"
        CACHE FILEPATH
        "Default wav file for the smoke test"
    )
    set(
        Q3ASR_TEST_AUDIO_ENGLISH
        "${CMAKE_CURRENT_SOURCE_DIR}/testdata/q3asr-input.wav"
        CACHE FILEPATH
        "Optional English regression wav file for parity checks"
    )

    if(
        EXISTS "${Q3ASR_TEST_TEXT_MODEL}" AND
        EXISTS "${Q3ASR_TEST_MMPROJ_MODEL}" AND
        EXISTS "${Q3ASR_TEST_AUDIO}"
    )
        add_test(
            NAME q3asr-smoke
            COMMAND q3asr-smoke-test
                --text-model "${Q3ASR_TEST_TEXT_MODEL}"
                --mmproj-model "${Q3ASR_TEST_MMPROJ_MODEL}"
                --audio "${Q3ASR_TEST_AUDIO}"
                --expect-language "Chinese"
        )

        if(EXISTS "${Q3ASR_TEST_AUDIO_ENGLISH}")
            add_test(
                NAME q3asr-english-regression
                COMMAND q3asr-smoke-test
                    --text-model "${Q3ASR_TEST_TEXT_MODEL}"
                    --mmproj-model "${Q3ASR_TEST_MMPROJ_MODEL}"
                    --audio "${Q3ASR_TEST_AUDIO_ENGLISH}"
                    --expect-language "English"
                    --expect-substring "/r/apple"
            )
        endif()
    else()
        message(STATUS "q3asr-smoke test not registered because default model/audio files are missing")
    endif()
endif()
