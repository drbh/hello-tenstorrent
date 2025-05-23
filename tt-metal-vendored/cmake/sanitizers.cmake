get_property(isMultiConfig GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(isMultiConfig)
    if(NOT "ASan" IN_LIST CMAKE_CONFIGURATION_TYPES)
        list(APPEND CMAKE_CONFIGURATION_TYPES ASan)
    endif()
    if(NOT "TSan" IN_LIST CMAKE_CONFIGURATION_TYPES)
        list(APPEND CMAKE_CONFIGURATION_TYPES TSan)
    endif()
endif()

set_property(
    GLOBAL
    APPEND
    PROPERTY
        DEBUG_CONFIGURATIONS
            ASan
            TSan
)

# ASan, LSan and UBSan do not conflict with each other and are each fast enough that we can combine them.
# Saves us from an explosion of pipelines to test our code.
set(asan_flags "-fsanitize=address -fsanitize=leak -fsanitize=undefined")
set(asan_compile_flags "${asan_flags} -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_ASAN "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${asan_compile_flags}")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${asan_compile_flags}")
set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} ${asan_flags}")
set(CMAKE_SHARED_LINKER_FLAGS_ASAN "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} ${asan_flags}")
# set(CMAKE_STATIC_LINKER_FLAGS_ASAN "${CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO} ${asan_flags}")
set(CMAKE_MODULE_LINKER_FLAGS_ASAN "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO} ${asan_flags}")

set(CMAKE_C_FLAGS_TSAN "${CMAKE_C_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS_TSAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer")
set(CMAKE_SHARED_LINKER_FLAGS_TSAN
    "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer"
)
# set(CMAKE_STATIC_LINKER_FLAGS_TSAN
# "${CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer"
# )
set(CMAKE_MODULE_LINKER_FLAGS_TSAN
    "${CMAKE_MODULE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=thread -fno-omit-frame-pointer"
)
