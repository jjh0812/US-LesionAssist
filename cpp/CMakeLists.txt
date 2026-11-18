cmake_minimum_required(VERSION 3.19)
project(USLesionAssistCpp VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)
set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to unpacked ONNX Runtime C/C++ binary distribution")
if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "Set -DONNXRUNTIME_ROOT=C:/libs/onnxruntime-win-x64-<version>")
endif()
find_path(ORT_INCLUDE_DIR NAMES onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH REQUIRED)
find_library(ORT_LIBRARY NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(lesion_infer src/main.cpp)
target_include_directories(lesion_infer PRIVATE "${ORT_INCLUDE_DIR}" ${OpenCV_INCLUDE_DIRS})
target_link_libraries(lesion_infer PRIVATE "${ORT_LIBRARY}" ${OpenCV_LIBS})

option(USLA_COPY_ORT_DLL "Copy ONNX Runtime DLL as post-build step" OFF)

if(WIN32 AND USLA_COPY_ORT_DLL)
    if(EXISTS "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
        add_custom_command(TARGET lesion_infer POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll"
                "$<TARGET_FILE_DIR:lesion_infer>/onnxruntime.dll"
            COMMENT "Copy ONNX Runtime CPU DLL alongside executable")
    else()
        message(WARNING "onnxruntime.dll was not found at <root>/lib; put it on PATH or beside lesion_infer.exe")
    endif()
endif()
