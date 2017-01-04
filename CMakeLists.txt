IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(ctc_release)

IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")
ENDIF()

IF (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    add_definitions(-DAPPLE)
ENDIF()

include_directories(include)

FIND_PACKAGE(CUDA 6.5)
MESSAGE(STATUS "cuda found ${CUDA_FOUND}")

option(WITH_GPU "compile warp-ctc with cuda." ${CUDA_FOUND})
option(WITH_OMP "compile warp-ctc with openmp." ON)

if(NOT WITH_OMP)
    add_definitions(-DCTC_DISABLE_OMP)
endif()

# need to be at least 30 or __shfl_down in reduce wont compile
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

if (NOT APPLE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
ENDIF()

FIND_PACKAGE(Torch)

MESSAGE(STATUS "Torch found ${Torch_DIR}")

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


IF (WITH_GPU)

    MESSAGE(STATUS "Building shared library with GPU support")

    CUDA_ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cu src/reduce.cu)
    IF (!Torch_FOUND) 
        TARGET_LINK_LIBRARIES(warpctc ${CUDA_curand_LIBRARY})
    ENDIF()

    add_executable(test_cpu tests/test_cpu.cpp )
    TARGET_LINK_LIBRARIES(test_cpu warpctc)
    SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    cuda_add_executable(test_gpu tests/test_gpu.cu)
    TARGET_LINK_LIBRARIES(test_gpu warpctc ${CUDA_curand_LIBRARY})

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")

    INSTALL(FILES include/ctc.h DESTINATION "include")

    IF (Torch_FOUND)
        MESSAGE(STATUS "Building Torch Bindings with GPU support")
        INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS} "${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc")
        INCLUDE_DIRECTORIES(${Torch_INSTALL_INCLUDE} ${Torch_INSTALL_INCLUDE}/TH ${Torch_INSTALL_INCLUDE}/THC)

        TARGET_LINK_LIBRARIES(warpctc luajit luaT THC TH ${CUDA_curand_LIBRARY})
        INSTALL(TARGETS warpctc
                RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
                LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
                ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

        SET(src torch_binding/binding.cpp torch_binding/utils.c)
        SET(luasrc torch_binding/init.lua)

        ADD_TORCH_PACKAGE(warp_ctc "${src}" "${luasrc}")
        IF (APPLE)

            TARGET_LINK_LIBRARIES(warp_ctc warpctc luajit luaT THC TH ${CUDA_curand_LIBRARY})
        ELSE()
            TARGET_LINK_LIBRARIES(warp_ctc warpctc luajit luaT THC TH ${CUDA_curand_LIBRARY} gomp)
        ENDIF()
    ENDIF()


ELSE()
    MESSAGE(STATUS "Building shared library with no GPU support")

    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cpp)

    add_executable(test_cpu tests/test_cpu.cpp )
    TARGET_LINK_LIBRARIES(test_cpu warpctc)
    SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")

    INSTALL(FILES include/ctc.h DESTINATION "include")

    IF (Torch_FOUND)
        MESSAGE(STATUS "Building Torch Bindings with no GPU support")
        add_definitions(-DTORCH_NOGPU)
        INCLUDE_DIRECTORIES(${Torch_INSTALL_INCLUDE} ${Torch_INSTALL_INCLUDE}/TH)

        TARGET_LINK_LIBRARIES(warpctc luajit luaT TH)

        INSTALL(TARGETS warpctc
                RUNTIME DESTINATION "${Torch_INSTALL_BIN_SUBDIR}"
                LIBRARY DESTINATION "${Torch_INSTALL_LIB_SUBDIR}"
                ARCHIVE DESTINATION "${Torch_INSTALL_LIB_SUBDIR}")

        SET(src torch_binding/binding.cpp torch_binding/utils.c)
        SET(luasrc torch_binding/init.lua)

        ADD_TORCH_PACKAGE(warp_ctc "${src}" "${luasrc}")
        IF (APPLE)
            TARGET_LINK_LIBRARIES(warp_ctc warpctc luajit luaT TH)
        ELSE()
            TARGET_LINK_LIBRARIES(warp_ctc warpctc luajit luaT TH gomp)
        ENDIF()
    ENDIF()

ENDIF()


