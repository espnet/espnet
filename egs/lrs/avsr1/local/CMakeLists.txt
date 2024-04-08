cmake_minimum_required (VERSION 3.8)
set(CMAKE_CXX_STANDARD 17)

project(OpenFace VERSION 2.0.2)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

set(CMAKE_CONFIG_DIR etc/OpenFace)
set(CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_CONFIG_DIR}")
add_definitions(-DCONFIG_DIR="${CONFIG_DIR}")

# make sure we'll use OpenBLAS only: there's a header file naming difference between different
# implementations; so OpenFace wants OpenBLAS;
find_package(Threads)
find_package(OpenBLAS REQUIRED)
if ( ${OpenBLAS_FOUND} )
    MESSAGE("OpenBLAS information:")
    MESSAGE("  OpenBLAS_LIBRARIES: ${OpenBLAS_LIB}")
else()
    MESSAGE(FATAL_ERROR "OpenBLAS not found in the system.")
endif()

if ( ${OpenBLAS_INCLUDE_FOUND} )
    MESSAGE("  OpenBLAS_INCLUDE: ${OpenBLAS_INCLUDE_DIR}")
else()
    MESSAGE(WARNING "OpenBLAS include not found in the system. Using the one vended with OpenFace.")
	set(OpenBLAS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/3rdParty/OpenBLAS/include")
    MESSAGE("  OpenBLAS_INCLUDE: ${OpenBLAS_INCLUDE_DIR}")
endif()

find_package( OpenCV 4.0 REQUIRED COMPONENTS core imgproc calib3d highgui objdetect)
if(${OpenCV_FOUND})
	MESSAGE("OpenCV information:")
	MESSAGE("  OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
	MESSAGE("  OpenCV_LIBRARIES: ${OpenCV_LIBRARIES}")
	MESSAGE("  OpenCV_LIBRARY_DIRS: ${OpenCV_LINK_DIRECTORIES}")
else()
    MESSAGE(FATAL_ERROR "OpenCV not found in the system.")
endif()

find_package( Boost 1.5.9 COMPONENTS filesystem system)
if(${Boost_FOUND})
	MESSAGE("Boost information:")
	MESSAGE("  Boost_VERSION: ${Boost_VERSION}")
	MESSAGE("  Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
	MESSAGE("  Boost_LIBRARIES: ${Boost_LIBRARIES}")
	MESSAGE("  Boost_LIBRARY_DIRS: ${Boost_LIBRARY_DIRS}")
else()
    MESSAGE("Boost not found in the system.")
endif()


# Move LandmarkDetector model
file(GLOB files "lib/local/LandmarkDetector/model/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model)
endforeach()

# Move the hierarchical LandmarkDetector models
file(GLOB files "lib/local/LandmarkDetector/model/model*")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model)
	install(DIRECTORY ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model)
endforeach()

# Move detection validation models
file(GLOB files "lib/local/LandmarkDetector/model/detection_validation/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/detection_validation)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/detection_validation)
endforeach()

# Move patch experts
file(GLOB files "lib/local/LandmarkDetector/model/patch_experts/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/patch_experts)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/patch_experts)
endforeach()

# Move CEN patch experts
file(GLOB files "lib/local/LandmarkDetector/model/patch_experts/*.dat")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/patch_experts)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/patch_experts)
endforeach()

# Move MTCNN face detector
file(GLOB files "lib/local/LandmarkDetector/model/mtcnn_detector/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/mtcnn_detector)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/mtcnn_detector)
endforeach()

# Move MTCNN face detector
file(GLOB files "lib/local/LandmarkDetector/model/mtcnn_detector/*.dat")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/mtcnn_detector)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/mtcnn_detector)
endforeach()

# Move Point Distribution models
file(GLOB files "lib/local/LandmarkDetector/model/pdms/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/model/pdms)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/model/pdms)
endforeach()

# Move OpenCV classifiers
file(GLOB files "lib/3rdParty/OpenCV3.4/classifiers/*.xml")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/classifiers)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/classifiers)
endforeach()

# Move AU prediction modules
file(GLOB files "lib/local/FaceAnalyser/AU_predictors/*.txt")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/AU_predictors)
	install(FILES ${file} DESTINATION ${CMAKE_CONFIG_DIR}/AU_predictors)
endforeach()

# Move AU prediction modules
file(GLOB files "lib/local/FaceAnalyser/AU_predictors/svr*")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/AU_predictors)
	install(DIRECTORY ${file} DESTINATION ${CMAKE_CONFIG_DIR}/AU_predictors)
endforeach()

# Move AU prediction modules
file(GLOB files "lib/local/FaceAnalyser/AU_predictors/svm*")
foreach(file ${files})
	file(COPY ${file} DESTINATION ${CMAKE_BINARY_DIR}/bin/AU_predictors)
	install(DIRECTORY ${file} DESTINATION ${CMAKE_CONFIG_DIR}/AU_predictors)
endforeach()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
    if (GCC_VERSION VERSION_LESS 8.0)
		MESSAGE(FATAL_ERROR "Need a 8.0 or newer GCC compiler. Current GCC: ${GCC_VERSION}")
    else ()
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3")
    endif ()
endif ()

# dlib
find_package(dlib 19.13)
if(${dlib_FOUND})
    message("dlib information:")
    message("  dlib version: ${dlib_VERSION}")

    if (NOT TARGET dlib)
        add_library(dlib INTERFACE IMPORTED GLOBAL)
    endif()
else()
    message(FATAL_ERROR "dlib not found in the system, please install dlib")
endif()

# suppress auto_ptr deprecation warnings
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    add_compile_options("-Wno-deprecated-declarations")
endif()

# LandmarkDetector library
add_subdirectory(lib/local/LandmarkDetector)
# Facial Expression analysis library
add_subdirectory(lib/local/FaceAnalyser)
# Gaze estimation library
add_subdirectory(lib/local/GazeAnalyser)
# Utilities library
add_subdirectory(lib/local/Utilities)

# test if this file is a top list file
# thus we're building an OpenFace as a standalone
# project; otherwise OpenFace is being built as a
# part or larger tree;
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL "${CMAKE_SOURCE_DIR}")

    # for a standalone builds - allow installing package configs;
    message(STATUS "Standalone mode detected; Enabling configuration/targets export.")

    # export libraries for reuse
    include(CMakePackageConfigHelpers)

    set(LIB_INSTALL_DIR lib)
    set(CONFIG_DEST_DIR ${LIB_INSTALL_DIR}/cmake/OpenFace/)
    set(OpenFace_LIBRARIES OpenFace::GazeAnalyser OpenFace::FaceAnalyser OpenFace::LandmarkDetector OpenFace::Utilities)

    # export targets [build tree]
    export(EXPORT OpenFaceTargets
        NAMESPACE OpenFace::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_DEST_DIR}/OpenFaceTargets.cmake")

    # write package version file
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_DEST_DIR}/OpenFaceConfigVersion.cmake"
        COMPATIBILITY AnyNewerVersion)

    # define [build tree] bindir relative include dir
    foreach(lib ${OpenFace_LIBRARIES})
      if(TARGET ${lib})
        get_target_property(libname ${lib} "NAME")
        file(RELATIVE_PATH rel_incdir ${CMAKE_CURRENT_BINARY_DIR} "${CMAKE_CURRENT_SOURCE_DIR}/lib/local/${libname}/include")
        list(APPEND OPENFACE_INCLUDE_DIRS ${rel_incdir})
      endif()
    endforeach()
    list(REMOVE_DUPLICATES OPENFACE_INCLUDE_DIRS)

    # write package config file from template [build tree]
    # all PATH_VARS should be relative to a ${CMAKE_CURRENT_BINARY_DIR}
    # as it's the "prefix" of our non installed package in the build tree
    configure_package_config_file(cmake/OpenFaceConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_DEST_DIR}/OpenFaceConfig.cmake"
        INSTALL_DESTINATION ${CONFIG_DEST_DIR}
        PATH_VARS OPENFACE_INCLUDE_DIRS)

    # store current build dir in the CMake package registry
    # export(PACKAGE OpenFace)

    # install exported targets [install tree]
    install(EXPORT OpenFaceTargets
        FILE OpenFaceTargets.cmake
        NAMESPACE OpenFace::
        DESTINATION ${CONFIG_DEST_DIR})

    # redefine [install tree] prefix relative include dir
    set(OPENFACE_INCLUDE_DIRS "include/OpenFace")

    # write package config file from template [install tree]
    configure_package_config_file(cmake/OpenFaceConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/OpenFace/OpenFaceConfig.cmake"
        INSTALL_DESTINATION ${CONFIG_DEST_DIR}
        PATH_VARS OPENFACE_INCLUDE_DIRS)

    # install package configs
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/OpenFace/OpenFaceConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_DEST_DIR}/OpenFaceConfigVersion.cmake"
        DESTINATION ${CONFIG_DEST_DIR})
endif()

# executables
add_subdirectory(exe/FaceLandmarkImg)
add_subdirectory(exe/FaceLandmarkVid)
add_subdirectory(exe/FaceLandmarkVidMulti)
add_subdirectory(exe/FeatureExtraction)
