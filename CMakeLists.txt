# GreyML build config: CMakeLists.txt - CMake entry for targets/tooling.
cmake_minimum_required(VERSION 3.20)
project(greyarea VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

# CRITICAL FIX #1: Force Release mode to prevent Debug flag injection
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# CRITICAL FIX #2: Clear and set compiler flags BEFORE any targets
if(MSVC)
    # Clear all inherited flags that might cause conflicts
    set(CMAKE_C_FLAGS "")
    set(CMAKE_C_FLAGS_DEBUG "")
    set(CMAKE_C_FLAGS_RELEASE "")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "")
    set(CMAKE_C_FLAGS_MINSIZEREL "")
    
    # Set consistent, non-conflicting flags for ALL configurations
    set(CMAKE_C_FLAGS "/O2 /Oi /GL /arch:AVX2 /MD /DNDEBUG /D_CRT_SECURE_NO_WARNINGS")
    set(CMAKE_C_FLAGS_DEBUG ${CMAKE_C_FLAGS})
    set(CMAKE_C_FLAGS_RELEASE ${CMAKE_C_FLAGS})
    set(CMAKE_C_FLAGS_RELWITHDEBINFO ${CMAKE_C_FLAGS})
    set(CMAKE_C_FLAGS_MINSIZEREL ${CMAKE_C_FLAGS})
endif()

# CRITICAL FIX #3: Windows-specific definitions
if(WIN32)
    add_definitions(-DGREYAREA_EXPORTS)
    # Set Windows subsystem to CONSOLE for proper output
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
endif()

# Options
option(GA_USE_OPENMP "Enable OpenMP parallelization" ON)
option(GA_USE_BLAS "Use OpenBLAS for matrix operations" OFF)
option(GA_BUILD_TESTS "Build test suite" ON)
option(GA_BUILD_EXAMPLES "Build example programs" ON)
option(GA_BUILD_PYTHON "Build Python bindings" ON)

# Collect all source files with proper paths
file(GLOB_RECURSE CORE_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} 
    src/core/*.c
)
file(GLOB_RECURSE OPS_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/ops/*.c
)
file(GLOB_RECURSE AUTOGRAD_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/autograd/*.c
)
file(GLOB_RECURSE NN_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/nn/*.c
)
file(GLOB_RECURSE OPTIM_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/optim/*.c
)
file(GLOB_RECURSE LOSS_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/loss/*.c
)
file(GLOB_RECURSE ML_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/ml/*.c
)
file(GLOB_RECURSE IO_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/io/*.c
)
file(GLOB_RECURSE SIMD_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    src/simd/*.c
)

# Combine all sources
set(ALL_SRCS 
    ${CORE_SRCS}
    ${OPS_SRCS}
    ${AUTOGRAD_SRCS}
    ${NN_SRCS}
    ${OPTIM_SRCS}
    ${LOSS_SRCS}
    ${ML_SRCS}
    ${IO_SRCS}
    ${SIMD_SRCS}
)

# Create library
add_library(greyarea SHARED ${ALL_SRCS})

if(GA_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(greyarea PUBLIC OpenMP::OpenMP_C)
endif()

# Set include directories
target_include_directories(greyarea PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# CRITICAL FIX #4: Set target properties to prevent flag inheritance
if(MSVC)
    set_target_properties(greyarea PROPERTIES
        COMPILE_FLAGS "/O2 /Oi /GL /arch:AVX2 /MD /DNDEBUG"
        LINK_FLAGS "/OPT:REF /OPT:ICF /LTCG"
        # Prevent CMake from adding its own flags
        LINK_FLAGS_DEBUG ""
        LINK_FLAGS_RELEASE ""
        LINK_FLAGS_MINSIZEREL ""
        LINK_FLAGS_RELWITHDEBINFO ""
    )
endif()

# CRITICAL FIX #5: Add math library on Unix (no-op on Windows)
if(NOT WIN32)
    target_link_libraries(greyarea PUBLIC m)
endif()

# Copy DLL to Python package
add_custom_command(TARGET greyarea POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_SOURCE_DIR}/python/greyml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "$<TARGET_FILE:greyarea>"
        "${CMAKE_SOURCE_DIR}/python/greyml/$<TARGET_FILE_NAME:greyarea>"
    COMMENT "Copying DLL to Python package"
    VERBATIM
)

# Install rules (for future distribution)
if(GA_BUILD_PYTHON)
    install(TARGETS greyarea
        RUNTIME DESTINATION python/greyml
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
else()
    install(TARGETS greyarea
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()

install(DIRECTORY include/ DESTINATION include)
