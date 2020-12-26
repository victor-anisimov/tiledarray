# -*- mode: cmake -*-

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
include(AppendFlags)

# if CUDA is enabled (assuming CUDA version is 9 or 10) need Eigen 3.3.7
# see https://gitlab.com/libeigen/eigen/issues/1491
if (ENABLE_CUDA AND ${TA_TRACKED_EIGEN_VERSION} VERSION_LESS 3.3.7)
  message(FATAL_ERROR "** ENABLE_CUDA requires Eigen 3.3.7 or greater")
endif()

# Check for existing Eigen
# prefer CMake-configured-and-installed instance
# re:NO_CMAKE_PACKAGE_REGISTRY: eigen3 registers its *build* tree with the user package registry ...
#                               to avoid issues with wiped build directory look for installed eigen
find_package(Eigen3 ${TA_TRACKED_EIGEN_VERSION} NO_MODULE QUIET NO_CMAKE_PACKAGE_REGISTRY)

if (NOT TARGET Eigen3::Eigen)

  if (TA_EXPERT)
    message("** Eigen3 was not found")
    message(FATAL_ERROR "** Downloading and building Eigen3 is explicitly disabled in TA_EXPERT mode")
  endif()

  set (
    TA_INSTALL_EIGEN_URL
    https://gitlab.com/libeigen/eigen/-/archive/${TA_INSTALL_EIGEN_VERSION}/eigen-${TA_INSTALL_EIGEN_VERSION}.tar.bz2
    )
  
  message("Downloading Eigen-${TA_INSTALL_EIGEN_VERSION} from ${TA_INSTALL_EIGEN_URL}")

  include(FetchContent)
  FetchContent_Declare(
    eigen
    URL ${TA_INSTALL_EIGEN_URL}
    URL_HASH MD5=${TA_INSTALL_EIGEN_URL_HASH}
    )

  if(NOT eigen_POPULATED)
    FetchContent_populate(eigen)
    add_library (eigen INTERFACE)
    target_compile_definitions (eigen INTERFACE ${EIGEN_DEFINITIONS})
    target_include_directories (eigen INTERFACE
      $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
      $<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>
      )
    # Export as title case Eigen
    set_target_properties (eigen PROPERTIES EXPORT_NAME Eigen)
    add_library(Eigen3::Eigen ALIAS eigen)
  endif(NOT eigen_POPULATED)

endif()

add_library(TiledArray_Eigen INTERFACE)
foreach(prop INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_LIBRARIES INTERFACE_POSITION_INDEPENDENT_CODE)
  get_property(EIGEN3_${prop} TARGET Eigen3::Eigen PROPERTY ${prop})
  set_property(TARGET TiledArray_Eigen PROPERTY ${prop} ${EIGEN3_${prop}})
  message("!!! ${prop} ${EIGEN3_${prop}}")
endforeach()

# Perform a compile check with Eigen
cmake_push_check_state()

# INTERFACE libraries cannot be used as CMAKE_REQUIRED_LIBRARIES, so must manually transfer deps info
get_property(EIGEN3_INCLUDE_DIRS TARGET TiledArray_Eigen PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
if (NOT MADNESS_INTERNAL_INCLUDE_DIRS)
  message(FATAL_ERROR "eigen.cmake must be loaded after calling detect_MADNESS_config()")
endif()
list(APPEND CMAKE_REQUIRED_INCLUDES ${EIGEN3_INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/src ${PROJECT_SOURCE_DIR}/src
  ${MADNESS_INTERNAL_INCLUDE_DIRS} ${LAPACK_INCLUDE_DIRS})
list(APPEND CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
foreach(_def ${LAPACK_COMPILE_DEFINITIONS})
  list(APPEND CMAKE_REQUIRED_DEFINITIONS "-D${_def}")
endforeach()
list(APPEND CMAKE_REQUIRED_FLAGS ${LAPACK_COMPILE_OPTIONS})

CHECK_CXX_SOURCE_COMPILES("
    #include <Eigen/Dense>
    #include <Eigen/SparseCore>
    #include <iostream>
    int main(int argc, char* argv[]){
      Eigen::MatrixXd m = Eigen::MatrixXd::Random(5, 5);
      m = m.transpose() + m;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(m);
      Eigen::MatrixXd m_invsqrt = eig.operatorInverseSqrt();
      std::cout << m_invsqrt << std::endl;
    }"
  EIGEN3_COMPILES)

cmake_pop_check_state()

if (NOT EIGEN3_COMPILES)
  message(FATAL_ERROR "Eigen3 found, but failed to compile test program")
endif()


# finish configuring TiledArray_Eigen and install
if (TARGET TiledArray_Eigen)
  # TiledArray_Eigen uses LAPACK/MKL
  target_link_libraries(TiledArray_Eigen INTERFACE ${LAPACK_LIBRARIES})
  target_include_directories(TiledArray_Eigen INTERFACE ${LAPACK_INCLUDE_DIRS})
  target_compile_definitions(TiledArray_Eigen INTERFACE ${LAPACK_COMPILE_DEFINITIONS})
  target_compile_options(TiledArray_Eigen INTERFACE ${LAPACK_COMPILE_OPTIONS})
  set(TiledArray_Eigen_VERSION "${Eigen3_VERSION}" CACHE STRING "Eigen3_VERSION of the library interfaced by TiledArray_Eigen target")
  # Eigen's prototypes for BLAS interface libraries do not match MADNESS cblas
  if (MADNESS_HAS_MKL)
    # target_compile_definitions(TiledArray_Eigen INTERFACE EIGEN_USE_MKL EIGEN_USE_BLAS)
  else(MADNESS_HAS_MKL)
    # target_compile_definitions(TiledArray_Eigen INTERFACE EIGEN_USE_BLAS)
  endif(MADNESS_HAS_MKL)
  install(TARGETS TiledArray_Eigen EXPORT tiledarray COMPONENT tiledarray)
endif(TARGET TiledArray_Eigen)
