cmake_minimum_required(VERSION 3.18.0)

set(SYCL_FOUND TRUE)
set(TILEDARRAY_HAS_SYCL 1 CACHE BOOL "Whether TiledArray has SYCL support")

##
## Umpire
##
include(external/umpire.cmake)

##
## libreTT
##
include(external/librett.cmake)
