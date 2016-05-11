CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.5.0 REQUIRED
  HINTS ../../../install 
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(myloop)

include_directories($ENV{MYBENCHMARK_DIR}/include)

find_package(Threads REQUIRED)

ADD_EXECUTABLE(myloop main.cc)

ADD_LIBRARY(benchmark STATIC IMPORTED)
SET_TARGET_PROPERTIES(benchmark PROPERTIES
  IMPORTED_LOCATION
  $ENV{MYBENCHMARK_DIR}/lib/libbenchmark.a)
target_link_libraries(myloop benchmark ${CMAKE_THREAD_LIBS_INIT})

DEAL_II_SETUP_TARGET(myloop)
