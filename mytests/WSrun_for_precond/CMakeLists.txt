CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.5.0 REQUIRED
  HINTS ../../../install 
  )

DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(wsrun)

include_directories(/mnt/data/jwitte/mybenchmark/install/include)
find_package(Threads REQUIRED)

FILE(GLOB sources source/*.cc)
include_directories(include)
ADD_EXECUTABLE(wsrun ${sources})

ADD_LIBRARY(benchmark STATIC IMPORTED)
SET_TARGET_PROPERTIES(benchmark PROPERTIES
  IMPORTED_LOCATION
  /mnt/data/jwitte/mybenchmark/install/lib/libbenchmark.a)
target_link_libraries(wsrun benchmark ${CMAKE_THREAD_LIBS_INIT})

DEAL_II_SETUP_TARGET(wsrun)