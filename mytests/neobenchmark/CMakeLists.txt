CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
FIND_PACKAGE(deal.II 8.3 REQUIRED
  HINTS /mnt/data/jwitte/mydealii/install
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(laplace_meshworker)


set(PARALLEL_LA 2 CACHE STRING "Use deal.II (0) PETSc(1) or Trilinos(2)?")
add_definitions(-DPARALLEL_LA=${PARALLEL_LA})

set(MAXTHREADS 0 CACHE STRING "Maximum number of threads used per MPI process, 0 means no limit")
add_definitions(-DMAXTHREADS=${MAXTHREADS})

option(CG "Use FE_Q?" OFF)
if(CG)
  add_definitions(-DCG)
endif()

option(PERIODIC "Periodic boundary conditions?" OFF)
if(PERIODIC)
  add_definitions(-DPERIODIC)
endif()

option(MG "Multigrid Preconditioner?" ON)
if (MG)
  add_definitions(-DMG)
endif()

set(SMOOTHENINGSTEPS 10 CACHE STRING "Number of smoothening steps")
add_definitions(-DSMOOTHENINGSTEPS=${SMOOTHENINGSTEPS})

INCLUDE_DIRECTORIES(include include/local_integrators)
include_directories(/mnt/data/jwitte/mybenchmark/install/include)

find_package(Threads REQUIRED)

ADD_LIBRARY(benchmark STATIC IMPORTED)
SET_TARGET_PROPERTIES(benchmark PROPERTIES
  IMPORTED_LOCATION
  /mnt/data/jwitte/mybenchmark/install/lib/libbenchmark.a)

FILE(GLOB sources source/*.cc)

IF(CMAKE_BUILD_TYPE MATCHES "Debug")
  ADD_EXECUTABLE(laplace_meshworker.g ${sources})
  target_link_libraries(laplace_meshworker.g benchmark ${CMAKE_THREAD_LIBS_INIT})
  DEAL_II_SETUP_TARGET(laplace_meshworker.g)
ELSEIF(CMAKE_BUILD_TYPE MATCHES "Release")
  ADD_EXECUTABLE(laplace_meshworker ${sources})
  target_link_libraries(laplace_meshworker benchmark ${CMAKE_THREAD_LIBS_INIT})
  DEAL_II_SETUP_TARGET(laplace_meshworker)
ELSE()
  MESSAGE(FATAL_ERROR
    "CMAKE_BUILD_TYPE does neither match Release nor Debug!"
    )
ENDIF()

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

ADD_CUSTOM_TARGET(both)
ADD_DEPENDENCIES(both debug release)

