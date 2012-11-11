#####
##
## Copyright (C) 2012 by the deal.II authors
##
## This file is part of the deal.II library.
##
## <TODO: Full License information>
## This file is dual licensed under QPL 1.0 and LGPL 2.1 or any later
## version of the LGPL license.
##
## Author: Matthias Maier <matthias.maier@iwr.uni-heidelberg.de>
##
#####

#
# General setup for GCC and compilers sufficiently close to GCC
#
# Please read the fat note in setup_compiler_flags.cmake prior to
# editing this file.
#

IF( CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
    CMAKE_CXX_COMPILER_VERSION VERSION_LESS "3.4" )
  MESSAGE(WARNING "\n"
    "You're using an old version of the GNU Compiler Collection (gcc/g++)!\n"
    "It is strongly recommended to use at least version 3.4.\n"
    )
ENDIF()


########################
#                      #
#    General setup:    #
#                      #
########################

#
# Set the pic flag.
#
ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-fpic")

#
# Check whether the -as-needed flag is available. If so set it to link
# the deal.II library with it.
#
ENABLE_IF_LINKS(CMAKE_SHARED_LINKER_FLAGS "-Wl,--as-needed")

#
# Set -pedantic if the compiler supports it.
#
IF(NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
        CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.4"))
  ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-pedantic")
ENDIF()

#
# Setup various warnings:
#
ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wall")
ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wpointer-arith")
ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wwrite-strings")
ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wsynth")
ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wsign-compare")
ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wswitch")

#
# Newer versions of gcc have a flag -Wunused-local-typedefs that, though in
# principle a good idea, triggers a lot in BOOST in various places.
# Unfortunately, this warning is included in -W/-Wall, so disable it if the
# compiler supports it.
#
ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wno-unused-local-typedefs")

#
# Silence the -Wnewline-eof warning. We trigger this in various 3rd party
# headers...
#
ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wno-newline-eof")


IF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  #
  # *Boy*, clang seems to be the very definition of "pedantic" in
  # "-pedantic" mode, so disable a bunch of harmless warnings
  # (that are mainly triggered in third party headers so that we cannot
  # easily fix them...)
  #
  ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wno-delete-non-virtual-dtor") # not harmless but needed for boost <1.50.0
  ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wno-long-long")
  ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wno-unused-function")
  ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wno-unused-variable")
ENDIF()


IF(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  #
  # Use -Wno-long-long on Apple Darwin to avoid some unnecessary
  # warnings. However, newer gccs on that platform do not have
  # this flag any more, so check whether we can indeed do this
  #
  ENABLE_IF_SUPPORTED(CMAKE_CXX_FLAGS "-Wno-long-double")
ENDIF()


#############################
#                           #
#    For Release target:    #
#                           #
#############################

IF (CMAKE_BUILD_TYPE MATCHES "Release")
  #
  # General optimization flags:
  #
  ADD_FLAGS(DEAL_II_CXX_FLAGS_RELEASE "-O2")

  ENABLE_IF_SUPPORTED(DEAL_II_CXX_FLAGS_RELEASE "-funroll-loops")
  ENABLE_IF_SUPPORTED(DEAL_II_CXX_FLAGS_RELEASE "-fstrict-aliasing")
  ENABLE_IF_SUPPORTED(DEAL_II_CXX_FLAGS_RELEASE "-felide-constructors")

  ENABLE_IF_SUPPORTED(DEAL_II_CXX_FLAGS_RELEASE "-Wno-unused")
ENDIF()


###########################
#                         #
#    For Debug target:    #
#                         #
###########################

IF (CMAKE_BUILD_TYPE MATCHES "Debug")

  LIST(APPEND DEAL_II_DEFINITIONS_DEBUG "DEBUG")
  LIST(APPEND DEAL_II_USER_DEFINITIONS_DEBUG "DEBUG")

  ADD_FLAGS(DEAL_II_CXX_FLAGS_DEBUG "-O0")

  ENABLE_IF_SUPPORTED(DEAL_II_CXX_FLAGS_DEBUG "-ggdb")
  ENABLE_IF_SUPPORTED(DEAL_II_SHARED_LINKER_FLAGS_DEBUG "-ggdb")
  #
  # If -ggdb is not available, fall back to -g:
  #
  IF(NOT DEAL_II_HAVE_FLAG_-ggdb)
    ENABLE_IF_SUPPORTED(DEAL_II_CXX_FLAGS_DEBUG "-g")
    ENABLE_IF_SUPPORTED(DEAL_II_SHARED_LINKER_FLAGS_DEBUG "-g")
  ENDIF()
ENDIF()


#########################
#                       #
#    Set up C FLAGS:    #
#                       #
#########################

#
# For the moment we assume that CC and CXX are the same compiler and that
# we can set (almost) the same default flags for both:
#
SET(CMAKE_C_FLAGS ${CMAKE_CXX_FLAGS})
SET(DEAL_II_C_FLAGS_RELEASE ${DEAL_II_CXX_FLAGS_RELEASE})
SET(DEAL_II_C_FLAGS_DEBUG ${DEAL_II_CXX_FLAGS_DEBUG})

#
# OK, touché, touché. We have to strip flags not supported by a C target:
#
STRIP_FLAG(CMAKE_C_FLAGS "-Wsynth")
STRIP_FLAG(DEAL_II_C_FLAGS_RELEASE "-felide-constructors")

#
# and disable some warnings:
#
STRIP_FLAG(CMAKE_C_FLAGS "-Wall") # There is no other way to disable -Wunknown-pragma atm...
STRIP_FLAG(CMAKE_C_FLAGS "-Wsign-compare")
STRIP_FLAG(CMAKE_C_FLAGS "-Wwrite-strings")

