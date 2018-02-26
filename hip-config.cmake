
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was hip-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

include(CMakeFindDependencyMacro OPTIONAL RESULT_VARIABLE _CMakeFindDependencyMacro_FOUND)
if (NOT _CMakeFindDependencyMacro_FOUND)
  macro(find_dependency dep)
    if (NOT ${dep}_FOUND)
      set(cmake_fd_version)
      if (${ARGC} GREATER 1)
        set(cmake_fd_version ${ARGV1})
      endif()
      set(cmake_fd_exact_arg)
      if(${CMAKE_FIND_PACKAGE_NAME}_FIND_VERSION_EXACT)
        set(cmake_fd_exact_arg EXACT)
      endif()
      set(cmake_fd_quiet_arg)
      if(${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
        set(cmake_fd_quiet_arg QUIET)
      endif()
      set(cmake_fd_required_arg)
      if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
        set(cmake_fd_required_arg REQUIRED)
      endif()
      find_package(${dep} ${cmake_fd_version}
          ${cmake_fd_exact_arg}
          ${cmake_fd_quiet_arg}
          ${cmake_fd_required_arg}
      )
      string(TOUPPER ${dep} cmake_dep_upper)
      if (NOT ${dep}_FOUND AND NOT ${cmake_dep_upper}_FOUND)
        set(${CMAKE_FIND_PACKAGE_NAME}_NOT_FOUND_MESSAGE "${CMAKE_FIND_PACKAGE_NAME} could not be found because dependency ${dep} could not be found.")
        set(${CMAKE_FIND_PACKAGE_NAME}_FOUND False)
        return()
      endif()
      set(cmake_fd_version)
      set(cmake_fd_required_arg)
      set(cmake_fd_quiet_arg)
      set(cmake_fd_exact_arg)
    endif()
  endmacro()
endif()


set_and_check( hip_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include" )
set_and_check( hip_INCLUDE_DIRS "${hip_INCLUDE_DIR}" )
set_and_check( hip_LIB_INSTALL_DIR "${PACKAGE_PREFIX_DIR}/lib" )
set_and_check( hip_BIN_INSTALL_DIR "${PACKAGE_PREFIX_DIR}/bin" )

set_and_check(hip_HIPCC_EXECUTABLE "${hip_BIN_INSTALL_DIR}/hipcc")
set_and_check(hip_HIPCONFIG_EXECUTABLE "${hip_BIN_INSTALL_DIR}/hipconfig")

#find_dependency(hcc)
#include( "${CMAKE_CURRENT_LIST_DIR}/hip-targets.cmake" )

set( hip_LIBRARIES hip::hip_hcc)
set( hip_LIBRARY ${hip_LIBRARIES})

set(HIP_INCLUDE_DIR ${hip_INCLUDE_DIR})
set(HIP_INCLUDE_DIRS ${hip_INCLUDE_DIRS})
set(HIP_LIB_INSTALL_DIR ${hip_LIB_INSTALL_DIR})
set(HIP_BIN_INSTALL_DIR ${hip_BIN_INSTALL_DIR})
set(HIP_LIBRARIES ${hip_LIBRARIES})
set(HIP_LIBRARY ${hip_LIBRARY})
set(HIP_HIPCC_EXECUTABLE ${hip_HIPCC_EXECUTABLE})
set(HIP_HIPCONFIG_EXECUTABLE ${hip_HIPCONFIG_EXECUTABLE})

