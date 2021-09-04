# Install script for directory: C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/build/x64-Debug/tmp/include/" REGEX "/\\.svn$" EXCLUDE REGEX "/[^/]*\\.cmake$" EXCLUDE REGEX "/[^/]*\\~$" EXCLUDE REGEX "/[^/]*orig$" EXCLUDE)
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/build/x64-Debug/armadillo.lib")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake"
         "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/build/x64-Debug/CMakeFiles/Export/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake/ArmadilloLibraryDepends-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake" TYPE FILE FILES "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/build/x64-Debug/CMakeFiles/Export/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Armadillo/CMake" TYPE FILE FILES "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/build/x64-Debug/CMakeFiles/Export/share/Armadillo/CMake/ArmadilloLibraryDepends-debug.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/install/x64-Debug/share/Armadillo/CMake/ArmadilloConfig.cmake;C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/install/x64-Debug/share/Armadillo/CMake/ArmadilloConfigVersion.cmake")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/install/x64-Debug/share/Armadillo/CMake" TYPE FILE FILES
    "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/build/x64-Debug/InstallFiles/ArmadilloConfig.cmake"
    "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/build/x64-Debug/InstallFiles/ArmadilloConfigVersion.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/build/x64-Debug/tmp/misc/armadillo.pc")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/build/x64-Debug/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")