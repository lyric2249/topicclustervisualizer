# - Config file for the Armadillo package
# It defines the following variables
#  ARMADILLO_INCLUDE_DIRS - include directories for Armadillo
#  ARMADILLO_LIBRARY_DIRS - library directories for Armadillo (normally not used!)
#  ARMADILLO_LIBRARIES    - libraries to link against

# Tell the user project where to find our headers and libraries
set(ARMADILLO_INCLUDE_DIRS "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/install/x64-Debug/include")
set(ARMADILLO_LIBRARY_DIRS "C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/install/x64-Debug/lib")

# Our library dependencies (contains definitions for IMPORTED targets)
include("C:/Users/Song1/source/repos/functioncc/functioncc/armadillo-10.6.2/out/install/x64-Debug/share/Armadillo/CMake/ArmadilloLibraryDepends.cmake")

# These are IMPORTED targets created by ArmadilloLibraryDepends.cmake
set(ARMADILLO_LIBRARIES armadillo)

