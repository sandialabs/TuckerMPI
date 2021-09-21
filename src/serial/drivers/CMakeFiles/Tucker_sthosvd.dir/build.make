# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fangq18/TuckerMPI/src/serial

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fangq18/TuckerMPI/src/serial

# Include any dependencies generated for this target.
include drivers/CMakeFiles/Tucker_sthosvd.dir/depend.make

# Include the progress variables for this target.
include drivers/CMakeFiles/Tucker_sthosvd.dir/progress.make

# Include the compile flags for this target's objects.
include drivers/CMakeFiles/Tucker_sthosvd.dir/flags.make

drivers/CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.o: drivers/CMakeFiles/Tucker_sthosvd.dir/flags.make
drivers/CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.o: drivers/Tucker_sthosvd.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fangq18/TuckerMPI/src/serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object drivers/CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.o"
	cd /home/fangq18/TuckerMPI/src/serial/drivers && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.o -c /home/fangq18/TuckerMPI/src/serial/drivers/Tucker_sthosvd.cpp

drivers/CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.i"
	cd /home/fangq18/TuckerMPI/src/serial/drivers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fangq18/TuckerMPI/src/serial/drivers/Tucker_sthosvd.cpp > CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.i

drivers/CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.s"
	cd /home/fangq18/TuckerMPI/src/serial/drivers && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fangq18/TuckerMPI/src/serial/drivers/Tucker_sthosvd.cpp -o CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.s

# Object files for target Tucker_sthosvd
Tucker_sthosvd_OBJECTS = \
"CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.o"

# External object files for target Tucker_sthosvd
Tucker_sthosvd_EXTERNAL_OBJECTS =

drivers/Tucker_sthosvd: drivers/CMakeFiles/Tucker_sthosvd.dir/Tucker_sthosvd.o
drivers/Tucker_sthosvd: drivers/CMakeFiles/Tucker_sthosvd.dir/build.make
drivers/Tucker_sthosvd: libserial_tucker.so
drivers/Tucker_sthosvd: drivers/CMakeFiles/Tucker_sthosvd.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fangq18/TuckerMPI/src/serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Tucker_sthosvd"
	cd /home/fangq18/TuckerMPI/src/serial/drivers && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Tucker_sthosvd.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
drivers/CMakeFiles/Tucker_sthosvd.dir/build: drivers/Tucker_sthosvd

.PHONY : drivers/CMakeFiles/Tucker_sthosvd.dir/build

drivers/CMakeFiles/Tucker_sthosvd.dir/clean:
	cd /home/fangq18/TuckerMPI/src/serial/drivers && $(CMAKE_COMMAND) -P CMakeFiles/Tucker_sthosvd.dir/cmake_clean.cmake
.PHONY : drivers/CMakeFiles/Tucker_sthosvd.dir/clean

drivers/CMakeFiles/Tucker_sthosvd.dir/depend:
	cd /home/fangq18/TuckerMPI/src/serial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fangq18/TuckerMPI/src/serial /home/fangq18/TuckerMPI/src/serial/drivers /home/fangq18/TuckerMPI/src/serial /home/fangq18/TuckerMPI/src/serial/drivers /home/fangq18/TuckerMPI/src/serial/drivers/CMakeFiles/Tucker_sthosvd.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : drivers/CMakeFiles/Tucker_sthosvd.dir/depend

