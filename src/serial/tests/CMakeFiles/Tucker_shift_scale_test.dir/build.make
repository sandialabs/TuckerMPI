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
include tests/CMakeFiles/Tucker_shift_scale_test.dir/depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/Tucker_shift_scale_test.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/Tucker_shift_scale_test.dir/flags.make

tests/CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.o: tests/CMakeFiles/Tucker_shift_scale_test.dir/flags.make
tests/CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.o: tests/Tucker_shift_scale_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fangq18/TuckerMPI/src/serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.o"
	cd /home/fangq18/TuckerMPI/src/serial/tests && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.o -c /home/fangq18/TuckerMPI/src/serial/tests/Tucker_shift_scale_test.cpp

tests/CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.i"
	cd /home/fangq18/TuckerMPI/src/serial/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fangq18/TuckerMPI/src/serial/tests/Tucker_shift_scale_test.cpp > CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.i

tests/CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.s"
	cd /home/fangq18/TuckerMPI/src/serial/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fangq18/TuckerMPI/src/serial/tests/Tucker_shift_scale_test.cpp -o CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.s

# Object files for target Tucker_shift_scale_test
Tucker_shift_scale_test_OBJECTS = \
"CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.o"

# External object files for target Tucker_shift_scale_test
Tucker_shift_scale_test_EXTERNAL_OBJECTS =

tests/Tucker_shift_scale_test: tests/CMakeFiles/Tucker_shift_scale_test.dir/Tucker_shift_scale_test.o
tests/Tucker_shift_scale_test: tests/CMakeFiles/Tucker_shift_scale_test.dir/build.make
tests/Tucker_shift_scale_test: libserial_tucker.so
tests/Tucker_shift_scale_test: tests/CMakeFiles/Tucker_shift_scale_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fangq18/TuckerMPI/src/serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable Tucker_shift_scale_test"
	cd /home/fangq18/TuckerMPI/src/serial/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Tucker_shift_scale_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/Tucker_shift_scale_test.dir/build: tests/Tucker_shift_scale_test

.PHONY : tests/CMakeFiles/Tucker_shift_scale_test.dir/build

tests/CMakeFiles/Tucker_shift_scale_test.dir/clean:
	cd /home/fangq18/TuckerMPI/src/serial/tests && $(CMAKE_COMMAND) -P CMakeFiles/Tucker_shift_scale_test.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/Tucker_shift_scale_test.dir/clean

tests/CMakeFiles/Tucker_shift_scale_test.dir/depend:
	cd /home/fangq18/TuckerMPI/src/serial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fangq18/TuckerMPI/src/serial /home/fangq18/TuckerMPI/src/serial/tests /home/fangq18/TuckerMPI/src/serial /home/fangq18/TuckerMPI/src/serial/tests /home/fangq18/TuckerMPI/src/serial/tests/CMakeFiles/Tucker_shift_scale_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/Tucker_shift_scale_test.dir/depend

