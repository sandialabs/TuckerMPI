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
include compare/CMakeFiles/qr_routine_comp.dir/depend.make

# Include the progress variables for this target.
include compare/CMakeFiles/qr_routine_comp.dir/progress.make

# Include the compile flags for this target's objects.
include compare/CMakeFiles/qr_routine_comp.dir/flags.make

compare/CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.o: compare/CMakeFiles/qr_routine_comp.dir/flags.make
compare/CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.o: compare/LAPACK_qr_routine_comp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/fangq18/TuckerMPI/src/serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object compare/CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.o"
	cd /home/fangq18/TuckerMPI/src/serial/compare && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.o -c /home/fangq18/TuckerMPI/src/serial/compare/LAPACK_qr_routine_comp.cpp

compare/CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.i"
	cd /home/fangq18/TuckerMPI/src/serial/compare && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/fangq18/TuckerMPI/src/serial/compare/LAPACK_qr_routine_comp.cpp > CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.i

compare/CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.s"
	cd /home/fangq18/TuckerMPI/src/serial/compare && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/fangq18/TuckerMPI/src/serial/compare/LAPACK_qr_routine_comp.cpp -o CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.s

# Object files for target qr_routine_comp
qr_routine_comp_OBJECTS = \
"CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.o"

# External object files for target qr_routine_comp
qr_routine_comp_EXTERNAL_OBJECTS =

compare/qr_routine_comp: compare/CMakeFiles/qr_routine_comp.dir/LAPACK_qr_routine_comp.o
compare/qr_routine_comp: compare/CMakeFiles/qr_routine_comp.dir/build.make
compare/qr_routine_comp: libserial_tucker.so
compare/qr_routine_comp: compare/CMakeFiles/qr_routine_comp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/fangq18/TuckerMPI/src/serial/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable qr_routine_comp"
	cd /home/fangq18/TuckerMPI/src/serial/compare && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/qr_routine_comp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
compare/CMakeFiles/qr_routine_comp.dir/build: compare/qr_routine_comp

.PHONY : compare/CMakeFiles/qr_routine_comp.dir/build

compare/CMakeFiles/qr_routine_comp.dir/clean:
	cd /home/fangq18/TuckerMPI/src/serial/compare && $(CMAKE_COMMAND) -P CMakeFiles/qr_routine_comp.dir/cmake_clean.cmake
.PHONY : compare/CMakeFiles/qr_routine_comp.dir/clean

compare/CMakeFiles/qr_routine_comp.dir/depend:
	cd /home/fangq18/TuckerMPI/src/serial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fangq18/TuckerMPI/src/serial /home/fangq18/TuckerMPI/src/serial/compare /home/fangq18/TuckerMPI/src/serial /home/fangq18/TuckerMPI/src/serial/compare /home/fangq18/TuckerMPI/src/serial/compare/CMakeFiles/qr_routine_comp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : compare/CMakeFiles/qr_routine_comp.dir/depend

