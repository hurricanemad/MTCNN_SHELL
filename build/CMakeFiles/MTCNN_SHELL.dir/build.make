# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /home/dox/Algorithm/mtcnn-shell/mtcnn_shell

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build

# Include any dependencies generated for this target.
include CMakeFiles/MTCNN_SHELL.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/MTCNN_SHELL.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/MTCNN_SHELL.dir/flags.make

CMakeFiles/MTCNN_SHELL.dir/main.cpp.o: CMakeFiles/MTCNN_SHELL.dir/flags.make
CMakeFiles/MTCNN_SHELL.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/MTCNN_SHELL.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MTCNN_SHELL.dir/main.cpp.o -c /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/main.cpp

CMakeFiles/MTCNN_SHELL.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MTCNN_SHELL.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/main.cpp > CMakeFiles/MTCNN_SHELL.dir/main.cpp.i

CMakeFiles/MTCNN_SHELL.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MTCNN_SHELL.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/main.cpp -o CMakeFiles/MTCNN_SHELL.dir/main.cpp.s

CMakeFiles/MTCNN_SHELL.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/MTCNN_SHELL.dir/main.cpp.o.requires

CMakeFiles/MTCNN_SHELL.dir/main.cpp.o.provides: CMakeFiles/MTCNN_SHELL.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/MTCNN_SHELL.dir/build.make CMakeFiles/MTCNN_SHELL.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/MTCNN_SHELL.dir/main.cpp.o.provides

CMakeFiles/MTCNN_SHELL.dir/main.cpp.o.provides.build: CMakeFiles/MTCNN_SHELL.dir/main.cpp.o


# Object files for target MTCNN_SHELL
MTCNN_SHELL_OBJECTS = \
"CMakeFiles/MTCNN_SHELL.dir/main.cpp.o"

# External object files for target MTCNN_SHELL
MTCNN_SHELL_EXTERNAL_OBJECTS =

MTCNN_SHELL: CMakeFiles/MTCNN_SHELL.dir/main.cpp.o
MTCNN_SHELL: CMakeFiles/MTCNN_SHELL.dir/build.make
MTCNN_SHELL: lib/libMTCNN_FACE_DECT.so
MTCNN_SHELL: /home/dox/Algorithm/caffe/cbuild/lib/libcaffe.so.1.0.0
MTCNN_SHELL: /home/dox/Algorithm/caffe/cbuild/lib/libcaffeproto.a
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libboost_system.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libboost_thread.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libpthread.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libglog.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libgflags.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libsz.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libz.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libdl.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libm.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libpthread.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libglog.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libgflags.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libsz.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libz.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libdl.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libm.so
MTCNN_SHELL: /home/dox/protobuf/lib/libprotobuf.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/liblmdb.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libleveldb.so
MTCNN_SHELL: /usr/local/cuda/lib64/libcudart.so
MTCNN_SHELL: /usr/local/cuda/lib64/libcurand.so
MTCNN_SHELL: /usr/local/cuda/lib64/libcublas.so
MTCNN_SHELL: /usr/local/cuda/lib64/libcudnn.so
MTCNN_SHELL: /usr/lib/libopenblas.so
MTCNN_SHELL: /usr/lib/x86_64-linux-gnu/libboost_python.so
MTCNN_SHELL: /usr/local/lib/libopencv_cudabgsegm.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudaobjdetect.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudastereo.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_stitching.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudafeatures2d.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_superres.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudacodec.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_videostab.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudaoptflow.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudalegacy.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudawarping.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_photo.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudaimgproc.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudafilters.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_aruco.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_bgsegm.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_bioinspired.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_ccalib.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_dnn_modern.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_dpm.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_face.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_freetype.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_fuzzy.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_hdf.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_img_hash.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_line_descriptor.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_optflow.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_reg.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_rgbd.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_saliency.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_stereo.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_structured_light.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_phase_unwrapping.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_surface_matching.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_tracking.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_dnn.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_datasets.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_plot.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_text.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_ml.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_xfeatures2d.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudaarithm.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_shape.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_video.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_ximgproc.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_calib3d.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_features2d.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_flann.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_highgui.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_videoio.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_xobjdetect.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_objdetect.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_xphoto.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_imgproc.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_core.so.3.3.0
MTCNN_SHELL: /usr/local/lib/libopencv_cudev.so.3.3.0
MTCNN_SHELL: CMakeFiles/MTCNN_SHELL.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable MTCNN_SHELL"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MTCNN_SHELL.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/MTCNN_SHELL.dir/build: MTCNN_SHELL

.PHONY : CMakeFiles/MTCNN_SHELL.dir/build

CMakeFiles/MTCNN_SHELL.dir/requires: CMakeFiles/MTCNN_SHELL.dir/main.cpp.o.requires

.PHONY : CMakeFiles/MTCNN_SHELL.dir/requires

CMakeFiles/MTCNN_SHELL.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/MTCNN_SHELL.dir/cmake_clean.cmake
.PHONY : CMakeFiles/MTCNN_SHELL.dir/clean

CMakeFiles/MTCNN_SHELL.dir/depend:
	cd /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dox/Algorithm/mtcnn-shell/mtcnn_shell /home/dox/Algorithm/mtcnn-shell/mtcnn_shell /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/CMakeFiles/MTCNN_SHELL.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/MTCNN_SHELL.dir/depend
