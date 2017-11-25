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
include lib/CMakeFiles/MTCNN_FACE_DECT.dir/depend.make

# Include the progress variables for this target.
include lib/CMakeFiles/MTCNN_FACE_DECT.dir/progress.make

# Include the compile flags for this target's objects.
include lib/CMakeFiles/MTCNN_FACE_DECT.dir/flags.make

lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o: lib/CMakeFiles/MTCNN_FACE_DECT.dir/flags.make
lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o: ../src/detector.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o"
	cd /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/lib && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o -c /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/src/detector.cpp

lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.i"
	cd /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/lib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/src/detector.cpp > CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.i

lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.s"
	cd /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/lib && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/src/detector.cpp -o CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.s

lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o.requires:

.PHONY : lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o.requires

lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o.provides: lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o.requires
	$(MAKE) -f lib/CMakeFiles/MTCNN_FACE_DECT.dir/build.make lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o.provides.build
.PHONY : lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o.provides

lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o.provides.build: lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o


# Object files for target MTCNN_FACE_DECT
MTCNN_FACE_DECT_OBJECTS = \
"CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o"

# External object files for target MTCNN_FACE_DECT
MTCNN_FACE_DECT_EXTERNAL_OBJECTS =

lib/libMTCNN_FACE_DECT.so: lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o
lib/libMTCNN_FACE_DECT.so: lib/CMakeFiles/MTCNN_FACE_DECT.dir/build.make
lib/libMTCNN_FACE_DECT.so: /home/dox/Algorithm/caffe/cbuild/lib/libcaffe.so.1.0.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudabgsegm.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudaobjdetect.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudastereo.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_stitching.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_superres.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_videostab.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_aruco.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_bgsegm.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_bioinspired.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_ccalib.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_dnn_modern.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_dpm.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_face.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_freetype.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_fuzzy.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_hdf.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_img_hash.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_line_descriptor.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_optflow.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_reg.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_rgbd.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_saliency.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_stereo.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_structured_light.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_surface_matching.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_tracking.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_xfeatures2d.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_ximgproc.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_xobjdetect.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_xphoto.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /home/dox/Algorithm/caffe/cbuild/lib/libcaffeproto.a
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libboost_system.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libboost_thread.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libglog.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libgflags.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libsz.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libz.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libdl.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libm.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libpthread.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libglog.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libgflags.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libsz.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libz.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libdl.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libm.so
lib/libMTCNN_FACE_DECT.so: /home/dox/protobuf/lib/libprotobuf.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/liblmdb.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libleveldb.so
lib/libMTCNN_FACE_DECT.so: /usr/local/cuda/lib64/libcudart.so
lib/libMTCNN_FACE_DECT.so: /usr/local/cuda/lib64/libcurand.so
lib/libMTCNN_FACE_DECT.so: /usr/local/cuda/lib64/libcublas.so
lib/libMTCNN_FACE_DECT.so: /usr/local/cuda/lib64/libcudnn.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/libopenblas.so
lib/libMTCNN_FACE_DECT.so: /usr/lib/x86_64-linux-gnu/libboost_python.so
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudafeatures2d.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_shape.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudacodec.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudaoptflow.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudalegacy.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudawarping.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_photo.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudaimgproc.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudafilters.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudaarithm.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_calib3d.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_phase_unwrapping.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_dnn.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_video.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_datasets.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_plot.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_text.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_features2d.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_flann.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_highgui.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_ml.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_videoio.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_imgcodecs.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_objdetect.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_imgproc.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_core.so.3.3.0
lib/libMTCNN_FACE_DECT.so: /usr/local/lib/libopencv_cudev.so.3.3.0
lib/libMTCNN_FACE_DECT.so: lib/CMakeFiles/MTCNN_FACE_DECT.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libMTCNN_FACE_DECT.so"
	cd /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/lib && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/MTCNN_FACE_DECT.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
lib/CMakeFiles/MTCNN_FACE_DECT.dir/build: lib/libMTCNN_FACE_DECT.so

.PHONY : lib/CMakeFiles/MTCNN_FACE_DECT.dir/build

lib/CMakeFiles/MTCNN_FACE_DECT.dir/requires: lib/CMakeFiles/MTCNN_FACE_DECT.dir/detector.cpp.o.requires

.PHONY : lib/CMakeFiles/MTCNN_FACE_DECT.dir/requires

lib/CMakeFiles/MTCNN_FACE_DECT.dir/clean:
	cd /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/lib && $(CMAKE_COMMAND) -P CMakeFiles/MTCNN_FACE_DECT.dir/cmake_clean.cmake
.PHONY : lib/CMakeFiles/MTCNN_FACE_DECT.dir/clean

lib/CMakeFiles/MTCNN_FACE_DECT.dir/depend:
	cd /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dox/Algorithm/mtcnn-shell/mtcnn_shell /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/src /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/lib /home/dox/Algorithm/mtcnn-shell/mtcnn_shell/build/lib/CMakeFiles/MTCNN_FACE_DECT.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : lib/CMakeFiles/MTCNN_FACE_DECT.dir/depend
