DIR_OBJ := build/obj
DIR_3RDPARTY := 3rdparty
DIR_SRC := src
DIR_INC_PROJ := inc
DIR_BIN := build/bin
DIR_CFG := cfg
DIR_SCRIPT := sh
DIR_OPENCV := $(DIR_3RDPARTY)/opencv-2.4.5
DIR_INC_OPENCV := $(DIR_OPENCV)/include
DIR_LIB_OPENCV := $(DIR_OPENCV)/lib
MODUL_OPENCV := opencv_objdetect opencv_ml opencv_highgui opencv_imgproc opencv_core
MODUL_OPENCV := $(foreach i, $(MODUL_OPENCV), $(DIR_LIB_OPENCV)/lib$i.a)
MODUL_OPENCV_DEP := libjpeg libjasper libpng libtiff IlmImf zlib
MODUL_OPENCV_DEP := $(foreach i, $(MODUL_OPENCV_DEP), $(DIR_OPENCV)/share/OpenCV/3rdparty/lib/lib$i.a)
MODUL_OPENCV := $(MODUL_OPENCV) $(MODUL_OPENCV_DEP)

DIR_VLFEAT := $(DIR_3RDPARTY)/vlfeat-0.9.16-c
DIR_INC_VLFEAT := $(DIR_VLFEAT)/vl

DIR_BOOST := $(DIR_3RDPARTY)/boost_1_54_0
DIR_INC_BOOST := $(DIR_BOOST)/include
DIR_LIB_BOOST := $(DIR_BOOST)/lib
MODUL_BOOST := filesystem log_setup log thread  system
MODUL_BOOST := $(foreach i, $(MODUL_BOOST), $(DIR_LIB_BOOST)/libboost_$i.a)

MODUL_VLFEAT := $(DIR_VLFEAT)/libvlfeat.a

DIR_LIBLINEAR := $(DIR_3RDPARTY)/liblinear-1.92
DIR_INC_LIBLINEAR := $(DIR_LIBLINEAR)
MODUL_LIBLINEAR := $(DIR_LIBLINEAR)/linear.o $(DIR_LIBLINEAR)/tron.o $(DIR_LIBLINEAR)/blas/blas.a

DIR_MAXFLOW := $(DIR_3RDPARTY)/maxflow-v3.02.src
DIR_INC_MAXFLOW := $(DIR_MAXFLOW)
MODUL_MAXFLOW := $(DIR_MAXFLOW)/libmaxflow.a

DIR_INCS := $(DIR_INC_PROJ) $(DIR_INC_VLFEAT) $(DIR_INC_OPENCV) $(DIR_INC_BOOST) $(DIR_INC_LIBLINEAR) $(DIR_INC_MAXFLOW)

CFLAGS_COMMON := $(foreach d, $(DIR_INCS), -I$d) -pthread -fopenmp -O2 -std=c++11
LDFLAGS_COMMON := $(MODUL_COMMON) -pthread -fopenmp

include mak/ext.mak
include mak/obj.mak
include mak/lib.mak
include mak/tests.mak
include mak/release.mak

all:
	@echo ***BUILD SUCCESS***

clean:
	rm -f $(DIR_OBJ)/*
	rm -f $(DIR_BIN)/*


no_dep_targets += clean archclean distclean info help

ifeq ($(filter $(no_dep_targets), $(MAKECMDGOALS)),)
-include $(deps)
endif
