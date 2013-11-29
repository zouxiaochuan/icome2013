tests-targets := $(DIR_BIN)/TestUtilsOpencv \
                 $(DIR_BIN)/TestLocalExtractorDenseSiftVl \
                 $(DIR_BIN)/TestImageDatasetBaidu \
                 $(DIR_BIN)/TestCodebookBuilderKmeansVl \
                 $(DIR_BIN)/TestLocalExtractorDenseSiftMultiScale \
                 $(DIR_BIN)/TestGlobalConfig \
                 $(DIR_BIN)/TestSegmenterBatchBaidu \
                 $(DIR_BIN)/TestDetectorBatchBaidu \
                 $(DIR_BIN)/cfg_test.json

all: $(tests-targets)

$(DIR_BIN)/TestUtilsOpencv : $(DIR_OBJ)/TestUtilsOpencv.o $(MODUL_OPENCV) $(MODUL_BOOST)
	$(CXX) -o $(@) $(^)  $(LDFLAGS_COMMON)

$(DIR_BIN)/TestLocalExtractorDenseSiftVl : $(DIR_OBJ)/TestLocalExtractorDenseSiftVl.o \
                                       $(DIR_OBJ)/LocalExtractorDenseSiftVl.o \
                                       $(DIR_OBJ)/UtilsMatlab.o \
                                       $(MODUL_OPENCV) \
                                       $(MODUL_BOOST) \
                                       $(MODUL_VLFEAT)
	$(CXX) -o $(@) $(^) $(LDFLAGS_COMMON)

$(DIR_BIN)/TestImageDatasetBaidu : $(DIR_OBJ)/TestImageDatasetBaidu.o \
                                   $(DIR_OBJ)/ImageDatasetBaidu.o \
                                   $(MODUL_OPENCV) \
                                   $(MODUL_BOOST)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TestCodebookBuilderKmeansVl : $(DIR_OBJ)/TestCodebookBuilderKmeansVl.o \
                                         $(DIR_OBJ)/common.a \
                                         $(MODUL_OPENCV) \
                                         $(MODUL_BOOST) \
                                         $(MODUL_VLFEAT)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TestLocalExtractorDenseSiftMultiScale : $(DIR_OBJ)/TestLocalExtractorDenseSiftMultiScale.o \
                                                   $(DIR_OBJ)/common.a \
                                                   $(MODUL_OPENCV) \
                                                   $(MODUL_BOOST) \
                                                   $(MODUL_VLFEAT)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TestGlobalConfig : $(DIR_OBJ)/TestGlobalConfig.o \
                              $(DIR_OBJ)/common.a \
                              $(MODUL_OPENCV) \
                              $(MODUL_BOOST) \
                              $(MODUL_VLFEAT) \
                              $(MODUL_LIBLINEAR) \
                              $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TestSegmenterBatchBaidu : $(DIR_OBJ)/TestSegmenterBatchBaidu.o \
                              $(DIR_OBJ)/common.a \
                              $(MODUL_OPENCV) \
                              $(MODUL_BOOST) \
                              $(MODUL_VLFEAT) \
                              $(MODUL_LIBLINEAR) \
                              $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TestDetectorBatchBaidu: $(DIR_OBJ)/TestDetectorBatchBaidu.o \
                              $(DIR_OBJ)/common.a \
                              $(MODUL_OPENCV) \
                              $(MODUL_BOOST) \
                              $(MODUL_VLFEAT) \
                              $(MODUL_LIBLINEAR) \
                              $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/cfg_test.json : $(DIR_CFG)/cfg_test.json
	cp -f $^ $@
