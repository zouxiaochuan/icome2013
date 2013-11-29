release-targets := $(DIR_BIN)/BuildCodebookBaidu \
                   $(DIR_BIN)/EncodeBaidu \
                   $(DIR_BIN)/ExtractFeatureBaidu \
                   $(DIR_BIN)/TestClassifierBaidu \
                   $(DIR_BIN)/TrainClassifierBaidu \
                   $(DIR_BIN)/TestResultBaidu \
                   $(DIR_BIN)/TestResultSegmentation \
                   $(DIR_BIN)/TrainDetectorBaidu \
                   $(DIR_BIN)/GenerateFaceTemplateBaidu \
                   $(DIR_BIN)/CountBoundingBoxStatistic \
                   $(DIR_BIN)/testsegmentation.sh \
                   $(DIR_BIN)/testclassification.sh

all : $(release-targets)

$(DIR_BIN)/BuildCodebookBaidu : $(DIR_OBJ)/BuildCodebookBaidu.o \
                                $(DIR_OBJ)/common.a \
                                $(MODUL_LIBLINEAR) \
                                $(MODUL_OPENCV) \
                                $(MODUL_BOOST) \
                                $(MODUL_VLFEAT) \
                                $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/EncodeBaidu : $(DIR_OBJ)/EncodeBaidu.o \
                         $(DIR_OBJ)/common.a \
                         $(MODUL_LIBLINEAR) \
                         $(MODUL_OPENCV) \
                         $(MODUL_BOOST) \
                         $(MODUL_VLFEAT) \
                         $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/ExtractFeatureBaidu: $(DIR_OBJ)/ExtractFeatureBaidu.o \
                                $(DIR_OBJ)/common.a \
                                $(MODUL_LIBLINEAR) \
                                $(MODUL_OPENCV) \
                                $(MODUL_BOOST) \
                                $(MODUL_VLFEAT) \
                                $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TestClassifierBaidu: $(DIR_OBJ)/TestClassifierBaidu.o \
                                $(DIR_OBJ)/common.a \
                                $(MODUL_OPENCV) \
                                $(MODUL_BOOST) \
                                $(MODUL_VLFEAT) \
                                $(MODUL_LIBLINEAR) \
                                $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TrainClassifierBaidu: $(DIR_OBJ)/TrainClassifierBaidu.o \
                                 $(DIR_OBJ)/common.a \
                                 $(MODUL_OPENCV) \
                                 $(MODUL_BOOST) \
                                 $(MODUL_VLFEAT) \
                                 $(MODUL_LIBLINEAR) \
                                 $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TestResultBaidu: $(DIR_OBJ)/TestResultBaidu.o \
                            $(DIR_OBJ)/common.a \
                            $(MODUL_OPENCV) \
                            $(MODUL_BOOST) \
                            $(MODUL_VLFEAT) \
                            $(MODUL_LIBLINEAR) \
                            $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TrainDetectorBaidu: $(DIR_OBJ)/TrainDetectorBaidu.o \
                               $(DIR_OBJ)/common.a \
                               $(MODUL_OPENCV) \
                               $(MODUL_BOOST) \
                               $(MODUL_VLFEAT) \
                               $(MODUL_LIBLINEAR) \
                               $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/GenerateFaceTemplateBaidu: $(DIR_OBJ)/GenerateFaceTemplateBaidu.o \
                               $(DIR_OBJ)/common.a \
                               $(MODUL_OPENCV) \
                               $(MODUL_BOOST) \
                               $(MODUL_VLFEAT) \
                               $(MODUL_LIBLINEAR) \
                               $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)
	
$(DIR_BIN)/CountBoundingBoxStatistic: $(DIR_OBJ)/CountBoundingBoxStatistic.o \
                               $(DIR_OBJ)/common.a \
                               $(MODUL_OPENCV) \
                               $(MODUL_BOOST) \
                               $(MODUL_VLFEAT) \
                               $(MODUL_LIBLINEAR) \
                               $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/TestResultSegmentation: $(DIR_OBJ)/TestResultSegmentation.o \
                               $(DIR_OBJ)/common.a \
                               $(MODUL_OPENCV) \
                               $(MODUL_BOOST) \
                               $(MODUL_VLFEAT) \
                               $(MODUL_LIBLINEAR) \
                               $(MODUL_MAXFLOW)
	$(CXX) -o $@ $^ $(LDFLAGS_COMMON)

$(DIR_BIN)/testsegmentation.sh: $(DIR_SCRIPT)/testsegmentation.sh
	cp -f $^ $@

$(DIR_BIN)/testclassification.sh: $(DIR_SCRIPT)/testclassification.sh
	cp -f $^ $@