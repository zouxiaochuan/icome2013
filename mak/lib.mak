lib-targets: $(DIR_OBJ)/common.a

all: $(lib-targets)

$(DIR_OBJ)/common.a: $(DIR_OBJ)/GlobalConfig.o \
                     $(DIR_OBJ)/LocalExtractorDenseSiftMultiScale.o \
                     $(DIR_OBJ)/LocalExtractorDenseSiftVl.o \
                     $(DIR_OBJ)/CodebookBuilderKmeansVl.o \
                     $(DIR_OBJ)/UtilsMatlab.o \
                     $(DIR_OBJ)/UtilsValidate.o \
                     $(DIR_OBJ)/UtilsSegmentation.o \
                     $(DIR_OBJ)/UtilsSuperpixel.o \
                     $(DIR_OBJ)/UtilsDetection.o \
                     $(DIR_OBJ)/ImageDatasetBaidu.o \
                     $(DIR_OBJ)/ACodebookBuilder.o \
                     $(DIR_OBJ)/CodebookVQ.o \
                     $(DIR_OBJ)/AEncoder.o \
                     $(DIR_OBJ)/EncoderBoFSoft.o \
                     $(DIR_OBJ)/FeatureExtractorSPM.o \
                     $(DIR_OBJ)/PoolerMax.o \
                     $(DIR_OBJ)/PoolerAvg.o \
                     $(DIR_OBJ)/ClassifierLiblinear.o \
                     $(DIR_OBJ)/ADetector.o \
                     $(DIR_OBJ)/DetectorHoGTemplate.o \
                     $(DIR_OBJ)/DetectorLatentSVMOpencv.o \
                     $(DIR_OBJ)/DetectorHumanBaidu.o \
                     $(DIR_OBJ)/SegmenterHumanSimple.o \
                     $(DIR_OBJ)/SegmenterHumanFaceTemplate.o \
                     $(DIR_OBJ)/SegmenterHumanFaceTemplateGrabcut.o \
                     $(DIR_OBJ)/SegmenterHumanDet.o \
                     $(DIR_OBJ)/superpixel.o \
                     $(DIR_OBJ)/ClassifierProbaEstimate.o \
                     $(DIR_OBJ)/ProbaEstimatorNaiveBayesSmooth.o \
                     $(DIR_OBJ)/ProbaEstimatorGMMOpencv.o
	ar rvs $@ $^
