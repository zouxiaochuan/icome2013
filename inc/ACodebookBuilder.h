#ifndef _ACODEBOOKBUILDER_H_
#define _ACODEBOOKBUILDER_H_

#include "ACodebook.h"
#include "AImageDataset.h"
#include "ALocalExtractor.h"

#include <opencv2/opencv.hpp>

using namespace cv;

class ACodebookBuilder
{
public:
	virtual void build(const Mat_<float> data, ACodebook& cb) = 0;

public:
	void buildFromDataset(ACodebook& cb, const AImageDataset& dataset, ALocalExtractor& extractor, int trainingNum);
};


#endif