#ifndef _AENCODER_H_
#define _AENCODER_H_

#include <opencv2/opencv.hpp>
#include "ALocalExtractor.h"
#include <string>

class AEncoder
{
public:
	virtual void encode(const cv::Mat_<float>& descriptors, cv::Mat_<float>& encoded) = 0;
	virtual int codeDimension() const = 0;

public:

	void encodeFromDirectory(const std::string& pathImage, ALocalExtractor& extractor, const std::string& pathOutput);
};

#endif