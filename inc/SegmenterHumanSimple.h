#ifndef _SEGMENTERHUMANSIMPLE_H_
#define _SEGMENTERHUMANSIMPLE_H_

#include <opencv2/opencv.hpp>
#include "superpixel.h"
#include "ASegmenter.h"

class SegmenterHumanSimple : public ASegmenter
{
public:
	virtual void segment(const cv::Mat& img, cv::Mat_<uchar>& mask);
	void getPixelProbability(const cv::Mat& img, cv::Mat_<float>& prob, std::vector<cv::Rect> faces);

public:
	SegmenterHumanSimple(const std::string& mfilename);
	~SegmenterHumanSimple();

private:
	 void _prob2energy(const cv::Mat_<float>& prob, cv::Mat_<float>& fgdEnergy, cv::Mat_<float>& bgdEnergy);
	 void _getColorProba(const std::vector<SuperpixelStatistic>& spstat, const cv::Mat_<int>& label, cv::Mat_<float>& colorProba);

private:

	std::string _m_filenameFaceModel;
};

#endif