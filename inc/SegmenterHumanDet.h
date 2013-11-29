#ifndef _SEGMENTERHUMANDET_H_
#define _SEGMENTERHUMANDET_H_

#include <opencv2/opencv.hpp>
#include "superpixel.h"
#include "ASegmenter.h"
#include "ADetector.h"

class SegmenterHumanDet : public ASegmenter
{
public:
	virtual void segment(const cv::Mat& img, cv::Mat_<uchar>& mask);
	void getPixelProbability(const cv::Mat& img, cv::Mat_<float>& prob, std::vector<cv::Rect> faces);

public:
	SegmenterHumanDet(ADetector *pDetector, const std::string& templateFile);
	~SegmenterHumanDet();

private:
	 void _prob2energy(const cv::Mat_<float>& prob, cv::Mat_<float>& fgdEnergy, cv::Mat_<float>& bgdEnergy);
	 void _getColorProba(const cv::Mat& img, const cv::Mat_<int>& label, cv::Mat_<float>& colorProba);

private:

	ADetector* _m_pDetector;
	Mat_<float> _m_matTemplate;

};

#endif