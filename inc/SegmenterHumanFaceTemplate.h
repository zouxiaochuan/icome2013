#ifndef _SEGMENTERHUMANFACETEMPLATE_H_
#define _SEGMENTERHUMANFACETEMPLATE_H_

#include <opencv2/opencv.hpp>
#include "superpixel.h"
#include "ASegmenter.h"

class SegmenterHumanFaceTemplate : public ASegmenter
{
public:
	virtual void segment(const cv::Mat& img, cv::Mat_<uchar>& mask);
	void getPixelProbability(const cv::Mat& img, cv::Mat_<float>& prob, std::vector<cv::Rect> faces);

public:
	SegmenterHumanFaceTemplate(const std::string& mfilename, const std::string& templateFilename);
	~SegmenterHumanFaceTemplate();

private:
	 void _prob2energy(const cv::Mat_<float>& prob, cv::Mat_<float>& fgdEnergy, cv::Mat_<float>& bgdEnergy);
	 void _getColorProba(const cv::Mat& img, const cv::Mat_<int>& label, cv::Mat_<float>& colorProba);

private:

	std::string _m_filenameFaceModel;
	cv::Mat_<float> _m_matFaceTemplate;
	cv::Rect _m_rectDefaultFace;
	cv::Vec4f _m_vec4fDefaultFace;

};

#endif