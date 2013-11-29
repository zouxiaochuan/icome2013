#ifndef _DETECTORLATENTSVMOPENCV_H_
#define _DETECTORLATENTSVMOPENCV_H_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "AImageDatasetForDetection.h"
#include "ADetector.h"
#include "AClassifier.h"

class DetectorLatentSVMOpencv : public ADetector
{
public:
	virtual void detectSingleScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence){};
	virtual void detectMultiScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence, double scaleFactor
		, double maxSize, double minSize);

	virtual void save(const std::string& filename){};
	virtual void load(const std::string& filename){};

public:
	DetectorLatentSVMOpencv(const std::string& pathModel, int maxsize, float thresh);
	~DetectorLatentSVMOpencv(){};

private:

private:
	std::vector<std::string> _m_vecModelFilenames;
	int _m_intMaxSize;
	float _m_fltThresh;

};

#endif