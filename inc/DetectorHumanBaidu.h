#ifndef _DETECTORHUMANBAIDU_H_
#define _DETECTORHUMANBAIDU_H_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "AImageDatasetForSegmentation.h"
#include "ADetector.h"
#include "AClassifier.h"

class DetectorHumanBaidu : public ADetector
{
public:
	virtual void detectSingleScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence){};
	virtual void detectMultiScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence, double scaleFactor
		, double maxSize, double minSize);

	virtual void save(const std::string& filename){};
	virtual void load(const std::string& filename){};

public:
	DetectorHumanBaidu(const std::string modelpath, ADetector* pBaseDetector);
	~DetectorHumanBaidu();

	void learnFromDataset( AImageDatasetForSegmentation* pDataset);

private:
	void _extractFeature(const cv::Mat& img, std::vector<cv::Rect> boxes, cv::Mat_<float> confs, cv::Mat_<float>& feature);
	void _extractLabel(const cv::Mat& img, std::vector<cv::Rect> boxes, const cv::Mat& mask, cv::Mat_<int>& label);

private:
	ADetector* _m_pBaseDetector;

	std::string _m_strModelPath;

	CvGBTreesParams _m_paramClassifier;
};

#endif