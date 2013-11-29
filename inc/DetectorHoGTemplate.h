#ifndef _DETECTORHOGTEMPLATE_H_
#define _DETECTORHOGTEMPLATE_H_

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "AImageDatasetForDetection.h"
#include "ADetector.h"
#include "AClassifier.h"

class DetectorHoGTemplate : public ADetector
{
public:
	virtual void detectSingleScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence);
	virtual void detectMultiScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence, double scaleFactor
		, double maxSize, double minSize);

	virtual void save(const std::string& filename);
	virtual void load(const std::string& filename);

public:
	DetectorHoGTemplate(int cellSize, int wndSize);
	~DetectorHoGTemplate();
	void learnFromDataset( AImageDatasetForDetection& dataset);


private:
	void _extractHoGSingle(const cv::Mat& img, cv::Mat_<float>& feature);
	void _extractHoGDense(const cv::Mat& img, cv::Mat_<float>& feature, cv::Mat_<float>& location);
	void _randSubImage(const cv::Mat& img, int height, int width, int n, std::vector<cv::Mat>& subimg);
	AClassifier* _getOptimalModel( const cv::Mat_<int>& label, const cv::Mat_<float>& data);

private:

	int _m_cellSize;
	int _m_wndSize;
	double _m_aspratio;

	int _m_wndWidth;
	int _m_wndHeight;

	AClassifier* _mp_classifier;

	double _m_minRate;
	double _m_maxRate;

	int _m_maxWidth;
	int _m_minWidth;

};

#endif