#include "DetectorLatentSVMOpencv.h"
#include "UtilsBoost.hpp"
#include "UtilsOpencv.hpp"

using namespace cv;
using namespace std;

DetectorLatentSVMOpencv::DetectorLatentSVMOpencv(const string& pathModel, int maxsize, float thresh)
{
	vector<boost::filesystem::path> files;
	UtilsBoost::DirRecursive(boost::filesystem::path(pathModel),files);

	_m_vecModelFilenames.clear();
	for(int i=0;i<files.size();i++)
	{
		_m_vecModelFilenames.push_back(files[i].string());
	}

	this->_m_intMaxSize = maxsize;
	this->_m_fltThresh = thresh;
}

void DetectorLatentSVMOpencv::detectMultiScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence, double scaleFactor
		, double maxSize, double minSize)
{
	double r = 1.0;
	Mat imgBGR = img.clone();
	if (img.cols > this->_m_intMaxSize)
	{
		r = double(_m_intMaxSize) / img.cols;

		cv::resize(imgBGR,imgBGR,Size(),r,r);
	}

	LatentSvmDetector detector(this->_m_vecModelFilenames);

	vector<LatentSvmDetector::ObjectDetection> detections;
	detector.detect(imgBGR,detections);

	vector<float> confs;
	
	for(int i=0;i<(int)detections.size();i++)
	{
		if ( detections[i].score > this->_m_fltThresh)
		{
			Rect rect = detections[i].rect;
			boxes.push_back(Rect(int(rect.x/r),int(rect.y/r),int(rect.width/r),int(rect.height/r)));
			confs.push_back(detections[i].score);
		}
	}

	confidence.create((int)confs.size(),1);
	for(int i=0;i<confs.size();i++)
	{
		confidence(i) = confs[i];
	}
}