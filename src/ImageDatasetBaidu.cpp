#include "ImageDatasetBaidu.h"
#include "ConstantsBaidu.h"
#include "UtilsOpencv.hpp"

#include <string>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <fstream>
#include <vector>

using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace boost::algorithm;

ImageDatasetBaidu::ImageDatasetBaidu(const std::string& root)
{
	path pathLabelFile(root);
	pathLabelFile /= ::_PATH_BAIDU_TRAINING_LABEL_FILE;

	ifstream in(pathLabelFile.string().c_str());
	if ( in.fail() || in.bad())
	{
		throw std::runtime_error("cannnot read baidu file");
	}

	vector<int> labels;
	this->_m_vecFilenames.clear();

	do
	{
		int l;
		std::string f;
		in >> f >> l;
		if( f.length() > 0)
		{
			labels.push_back(l);
			this->_m_vecFilenames.push_back(f);
		}
	}while(!in.eof());

	this->_m_matLabels = Mat_<int>(labels, true);

	path pathImage(root);
	pathImage /= ::_PATH_BAIDU_TRAINING_IMAGE;
	this->_m_pathImage = pathImage.string();

	in.close();

	path pathMask(root);
	pathMask /= ::_PATH_BAIDU_TRAINING_MASK;
	this->_m_pathMask = pathMask.string();
}

Mat ImageDatasetBaidu::getImage(int idx) const
{
	string filename = this->_m_vecFilenames[idx];

	path pathImage(this->_m_pathImage);
	pathImage /= filename;
	
	return imread(pathImage.string());
}

int ImageDatasetBaidu::getSize() const
{
	return this->_m_matLabels.rows*this->_m_matLabels.cols;
}

string ImageDatasetBaidu::getRawName(int idx) const
{
	std::string rawname = this->_m_vecFilenames[idx];
	boost::algorithm::replace_last(rawname,".jpg","");
	return rawname;
}

Mat_<int> ImageDatasetBaidu::getLabels()
{
	return this->_m_matLabels;
}

void ImageDatasetBaidu::getBoundingBox(int idx, vector<Rect>& boxes)
{
	Mat mask = this->getMask(idx);
	if ( mask.empty())
	{
		return;
	}

	Rect box = UtilsOpencv::BoundingBoxFromMask(mask);

	box = UtilsOpencv::ScaleRect(box,Rect(0,0,mask.cols,mask.rows),1.2,1.2,1.2,1.2);

	boxes.push_back(box);

	return;
}

Mat ImageDatasetBaidu::getMask(int idx)
{
	string name = this->_m_vecFilenames[idx];

	boost::algorithm::replace_last(name,".","-profile.");
	path pathMaskFile(this->_m_pathMask);
	pathMaskFile /= name;

	if ( !boost::filesystem::exists(pathMaskFile))
	{
		return Mat();
	}

	Mat img = cv::imread(pathMaskFile.string());
	Mat imgraw = this->getImage(idx);

	if ( imgraw.empty())
	{
		return Mat();
	}

	cv::resize(img,img,Size(imgraw.cols,imgraw.rows));

	cv::cvtColor(img,img,CV_BGR2GRAY);

	cv::threshold(img,img,100,255,THRESH_BINARY);

	img = 255 - img;

	return img;
}
