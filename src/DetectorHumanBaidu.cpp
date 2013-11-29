#include "DetectorHumanBaidu.h"
#include "UtilsDetection.h"
#include "UtilsOpencv.hpp"
#include <boost/log/trivial.hpp>
#include <map>

using namespace cv;
using namespace std;

DetectorHumanBaidu::DetectorHumanBaidu(const std::string modelpath, ADetector* pBase)
{
	this->_m_strModelPath = modelpath;
	this->_m_paramClassifier = CvGBTreesParams(CvGBTrees::SQUARED_LOSS,200,0.8f,0.01f,3,false);
	this->_m_pBaseDetector = pBase;
}

DetectorHumanBaidu::~DetectorHumanBaidu()
{
}

void DetectorHumanBaidu::detectMultiScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence, double scaleFactor
		, double maxSize, double minSize)
{
	vector<Rect> rects;
	Mat_<float> scores;
	this->_m_pBaseDetector->detectMultiScale(img,rects,scores,scaleFactor,maxSize,minSize);
	Mat_<float> feature;

	this->_extractFeature(img,rects,scores,feature);

	//cv::GradientBoostingTrees cls;
	cv::RandomTrees cls;
	cls.load(this->_m_strModelPath.c_str());

	map<float,int> smap;
	for(int i=0;i<feature.rows;i++)
	{
		float pre = cls.predict_prob(feature.row(i));
		if ( smap.find(pre) == smap.end())
		{
			smap[pre] = i;
		}
	}

	confidence.create((int)rects.size(),1);
	int cnt = 0;
	for(map<float,int>::reverse_iterator i=smap.rbegin();i!=smap.rend();i++)
	{
		boxes.push_back(rects[i->second]);
		confidence(cnt++) = i->first;
	}
}

void DetectorHumanBaidu::learnFromDataset( AImageDatasetForSegmentation* pDataset)
{
	int size = pDataset->getSize();

	vector<Mat_<float> > vecFeatures;
	vector<Mat_<int> > vecLabels;

	BOOST_LOG_TRIVIAL(debug)  << "begin learning detector: ";

#pragma omp parallel for
	for( int i=0;i<size;i++)
	{
		Mat mask = pDataset->getMask(i);

		if ( !mask.empty())
		{
			Mat img = pDataset->getImage(i);

			if ( img.empty())
			{
				continue;
			}

			vector<Rect> boxes;
			Mat_<float> confs;
			this->_m_pBaseDetector->detectMultiScale(img,boxes,confs,0,0,0);

			if ( boxes.size() == 0)
			{
				BOOST_LOG_TRIVIAL(debug)  << "no detection, idx : " << i;
				continue;
			}

			BOOST_LOG_TRIVIAL(debug)  << "process idx : " << i;
			Mat_<float> fea;
			Mat_<int> lab;

			this->_extractFeature(img,boxes,confs,fea);
			this->_extractLabel(img,boxes,mask,lab);

#pragma omp critical
			{
			vecFeatures.push_back(fea);
			vecLabels.push_back(lab);
			}
		}
	}

	Mat_<float> feature;
	Mat_<int> label;
	UtilsOpencv::MergeRows(vecFeatures,feature);
	UtilsOpencv::MergeRows(vecLabels,label);

	BOOST_LOG_TRIVIAL(debug)  << "train classifier, data size: [" << feature.rows << "," << feature.cols << "]";

	//cv::GradientBoostingTrees cls;
	cv::RandomTreeParams params( 5, 10, 0, false, 10, NULL, false, 0,200,0.0f, CV_TERMCRIT_ITER );
	cv::RandomTrees cls;
	cls.train(feature,CV_ROW_SAMPLE,label,Mat(),Mat(),Mat(),Mat(),params);
	cls.save(this->_m_strModelPath.c_str());
}

void DetectorHumanBaidu::_extractFeature(const cv::Mat& img, std::vector<Rect> boxes, 
	cv::Mat_<float> confs, cv::Mat_<float>& feature)
{
	feature.create((int)boxes.size(),8);

	for(int i=0;i<boxes.size();i++)
	{
		Rect box = boxes[i];
		float conf = confs(i);

		feature(i,0) = float(box.x)/img.cols;
		feature(i,1) = float(box.x+box.width) / img.cols;
		feature(i,2) = float(box.y)/img.rows;
		feature(i,3) = float(box.y+box.height)/img.rows;
		feature(i,4) = feature(i,1)-feature(i,0);
		feature(i,5) = feature(i,3)-feature(i,2);
		feature(i,6) = feature(i,4)*feature(i,5);
		feature(i,7) = conf;
	}
}

void DetectorHumanBaidu::_extractLabel(const cv::Mat& img, 
	std::vector<cv::Rect> boxes, const cv::Mat& mask, cv::Mat_<int>& label)
{
	Mat_<float> scores;
	UtilsDetection::RankDetection(boxes,mask,scores);

	label.create((int)boxes.size(),1);
	label.setTo(0);

	float maxscore = 0.0;
	int maxidx = 0;
	for(int i=0;i<(int)scores.total();i++)
	{
		if ( scores(i) > maxscore)
		{
			maxscore = scores(i);
			maxidx = i;
		}
	}

	label(maxidx) = 1;
}