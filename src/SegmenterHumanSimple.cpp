#include "SegmenterHumanSimple.h"
#include "superpixel.h"
#include "UtilsSuperpixel.h"
#include "UtilsSegmentation.h"
#include "UtilsOpencv.hpp"
#include "ProbaEstimatorNaiveBayesSmooth.h"
#include "ClassifierProbaEstimate.h"

using namespace cv;
using namespace std;

void SegmenterHumanSimple::segment(const cv::Mat& img, Mat_<uchar>& mask)
{
	Mat imgBGR;
	Mat imgLAB;
	Mat imgBGRo;

	float rate = 500.0f/img.cols;

	GaussianBlur(img,imgBGRo,Size(),0.8,0.8);

	vector<Rect> faces;

	resize(imgBGRo,imgBGRo,Size(),rate,rate);
	cv::CascadeClassifier faceModel(this->_m_filenameFaceModel);
	faceModel.detectMultiScale(imgBGRo,faces);

	imgBGRo.convertTo( imgBGR, CV_32F, 1.0/255. );

	cvtColor( imgBGR, imgLAB, CV_BGR2Lab );

	Superpixel sp(1000,1,5);

	Mat_<int> segmentation = sp.segment(imgLAB);
	vector<SuperpixelStatistic> stat = sp.stat(imgLAB,imgBGR,segmentation);

	Mat_<float> prob;
	this->getPixelProbability(imgBGRo,prob,faces);
	Mat_<float> sprob;
	UtilsSuperpixel::Stat(segmentation,prob,stat,sprob);

	Mat_<int> initial(int(stat.size()),1);
	initial.setTo(1,sprob>0.5);
	initial.setTo(0,sprob<=0.5);
	Mat_<float> probaColor;
	int myx = cv::countNonZero(initial);
	this->_getColorProba(stat,initial,probaColor);

	Mat_<float> fgdInit,bgdInit,fgdColor,bgdColor;
	this->_prob2energy(sprob,fgdInit,bgdInit);
	this->_prob2energy(probaColor,fgdColor,bgdColor);
	Mat_<float> fgdEnergy, bgdEnergy;
	
	fgdEnergy = fgdInit + fgdColor;
	bgdEnergy = bgdInit + bgdColor;

	Mat_<int> label;
	mask.create(imgBGRo.rows,imgBGRo.cols);

	UtilsSegmentation::MaxFlowSuperpixel(stat,fgdEnergy,bgdEnergy,50.0,label);

	for( int i=0;i<mask.rows;i++)
	{
		for(int j=0;j<mask.cols;j++)
		{
			if ( label(segmentation(i,j)) > 0.5)
			{
				mask(i,j) = 255;
			}
			else
			{
				mask(i,j) = 0;
			}
		}
	}

	cv::resize(mask,mask,Size(img.cols,img.rows));
	mask.setTo(255,mask>128);
	mask.setTo(0,mask<=128);
}

SegmenterHumanSimple::SegmenterHumanSimple(const string& mfilename)
{
	this->_m_filenameFaceModel = mfilename;
}

SegmenterHumanSimple::~SegmenterHumanSimple()
{
}

void SegmenterHumanSimple::getPixelProbability(const cv::Mat& img, cv::Mat_<float>& prob, vector<Rect> faces)
{
	prob.create(img.rows,img.cols);
	prob.setTo(0.3f);
	for(int i=0;i<faces.size();i++)
	{
		Rect rect = faces[i];
		rect = UtilsOpencv::ScaleRect(rect,Rect(0,0,img.cols,img.rows),1.2,1.5,1.2,1.2);
		prob(Rect(rect.x,rect.y,rect.width,img.rows-rect.y)) = 0.7f;
	}
	
}

void SegmenterHumanSimple::_prob2energy(const cv::Mat_<float>& prob, cv::Mat_<float>& fgdEnergy, cv::Mat_<float>& bgdEnergy)
{
	fgdEnergy = prob.clone();
	bgdEnergy = prob.clone();

	cv::log(prob,fgdEnergy);
	cv::log(1-prob,bgdEnergy);
	fgdEnergy = -fgdEnergy;
	bgdEnergy = -bgdEnergy;
}

void SegmenterHumanSimple::_getColorProba(const std::vector<SuperpixelStatistic>& spstat, const cv::Mat_<int>& label, cv::Mat_<float>& colorProba)
{
	Mat_<uchar> data((int)spstat.size(),3);

	for(int i=0;i<spstat.size();i++)
	{
		data(i,0) = uchar(spstat[i].mean_color_[0]*255/100);
		data(i,1) = uchar(spstat[i].mean_color_[1]+128);
		data(i,2) = uchar(spstat[i].mean_color_[2]+128);
	}

	ProbaEstimatorNaiveBayesSmooth probaEstimator(256,0.8);
	ClassifierProbaEstimate cls(&probaEstimator);

	cls.train(data,label);
	cls.predictProba(data,colorProba);
}