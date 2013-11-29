#include "SegmenterHumanDet.h"
#include "superpixel.h"
#include "UtilsSuperpixel.h"
#include "UtilsSegmentation.h"
#include "UtilsOpencv.hpp"
#include "UtilsDetection.h"
#include "ProbaEstimatorNaiveBayesSmooth.h"
#include "ProbaEstimatorGMMOpencv.h"
#include "ClassifierProbaEstimate.h"

using namespace cv;
using namespace std;

void SegmenterHumanDet::segment(const cv::Mat& img, Mat_<uchar>& mask)
{
	Mat imgBGR;
	Mat imgLAB;
	Mat imgBGRo;

	double rate = 300.0/img.cols;

	GaussianBlur(img,imgBGRo,Size(),0.8,0.8);
	imgBGRo = img.clone();

	vector<Rect> faces;

	resize(imgBGRo,imgBGRo,Size(),rate,rate);

	imgBGR.create(imgBGRo.rows,imgBGRo.cols,CV_32FC3);
	imgBGRo.convertTo( imgBGR, CV_32F, 1.0/255. );
	cv::GaussianBlur(imgBGR,imgBGR,Size(),0.8,0.8);

	cvtColor( imgBGR, imgLAB, CV_BGR2Lab );

	Superpixel sp(1000,1,5);

	Mat_<int> segmentation = sp.segment(imgLAB);
	vector<SuperpixelStatistic> stat = sp.stat(imgLAB,imgBGR,segmentation);

	Mat_<float> prob;
	this->getPixelProbability(imgBGRo,prob,faces);
	Mat_<float> sprob;
	UtilsSuperpixel::Stat(segmentation,prob,stat,sprob);

	Mat_<int> initial(prob.rows,prob.cols);
	initial.setTo(1,prob>0.5);
	initial.setTo(0,prob<=0.5);
	Mat_<float> probaColor;
	int myx = cv::countNonZero(initial);
	Mat_<float> probaColorPixel;
	this->_getColorProba(imgBGR,initial,probaColorPixel);
	UtilsSuperpixel::Stat(segmentation,probaColorPixel,stat,probaColor);

	Mat_<float> fgdInit,bgdInit,fgdColor,bgdColor;
	this->_prob2energy(sprob,fgdInit,bgdInit);
	this->_prob2energy(probaColor,fgdColor,bgdColor);
	Mat_<float> fgdEnergy, bgdEnergy;

	fgdEnergy = 1.0*fgdInit + 1.0*fgdColor;
	bgdEnergy = 1.0*bgdInit + 1.0*bgdColor;

	Mat_<int> label;

	fgdEnergy *= 3;
	bgdEnergy *= 3;

	UtilsSegmentation::MaxFlowSuperpixel(stat,fgdEnergy,bgdEnergy,1.0,label);
	mask.create(imgBGRo.rows,imgBGRo.cols);

	for( int i=0;i<mask.rows;i++)
	{
		for(int j=0;j<mask.cols;j++)
		{
			if ( label(segmentation(i,j)) > 0.5)
			//if(prob(i,j)>0.5)
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

SegmenterHumanDet::SegmenterHumanDet(ADetector* p, const string& templateFile)
{
	this->_m_pDetector = p;
	UtilsOpencv::ReadTxt(templateFile,this->_m_matTemplate);
}

SegmenterHumanDet::~SegmenterHumanDet()
{
}

void SegmenterHumanDet::getPixelProbability(const cv::Mat& img, cv::Mat_<float>& prob, vector<Rect> faces)
{	
	vector<Rect> boxes;
	Mat_<float> confs;
	this->_m_pDetector->detectMultiScale(img,boxes,confs,0,0,0);
	
	//Mat_<float> tmp;
	//Rect r = UtilsDetection::FilterBoundingBox(img,boxes,confs, tmp);
	Rect r;
	if ( boxes.size() == 0)
	{
		r = Rect(0,0,img.cols,img.rows);
	}
	else
	{
		r = boxes[0];
	}

	prob.create(img.rows,img.cols);
	prob.setTo(0.0f);
	
	/*
	double rate= 80.0/r.width;

	Mat imgo;
	cv::resize(img,imgo,Size(),rate,rate);
	prob.create(imgo.rows,imgo.cols);
	prob.setTo(0.0f);

	Point centerDet(int((r.x+0.5*r.width)*rate),int((r.y+r.height*0.5)*rate));
	Point centerDict(this->_m_matTemplate.cols/2,this->_m_matTemplate.rows/2);

	for(int i=0;i<prob.rows;i++)
	{
		for(int j=0;j<prob.cols;j++)
		{
			int lx = j-centerDet.x+centerDict.x;
			int ly = i-centerDet.y+centerDict.y;

			if ( (lx<0)||(lx>=_m_matTemplate.cols)||(ly<0)||(ly>=_m_matTemplate.rows))
			{
				//do nothing
			}
			else
			{
				prob(i,j) = _m_matTemplate(ly,lx);
			}
		}
	}
	
	cv::resize(prob,prob,Size(img.cols,img.rows));
	GaussianBlur(prob,prob,Size(),0.8,0.8);
	*/

	//Rect inner = UtilsOpencv::ScaleRect(r,Rect(0,0,img.cols,img.rows),0.7,0.7,0.7,0.7);
	//Rect outer = UtilsOpencv::ScaleRect(r,Rect(0,0,img.cols,img.rows),1.2,1.2,1.2,1.2);

	//prob.setTo(0.0f);
	//prob(outer).setTo(0.4f);
	//prob(r).setTo(0.6f);
	//prob(inner).setTo(0.999f);
	prob.setTo(0.01f);
	prob(r).setTo(0.99f);

	//prob.setTo(0.9f,prob>0.9f);
	//prob.setTo(0.1f,prob<0.1f);
	int x = cv::countNonZero(prob>0.5);
	int y = 0;
}

void SegmenterHumanDet::_prob2energy(const cv::Mat_<float>& prob, cv::Mat_<float>& fgdEnergy, cv::Mat_<float>& bgdEnergy)
{
	fgdEnergy = prob.clone();
	bgdEnergy = prob.clone();

	cv::log(prob,fgdEnergy);
	cv::log(1-prob,bgdEnergy);
	fgdEnergy = -fgdEnergy;
	bgdEnergy = -bgdEnergy;
}

void SegmenterHumanDet::_getColorProba(const cv::Mat& img, const cv::Mat_<int>& label, cv::Mat_<float>& colorProba)
{
	Mat data = img.reshape(1,img.cols*img.rows);
	Mat la = label.reshape(1,label.cols*label.rows);

	ProbaEstimatorGMMOpencv probaEstimator;
	ClassifierProbaEstimate cls(&probaEstimator);

	cls.train(data,label);
	cls.predictProba(data,colorProba);

	colorProba = colorProba.reshape(1,img.rows);

	colorProba.setTo(0.9f,colorProba>0.9f);
	colorProba.setTo(0.1f,colorProba<0.1f);

	cv::GaussianBlur(colorProba,colorProba,Size(),0.8,0.8);
}
/*
void SegmenterHumanFaceTemplate::_getColorProba(const std::vector<SuperpixelStatistic>& spstat, const cv::Mat_<int>& label, cv::Mat_<float>& colorProba)
{
	typedef float TT;
	Mat_<TT> data((int)spstat.size(),3);

	for(int i=0;i<spstat.size();i++)
	{
		//data(i,0) = TT(spstat[i].mean_color_[0]*255/100);
		//data(i,1) = TT(spstat[i].mean_color_[1]+128);
		//data(i,2) = TT(spstat[i].mean_color_[2]+128);

		data(i,0) = TT(spstat[i].mean_rgb_[0]);
		data(i,1) = TT(spstat[i].mean_rgb_[1]);
		data(i,2) = TT(spstat[i].mean_rgb_[2]);
	}

	ProbaEstimatorGMMOpencv probaEstimator;
	ClassifierProbaEstimate cls(&probaEstimator);

	cls.train(data,label);
	cls.predictProba(data,colorProba);
}
*/