#include "SegmenterHumanFaceTemplateGrabcut.h"
#include "superpixel.h"
#include "UtilsSuperpixel.h"
#include "UtilsSegmentation.h"
#include "UtilsOpencv.hpp"
#include "ProbaEstimatorNaiveBayesSmooth.h"
#include "ProbaEstimatorGMMOpencv.h"
#include "ClassifierProbaEstimate.h"

using namespace cv;
using namespace std;

void SegmenterHumanFaceTemplateGrabcut::segment(const cv::Mat& img, Mat_<uchar>& mask)
{
	Mat imgBGR;
	Mat imgLAB;
	Mat imgBGRo;

	float rate = 300.0f/img.cols;

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

	Mat_<int> initial(prob.rows,prob.cols);
	initial.setTo(1,prob>0.5);
	initial.setTo(0,prob<=0.5);

	mask.create(prob.rows,prob.cols);
	mask.setTo(cv::GC_PR_BGD,prob<=0.5);
	mask.setTo(cv::GC_PR_FGD,prob>0.5);

	Mat bgdModel,fgdModel;
	cv::grabCut(imgBGRo,mask,Rect(),bgdModel,fgdModel,1,GC_INIT_WITH_MASK);
	mask.setTo(255,(mask==GC_FGD | mask==GC_PR_FGD));
	mask.setTo(0,(mask==GC_BGD | mask==GC_PR_BGD));

	cv::resize(mask,mask,Size(img.cols,img.rows));
	mask.setTo(255,mask>128);
	mask.setTo(0,mask<=128);
}

SegmenterHumanFaceTemplateGrabcut::SegmenterHumanFaceTemplateGrabcut(const string& mfilename, const string& templateFilename)
{
	this->_m_filenameFaceModel = mfilename;

	ifstream in(templateFilename);
	in >> this->_m_vec4fDefaultFace[0] >> this->_m_vec4fDefaultFace[1] >> this->_m_vec4fDefaultFace[2] >> this->_m_vec4fDefaultFace[3];
	
	int templateRows;
	int templateCols;

	in >> templateCols >> templateRows;

	this->_m_matFaceTemplate.create(templateRows,templateCols);
	UtilsOpencv::ReadTxt(in,this->_m_matFaceTemplate,templateRows,templateCols);
	in.close();

}

SegmenterHumanFaceTemplateGrabcut::~SegmenterHumanFaceTemplateGrabcut()
{
}

void SegmenterHumanFaceTemplateGrabcut::getPixelProbability(const cv::Mat& img, cv::Mat_<float>& prob, vector<Rect> faces)
{	
	float rad = 0.5f * (img.rows+img.cols);
	Rect defaultFace(int(_m_vec4fDefaultFace[0]*img.cols),int(_m_vec4fDefaultFace[1]*img.rows)
		,int(_m_vec4fDefaultFace[2]*rad), int(_m_vec4fDefaultFace[3]*rad));
	
	Rect face;
	if ( faces.size() == 0)
	{
		face = defaultFace;
	}
	else
	{
		face = UtilsOpencv::FindNearestRect(defaultFace,faces);
	}

	rad = 0.25f * (face.width+face.height);
	double r= 30.0/rad;

	Mat imgo;
	cv::resize(img,imgo,Size(),r,r);
	prob.create(imgo.rows,imgo.cols);

	Point centerFace(int((face.x+0.5*face.width)*r),int((face.y+face.height*0.5)*r));
	Point centerDict(this->_m_matFaceTemplate.cols/2,this->_m_matFaceTemplate.rows/2);

	for(int i=0;i<prob.rows;i++)
	{
		for(int j=0;j<prob.cols;j++)
		{
			int lx = j-centerFace.x+centerDict.x;
			int ly = i-centerFace.y+centerDict.y;

			if ( (lx<0)||(lx>=_m_matFaceTemplate.cols)||(ly<0)||(ly>=_m_matFaceTemplate.rows))
			{
				//do nothing
			}
			else
			{
				prob(i,j) = _m_matFaceTemplate(ly,lx);
			}
		}
	}
	
	cv::resize(prob,prob,Size(img.cols,img.rows));
	GaussianBlur(prob,prob,Size(),0.8,0.8);
	int x = cv::countNonZero(prob>0.5);

}

void SegmenterHumanFaceTemplateGrabcut::_prob2energy(const cv::Mat_<float>& prob, cv::Mat_<float>& fgdEnergy, cv::Mat_<float>& bgdEnergy)
{
	fgdEnergy = prob.clone();
	bgdEnergy = prob.clone();

	cv::log(prob,fgdEnergy);
	cv::log(1-prob,bgdEnergy);
	fgdEnergy = -fgdEnergy;
	bgdEnergy = -bgdEnergy;
}

void SegmenterHumanFaceTemplateGrabcut::_getColorProba(const cv::Mat& img, const cv::Mat_<int>& label, cv::Mat_<float>& colorProba)
{
	Mat data = img.reshape(1,img.cols*img.rows);
	Mat la = label.reshape(1,label.cols*label.rows);

	ProbaEstimatorGMMOpencv probaEstimator;
	ClassifierProbaEstimate cls(&probaEstimator);

	cls.train(data,label);
	cls.predictProba(data,colorProba);

	colorProba = colorProba.reshape(1,img.rows);
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