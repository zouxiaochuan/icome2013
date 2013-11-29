#include "DetectorHoGTemplate.h"
#include "UtilsOpencv.hpp"
#include "UtilsMatlab.h"
#include "hog.h"
#include "ClassifierLiblinear.h"
#include "UtilsValidate.h"
#include <boost/log/trivial.hpp>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

DetectorHoGTemplate::DetectorHoGTemplate(int cellSize, int wndSize)
{
	this->_m_cellSize = cellSize;
	this->_m_wndSize = wndSize;
	this->_mp_classifier = NULL;
}

DetectorHoGTemplate::~DetectorHoGTemplate()
{
	if (this->_mp_classifier)
	{
		delete this->_mp_classifier;
	}
}

void DetectorHoGTemplate::learnFromDataset( AImageDatasetForDetection& dataset)
{
	BOOST_LOG_TRIVIAL(debug) << "begin calculate aspratio";

	double sumaspr = 0.0;
	int nbox = 0;

	this->_m_maxRate = 0.0;
	this->_m_minRate = 1.0;
	this->_m_maxWidth = 0;
	this->_m_minWidth = 9999;

	for(int i=0;i<dataset.getSize();i++)
	{
		vector<Rect> boxes;
		dataset.getBoundingBox(i,boxes);

		for(int j=0;j<boxes.size();j++)
		{
			sumaspr += ((double)boxes[j].width)/boxes[j].height;
			nbox++;
		}

		if ( boxes.size() > 0)
		{
			Mat img = dataset.getImage(i);
			if ( !img.empty())
			{
				this->_m_maxRate = max( _m_maxRate, double(boxes[0].width) / img.cols);
				this->_m_minRate = min( _m_minRate, double(boxes[0].width) / img.cols);
			}
		}
	}

	this->_m_aspratio = sumaspr / nbox;
	BOOST_LOG_TRIVIAL(debug) << "end calculate aspratio: " << this->_m_aspratio;

	this->_m_wndWidth = this->_m_wndSize;
	this->_m_wndHeight = int(ceil(this->_m_wndWidth / this->_m_aspratio));

	double pixwidth = this->_m_wndWidth * this->_m_cellSize;
	double pixheight = this->_m_wndHeight * this->_m_cellSize;

	vector<Mat_<float> > mats;
	vector<Mat_<int> > labels;

	BOOST_LOG_TRIVIAL(debug) << "begin extract feature: ";

	for(int i=0;i<dataset.getSize();i++)
	{
		//if ( i%1000 == 0)
		{
			BOOST_LOG_TRIVIAL(debug) << "index: " << i;
		}

		vector<Rect> boxes;
		dataset.getBoundingBox(i,boxes);

		Mat img;
		img = dataset.getImage(i);

		if (img.empty())
		{
			continue;
		}

		for(int j=0;j<boxes.size();j++)
		{
			Rect box = boxes[j];
			double r = pixwidth/box.width;
			Mat img1;
			resize(img,img1,Size(),r,r);

			box.x = (int) (box.x * r);
			box.y = (int) (box.y * r);
			box.width = (int) (box.width * r);
			box.height = (int) (box.height * r);
			//box.height = (int) pixheight;

			this->_m_maxWidth = max(_m_maxWidth,img1.cols);
			this->_m_minWidth = min(_m_minWidth,img1.cols);

			Mat subimg;
			img1(box).copyTo(subimg);

			Mat_<float> feature;
			this->_extractHoGSingle(subimg, feature);
			mats.push_back(feature);
			Mat_<int> l(1,1);
			l.setTo(1);
			labels.push_back(l);
		}

		if ( img.rows < pixheight)
		{
			double rate = pixheight * 2/ img.rows;
			resize(img,img,Size(),rate,rate);
		}
		if ( img.cols < pixwidth)
		{
			double rate = pixwidth * 2/ img.cols;
			resize(img,img,Size(),rate,rate);
		}

		vector<Mat> subimgs;
		this->_randSubImage(img,(int)pixheight,(int)pixwidth,10,subimgs);

		for(int j=0;j<subimgs.size();j++)
		{
			Mat_<float> feature;
			this->_extractHoGSingle(subimgs[j],feature);
			mats.push_back(feature);
			Mat_<int> l(1,1);
			l.setTo(0);
			labels.push_back(l);
		}
	}

	BOOST_LOG_TRIVIAL(debug) << "end extract feature: ";

	Mat_<float> data;
	Mat_<int> label;
	UtilsOpencv::MergeRows(mats,data);
	UtilsOpencv::MergeRows(labels,label);
	
	BOOST_LOG_TRIVIAL(debug) << "data size: " << data.rows;
	BOOST_LOG_TRIVIAL(debug) << "positive number: " << countNonZero(label);
	BOOST_LOG_TRIVIAL(debug) << "negative number: " << label.rows - countNonZero(label);

	this->_mp_classifier = this->_getOptimalModel(label,data);
}

void DetectorHoGTemplate::_extractHoGSingle(const Mat& img, Mat_<float>& feature)
{
	Mat matlabImg;
	UtilsMatlab::Opencv2Matlab(img,matlabImg);
	VlHog * hog = vl_hog_new (VlHogVariantUoctti, 9, VL_FALSE) ;
	vl_hog_set_use_bilinear_orientation_assignments (hog, VL_FALSE) ;

	Mat floatImg;
	matlabImg.convertTo(floatImg,CV_32F,1.0f/255.0f);
	float* pImg = (float*) floatImg.data;

	vl_hog_put_image(hog, pImg, img.cols, img.rows, img.channels(), this->_m_cellSize) ;

	cv::MatND hogDesc;
	int hogWidth = (int) vl_hog_get_width(hog);
	int hogHeight = (int) vl_hog_get_height(hog);
	int hogDimension = (int) vl_hog_get_dimension(hog);

	int sizes[] = {hogDimension,hogHeight,hogWidth};
	hogDesc.create(3,sizes,CV_32F);
	vl_hog_extract(hog, (float*) hogDesc.data);

	int numDescriptor = hogWidth * hogHeight;
	int feaDimension = hogDimension * this->_m_wndWidth * this->_m_wndHeight;

	feature.create(1,feaDimension);

	int iDescriptor = 0;

	float* pDst = (float*) feature.data;
	int dimenStride = hogHeight*hogWidth;

	for( int yy = 0; yy < this->_m_wndHeight; yy++)
	{
		for( int xx = 0; xx < this->_m_wndWidth; xx++)
		{
			if ( (xx<0) || (xx>=hogWidth) || (yy<0) || ( yy>=hogHeight))
			{
				memset(pDst,0,sizeof(float)*hogDimension);
				pDst+=hogDimension;
			}
			else
			{
				float* pSrc = (float*) (hogDesc.data + hogDesc.step[1]*yy + hogDesc.step[2]*xx);
				for( int dimen = 0;dimen < hogDimension; dimen++)
				{
					*pDst = *pSrc;
					pDst++;
					pSrc += dimenStride;
				}
			}
		}//xx
	}//yy


	vl_hog_delete(hog);
}

void DetectorHoGTemplate::_extractHoGDense(const cv::Mat& img, cv::Mat_<float>& feature, cv::Mat_<float>& location)
{
	Mat matlabImg;
	UtilsMatlab::Opencv2Matlab(img,matlabImg);
	VlHog * hog = vl_hog_new (VlHogVariantUoctti, 9, VL_FALSE) ;
	vl_hog_set_use_bilinear_orientation_assignments (hog, VL_FALSE) ;

	Mat floatImg;
	matlabImg.convertTo(floatImg,CV_32F,1.0f/255.0f);
	float* pImg = (float*) floatImg.data;

	vl_hog_put_image(hog, pImg, img.cols, img.rows, img.channels(), this->_m_cellSize) ;

	cv::MatND hogDesc;
	int hogWidth = (int) vl_hog_get_width(hog);
	int hogHeight = (int) vl_hog_get_height(hog);
	int hogDimension = (int) vl_hog_get_dimension(hog);

	int sizes[] = {hogDimension,hogHeight,hogWidth};
	hogDesc.create(3,sizes,CV_32F);
	//float* data = (float*) vl_malloc(hogWidth*hogHeight*hogDimension*sizeof(float));
	vl_hog_extract(hog, (float*) hogDesc.data);

	//float* p = (float*) data;
	//for(int i=0;i<50;i+=1)
	//{
	//	cout << p[i] << endl;
	//}

	int lenWidth = this->_m_wndWidth / 2;
	int lenHeight = this->_m_wndHeight / 2;
	int numDescriptor = hogWidth * hogHeight;
	int feaDimension = hogDimension * this->_m_wndHeight * this->_m_wndWidth;

	feature.create(numDescriptor,feaDimension);
	location.create(numDescriptor,2);

	int iDescriptor = 0;

	for (int r = 0; r < hogHeight; r++)
	{
		for (int c = 0; c < hogWidth; c++)
		{
			Mat feaVec = feature.row(iDescriptor);
			Mat locVec = location.row(iDescriptor);
			int iFea = 0;
			iDescriptor++;
			locVec.at<float>(0) = (float) c*this->_m_cellSize;
			locVec.at<float>(1) = (float) r*this->_m_cellSize;

			float* pDst = (float*) feaVec.data;
			int dimenStride = hogHeight*hogWidth;

			for( int yy = r-lenHeight; yy < r-lenHeight+_m_wndHeight; yy++)
			{
				for( int xx = c-lenWidth; xx < c-lenWidth+_m_wndWidth; xx++)
				{
					if ( (xx<0) || (xx>=hogWidth) || (yy<0) || ( yy>=hogHeight))
					{
						memset(pDst,0,sizeof(float)*hogDimension);
						pDst+=hogDimension;
					}
					else
					{
						float* pSrc = (float*) (hogDesc.data + hogDesc.step[1]*yy + hogDesc.step[2]*xx);
						for( int dimen = 0;dimen < hogDimension; dimen++)
						{
							*pDst = *pSrc;
							pDst++;
							pSrc += dimenStride;
						}
					}

					iFea+=hogDimension;
				}//xx
			}//yy
		}//c
	}//r

	vl_hog_delete(hog);
}

void DetectorHoGTemplate::_randSubImage(const Mat& img, int height, int width, int n, vector<Mat>& subimg)
{
	Scalar randX;
	Scalar randY;

	cv::randu(randX,0,img.cols-width);
	cv::randu(randY,0,img.rows-height);

	Rect roi((int)randX[0],(int)randY[0],width,height);
	Mat simg;
	img(roi).copyTo(simg);
	subimg.push_back(simg);
}

AClassifier* DetectorHoGTemplate::_getOptimalModel( const cv::Mat_<int>& label, const cv::Mat_<float>& data)
{
	double optC = 0.0;
	double minLoss = 1.0;
	for(int i=-6;i<7;i++)
	{
		double c = pow(2.0,i);
		ClassifierLiblinear cls(2,c,1);
		Mat_<double> res;
		UtilsValidate::ValidateRandomBalance(label,data,&cls,res,5,5);
		double loss = cv::mean(res)[0];

		if (loss < minLoss)
		{
			minLoss = loss;
			optC = c;
		}
	}

	BOOST_LOG_TRIVIAL(debug) << "detector optimal loss: " << minLoss;
	ClassifierLiblinear* pcls =  new ClassifierLiblinear(2,optC,1.0);
	pcls->train(data,label);
	return pcls;
}

void DetectorHoGTemplate::save(const std::string& filename)
{
	string fn1 = filename;
	fn1.append("_p");
	ofstream out(fn1.c_str());
	out << this->_m_aspratio << endl;
	out << this->_m_cellSize << endl;
	out << this->_m_wndHeight << endl;
	out << this->_m_wndSize << endl;
	out << this->_m_wndWidth << endl;
	out << this->_m_maxRate << endl;
	out << this->_m_minRate << endl;
	out << this->_m_maxWidth << endl;
	out << this->_m_minWidth << endl;

	string fn2 = filename;
	fn2.append("_m");
	this->_mp_classifier->save(fn2);
	out.close();
}

void DetectorHoGTemplate::load(const std::string& filename)
{
	string fn1 = filename;
	fn1.append("_p");
	fstream in(fn1.c_str());
	in >> this->_m_aspratio;
	in >> this->_m_cellSize;
	in >> this->_m_wndHeight;
	in >> this->_m_wndSize;
	in >> this->_m_wndWidth;
	in >> this->_m_maxRate;
	in >> this->_m_minRate;
	in >> this->_m_maxWidth;
	in >> this->_m_minWidth;

	string fn2 = filename;
	fn2.append("_m");
	this->_mp_classifier = new ClassifierLiblinear();
	this->_mp_classifier->load(fn2);
	in.close();
}

void DetectorHoGTemplate::detectSingleScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence)
{
	Mat_<float> feature,location;
	this->_extractHoGDense(img,feature,location);

	double maxconf = 0.0;
	int maxidx = 0;
	int posx = 0;
	int posy = 0;

	Mat_<float> confs;
	this->_mp_classifier->predictProba(feature,confs);

	//cv::minMaxIdx(confs,NULL,&maxconf,NULL,&maxidx);
	for(int i=0;i<(int)confs.total();i++)
	{
		if ( confs(i) > maxconf)
		{
			maxconf = confs(i);
			maxidx = i;
		}
	}

	posx = (int) location(maxidx,0)+this->_m_cellSize/2;
	posy = (int) location(maxidx,1)+this->_m_cellSize/2;

	int pixlenWidth = this->_m_wndWidth * this->_m_cellSize / 2;
	int pixlenHeight = this->_m_wndHeight * this->_m_cellSize / 2;

	boxes.push_back(Rect(posx-pixlenWidth,posy-pixlenHeight,pixlenWidth*2,pixlenHeight*2));
	
	confidence.create(1,1);
	confidence(0) = confs(maxidx);
}

void DetectorHoGTemplate::detectMultiScale(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confidence, double scaleFactor
		, double maxSize, double minSize)
{
	double r  = maxSize / img.cols;

	Mat imgc = img.clone();

	cv::resize(imgc,imgc,Size(),r,r);

	Rect rect;
	float maxconf=0.0;
	while( imgc.cols > minSize)
	{
		vector<Rect> _boxes;
		Mat_<float> _confidence;
		this->detectSingleScale(imgc,_boxes,_confidence);

		for(int i=0;i<_boxes.size();i++)
		{
			if ( _confidence(i) > maxconf)
			{
				maxconf = _confidence(i);

				double r = double(img.cols)/imgc.cols;
				rect = Rect(int(_boxes[i].x*r),int(_boxes[i].y*r),int(_boxes[i].width*r),int(_boxes[i].height*r));
			}
		}

		cv::resize(imgc,imgc,Size(),1.0/scaleFactor,1.0/scaleFactor);
	}

	confidence.create(1,1);
	confidence(0) = maxconf;
	boxes.push_back(rect);
}