#include "GlobalConfig.h"
#include "AImageDatasetForSegmentation.h"
#include "ASegmenter.h"
#include "UtilsOpencv.hpp"

#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

const int DICT_WIDTH = 400;
const int DICT_HEIGHT = 800;

void updateMask(const Mat& maskgt_, Rect rect, Mat_<float>& tmask)
{
	double r= 80.0/rect.width;

	Mat maskgt;
	cv::resize(maskgt_,maskgt,Size(),r,r);
	cv::threshold(maskgt,maskgt,128,255,THRESH_BINARY);
	
	Point centerDict(DICT_WIDTH/2,DICT_HEIGHT/2);
	Point centerDet(rect.x + (int) (rect.width*0.5), rect.y + (int) (rect.height*0.5));
	centerDet = Point((int) (centerDet.x * r), (int) (centerDet.y * r));

	for(int i=0;i<maskgt.rows;i++)
	{
		for(int j=0;j<maskgt.cols;j++)
		{
			if ( maskgt.at<uchar>(i,j) > 0)
			{
				int lx = j-centerDet.x+centerDict.x;
				int ly = i-centerDet.y+centerDict.y;

				if ( (lx<0)||(lx>=DICT_WIDTH)||(ly<0)||(ly>=DICT_HEIGHT))
				{
					//do nothing
				}
				else
				{
					tmask(ly,lx)++;
				}
			}
		}
	}
}

Rect findNearestRect(Rect r, const vector<Rect>& rects)
{
	Rect ret;
	double mindist =  1e10;
	for(int i=0;i<rects.size();i++)
	{
		double dist = UtilsOpencv::RectDistance(r,rects[i]);
		if ( dist < mindist)
		{
			mindist = dist;
			ret = rects[i];
		}
	}

	return ret;
}

int main(int argn, char* argv[])
{
	GlobalConfig cfg;
	cfg.read(argv[1]);

	AImageDatasetForSegmentation* pDataset = dynamic_cast<AImageDatasetForSegmentation*>(cfg.getDataset());
	ADetector* pDetector = cfg.getDetector();

	Mat_<float> tmask(DICT_HEIGHT,DICT_WIDTH);
	tmask.setTo(0.0f);

	int size = pDataset->getSize();

#pragma omp parallel for
	for(int i=0;i<size;i++)
	{
		BOOST_LOG_TRIVIAL(info) << "idx:" << i;
		Mat mask = pDataset->getMask(i);

		if ( !mask.empty())
		{
			Mat img = pDataset->getImage(i);
			if (img.empty())
			{
				continue;
			}

			double r = 300.0 / img.cols;

			cv::resize(img,img,Size(),r,r);
			cv::resize(mask,mask,Size(img.cols,img.rows));
			mask.setTo(255,mask>128);
			mask.setTo(0,mask<=128);

			vector<Rect> boxes;
			Mat_<float> confs;

			pDetector->detectMultiScale(img,boxes,confs,0,0,0);

			if ( boxes.size() > 0)
			{
#pragma omp critical
				{
					updateMask(mask,boxes[0],tmask);
				}
			}
		}
	}

	double maxVal;
	cv::minMaxIdx(tmask,NULL,&maxVal);
	tmask /= maxVal;
	
	UtilsOpencv::WriteTxt(argv[2],tmask);
}