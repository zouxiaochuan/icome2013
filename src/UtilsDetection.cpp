#include "UtilsDetection.h"

using namespace cv;

cv::Rect UtilsDetection::FilterBoundingBox(const cv::Mat& img, std::vector<cv::Rect>& boxes, cv::Mat_<float>& confs,Mat_<float>& borders)
{
	float mindist = 9999.0f;
	Rect r;
	Point2f cent(img.cols/2.0f,img.rows/2.0f);
	for ( int i=0;i<boxes.size();i++)
	{
		Point2f rcent(boxes[i].x+boxes[i].width/2.0f,boxes[i].y+boxes[i].height/2.0f);

		float dist = (float) norm(rcent-cent);
		if ( dist < mindist)
		{
			mindist = dist;
			r = boxes[i];
		}
	}

	return r;
}

void UtilsDetection::RankDetection(std::vector<cv::Rect>& boxes, const cv::Mat& mask, cv::Mat_<float>& scores)
{
	Mat_<float> inter((int)boxes.size(),1);
	Mat_<float> unio((int)boxes.size(),1);

	inter.setTo(0.0f);
	unio.setTo(0.0f);

	for(int i=0;i<mask.rows;i++)
	{
		for(int j=0;j<mask.cols;j++)
		{
			Point c(j,i);
			uchar cval = mask.at<uchar>(i,j);

			for(int k=0;k<boxes.size();k++)
			{
				bool inside = c.inside(boxes[k]);

				if ( inside && (cval > 0))
				{
					inter(k)++;
				}
				if ( inside || (cval > 0))
				{
					unio(k)++;
				}
			}
		}
	}

	cv::divide(inter,unio,scores);
}