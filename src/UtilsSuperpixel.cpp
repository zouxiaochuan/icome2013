#include "UtilsSuperpixel.h"
#include "superpixel.h"

using namespace cv;

int UtilsSuperpixel::CountLabel(const cv::Mat_<int>& segmentation)
{
	int n_label = 0;
	for( int j=0; j<segmentation.rows; j++ )
		for( int i=0; i<segmentation.cols; i++ )
			if ( n_label <= segmentation(j,i) )
				n_label = segmentation(j,i) + 1;
	return n_label;
}

void UtilsSuperpixel::Stat(const cv::Mat_<int>& segmentation, const cv::Mat_<float>& data, 
	const std::vector<SuperpixelStatistic>& stats, Mat_<float>& mdata)
{
	mdata.create((int)stats.size(),1);
	mdata.setTo(0.0f);

	for(int i=0;i<segmentation.rows;i++)
	{
		for(int j=0;j<segmentation.cols;j++)
		{
			mdata(segmentation(i,j),0) += data(i,j);
		}
	}

	for(int i=0;i<(int)stats.size();i++)
	{
		mdata(i,0) /= stats[i].size_;
	}
}

int UtilsSuperpixel::CountEdge(const std::vector<SuperpixelStatistic>& stats)
{
	int cnt = 0;
	for(int i=0;i<stats.size();i++)
	{
		cnt += (int) stats[i].conn.size();
	}

	return cnt;
}