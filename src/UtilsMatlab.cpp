#include "UtilsMatlab.h"

void UtilsMatlab::Opencv2Matlab(const Mat& in, Mat& out)
{
	vector<Mat> mats;
	cv::split(in,mats);

	if ( mats.size() == 1)
	{
		out = in.clone();
		return;
	}

	int sizes[] = {(int)mats.size(),in.rows,in.cols};
	out.create(3,sizes,mats[0].type());

	for( unsigned i=0;i<mats.size();i++)
	{
		uchar* p = out.data + out.step[0] * i;
		memcpy(p, mats[i].data, out.step[0]);
	}
}