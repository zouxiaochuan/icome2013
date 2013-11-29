#include "LocalExtractorDenseSiftVl.h"
#include "UtilsMatlab.h"
#include "dsift.h"
#include <vector>

using namespace std;

void LocalExtractorDenseSiftVl::extractFromMat( const Mat& img, Mat_<float>& descriptors, Mat_<float>& location)
{
	Mat fimg = img.clone();

	if ( this->mMaxSize > 0)
	{
		double maxs = max(img.rows,img.cols);
		if ( maxs > this->mMaxSize)
		{
			double r = this->mMaxSize / maxs;
			cv::resize(fimg,fimg,Size(),r,r);
		}
	}

	if (fimg.channels() == 3)
	{
		cvtColor(fimg,fimg,CV_BGR2GRAY);
	}

	fimg.convertTo(fimg,CV_32F,1.0f/255.0f);

	//Mat matlabImg;
	//UtilsMatlab::Opencv2Matlab(fimg,matlabImg);

	int width = fimg.cols;
	int height = fimg.rows;

	VlDsiftFilter *dsift = vl_dsift_new_basic (width, height, this->mStep, this->mBinSize);
	vl_dsift_set_flat_window(dsift,VL_TRUE);

    //VlDsiftDescriptorGeometry const *geom ;
	float* data = fimg.ptr<float>();

    vl_dsift_process (dsift, data) ;

    VlDsiftKeypoint const *frames = vl_dsift_get_keypoints (dsift) ;
    float const *descrs = vl_dsift_get_descriptors (dsift) ;

	int descrSize = vl_dsift_get_descriptor_size(dsift);
	int pointNum = vl_dsift_get_keypoint_num(dsift);

	descriptors.create(pointNum,descrSize);
	location.create(pointNum,2);

	memcpy(descriptors.data,descrs,descrSize*pointNum*sizeof(float));

	for(int i=0;i<pointNum;i++)
	{
		location(i,0) = (float) frames[i].x+1;
		location(i,1) = (float) frames[i].y+1;
	}

	vl_dsift_delete(dsift);
}