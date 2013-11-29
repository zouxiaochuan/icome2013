#include "GlobalConfig.h"
#include "AImageDatasetForSegmentation.h"
#include "ASegmenter.h"
#include "UtilsSegmentation.h"

#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <iostream>

using namespace std;

int main(int argn, char* argv[])
{
	GlobalConfig cfg;
	cfg.read(argv[1]);

	AImageDatasetForSegmentation* pDataset = dynamic_cast<AImageDatasetForSegmentation*>(cfg.getDataset());
	ASegmenter* pSegmenter = cfg.getSegmenter();

	int cnt = 0;
	float tr = 0.0f;
	BOOST_LOG_TRIVIAL(info) << "begin process... ";

#pragma omp parallel for
	for(int i=0;i<pDataset->getSize();i++)
	{
		Mat mask = pDataset->getMask(i);
		if ( !mask.empty())
		{
			//BOOST_LOG_TRIVIAL(info) << "begin process file: " << i;
			Mat img = pDataset->getImage(i);
			Mat_<uchar> mymask;
			pSegmenter->segment(img,mymask);

			cv::resize(mymask,mymask,Size(mask.cols,mask.rows));
			mymask.setTo(255,mymask>128);
			mymask.setTo(0,mymask<=128);

			float r = UtilsSegmentation::CompareTwoMask(mask,mymask);

			BOOST_LOG_TRIVIAL(info) << "processed file " << i << " ,r=" << r;

#pragma omp critical
			{
				tr += r;
				cnt++;
			}
		}
	}

	cout << tr/cnt << endl;
}