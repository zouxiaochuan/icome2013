#include "GlobalConfig.h"
#include "ImageDatasetBaidu.h"
#include "DetectorHumanBaidu.h"
#include "UtilsDetection.h"

#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>

#include <iostream>

using namespace std;

int main(int argn, char* argv[])
{
	GlobalConfig cfg;
	cfg.read(argv[1]);

	string trainpath = argv[2];
	string testpath = argv[3];

	ImageDatasetBaidu trainSet(trainpath);
	ImageDatasetBaidu testSet(testpath);

	DetectorHumanBaidu* pDetector = dynamic_cast<DetectorHumanBaidu*>(cfg.getDetector());

	pDetector->learnFromDataset(&trainSet);

	int testsize = testSet.getSize();

	float cnt = 0.0f;
	float cntwrong = 0.0f;

#pragma omp parallel for
	for(int i=0;i<testsize;i++)
	{
		Mat mask = testSet.getMask(i);
		if ( !mask.empty())
		{
			Mat img = testSet.getImage(i);

			if ( img.empty())
			{
				continue;
			}

			vector<Rect> boxes;
			Mat_<float> confs;
			pDetector->detectMultiScale(img,boxes,confs,0,0,0);

			Mat_<float> gtscore;
			UtilsDetection::RankDetection(boxes,mask,gtscore);

#pragma omp critical
			{
			cnt++;
			for(int j=1;j<(int)gtscore.total();j++)
			{
				if ( gtscore(j) > gtscore(0))
				{
					BOOST_LOG_TRIVIAL(info) << "find wrong";
					cntwrong++;
					break;
				}
			}
			}
		}
	}

	cout << cntwrong / cnt << endl;
}