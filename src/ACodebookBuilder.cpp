#include "ACodebookBuilder.h"
#include "UtilsOpencv.hpp"

#include <vector>

using namespace std;

void ACodebookBuilder::buildFromDataset(ACodebook& cb, const AImageDataset& dataset, ALocalExtractor& extractor, int trainingNum)
{
	double nTrain = trainingNum;
	double nData = dataset.getSize();

	double samplePerImage = ceil(nTrain / nData);
	nTrain = samplePerImage * nData;

	//int nTotal = 0;

	vector<Mat_<float> > vecDesc;
	vecDesc.resize((int) nData);

#pragma omp parallel for
	for ( int i=0;i<(int)nData;i++)
	{
		Mat img = dataset.getImage(i);
		Mat_<float> descs,locs;
		extractor.extractFromMat(img,descs,locs);
		
		//removing zero rows
		vector<int> idxNnz;
		
		for(int j=0;j<descs.rows;j++)
		{
			if (!UtilsOpencv::IsZero(descs.row(j)))
			{
				idxNnz.push_back(j);
			}
		}
		Mat_<float> tmp;
		UtilsOpencv::SelectRows(descs,idxNnz,tmp);
		descs = tmp;

		if (descs.rows > samplePerImage)
		{
			Mat_<int> perm;
			UtilsOpencv::RandPerm(descs.rows,perm);
			Mat_<float> selDesc;
			
			perm = perm.rowRange(0,(int)samplePerImage);
			UtilsOpencv::SelectRows(descs,perm,selDesc);
			descs = selDesc;
		}

		vecDesc[i] = descs;
	}

	Mat_<float> data;
	UtilsOpencv::MergeRows(vecDesc,data);

	this->build(data,cb);
}
