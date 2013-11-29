#include "FeatureExtractorSPM.h"
#include "UtilsOpencv.hpp"
#include "PoolerAvg.h"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

using namespace cv;
using namespace std;


void FeatureExtractorSPM::extractFromMat(const cv::Mat& img, cv::Mat_<float>& feature)
{
	if (!img.empty())
	{
		Mat_<float> d,l;
		this->_m_pLocalExtractor->extractFromMat(img,d,l);
		Mat_<float> encoded;
		this->_m_pEncoder->encode(d,encoded);

		this->_extractFromEncoded(encoded,l,feature);
	}
	else
	{
		feature.create(1,this->featureDimension(img));
		feature.setTo(0.0f);
	}
}

int FeatureExtractorSPM::featureDimension(const cv::Mat& img)
{
	if ( this->_m_pEncoder)
	{
		return this->_SPMFactor() * this->_m_pEncoder->codeDimension();
	}
	else
	{
		return 0;
	}
}

FeatureExtractorSPM::FeatureExtractorSPM(const cv::Mat_<int>& spm, APooler* p)
{
	this->_m_matSPM = spm;

	this->_m_vecPooler.resize(spm.rows);
	for(int i=0;i<_m_vecPooler.size();i++)
	{
		//if (i==0)
		//{
		//	_m_vecPooler[i] = new PoolerAvg();
		//}
		//else
		{
			_m_vecPooler[i] = p;
		}
	}

	this->_m_pLocalExtractor = NULL;
	this->_m_pEncoder = NULL;
}

FeatureExtractorSPM::FeatureExtractorSPM(cv::Mat_<int>& spm, APooler* pp, ALocalExtractor* pl, AEncoder* pe)
{
	this->_m_matSPM = spm;
	this->_m_vecPooler.resize(spm.rows);
	for(int i=0;i<_m_vecPooler.size();i++)
	{
		//if ( i==0)
		//{
		//	_m_vecPooler[i] = new PoolerAvg();
		//}
		//else
		{
			_m_vecPooler[i] = pp;
		}
	}

	this->_m_pLocalExtractor = pl;
	this->_m_pEncoder = pe;
}


void FeatureExtractorSPM::_extractFromEncoded(const cv::Mat_<float>& encoded, const cv::Mat_<float>& location, cv::Mat_<float>& feature)
{
	if (encoded.empty() || location.empty())
	{
		return;
	}
	int nspm = this->_m_matSPM.rows;

	Mat_<float> xs = location.col(0);
	Mat_<float> ys = location.col(1);

	double xmax,ymax;
	cv::minMaxIdx(xs,NULL,&xmax);
	cv::minMaxIdx(ys,NULL,&ymax);

	xmax++;
	ymax++;
	vector<Mat_<float> > subfeatures;

	for(int i=0;i<nspm;i++)
	{
		int nx = _m_matSPM(i,0);
		int ny = _m_matSPM(i,1);

		int xitv = (int) ceil(xmax/nx);
		int yitv = (int) ceil(ymax/ny);

		for(int y=0;y<ymax;y+=yitv)
		{
			int yend = min(y+yitv,(int)ymax);
			Mat ySelect = (ys>=y) & ( ys<yend);
			for( int x=0;x<xmax;x+=xitv)
			{
				int xend = min(x+xitv,(int)xmax);

				Mat xSelect = (xs>=x) & (xs<xend);

				Mat select = xSelect & ySelect;

				Mat_<int> idxSelect;
				UtilsOpencv::Bool2Index(select,idxSelect);
				Mat_<float> codeSelect;
				UtilsOpencv::SelectRows(encoded,idxSelect,codeSelect);
				Mat_<float> subfea;
				this->_m_vecPooler[i]->pool(codeSelect,subfea);
				subfeatures.push_back(subfea);
			}
		}
	}

	UtilsOpencv::ConcatenateRows(subfeatures,feature);
}

int FeatureExtractorSPM::_SPMFactor()
{
	return (int) this->_m_matSPM.col(0).dot(this->_m_matSPM.col(1));
}

void FeatureExtractorSPM::extractFromEncodedPath(const std::string& encodedPath, AImageDataset* pdataset, int codeLen, cv::Mat_<float>& feature)
{
	int nData = pdataset->getSize();
	int featureLen = codeLen * this->_SPMFactor();
	feature.create(nData,featureLen);
	feature.setTo(0);

#pragma omp parallel for
	for(int i=0;i<pdataset->getSize();i++)
	{
		string rawname = pdataset->getRawName(i);
		boost::filesystem::path fullpath(encodedPath);
		fullpath /= rawname;
		fullpath.replace_extension(".xml.gz");
		if (!boost::filesystem::exists(fullpath))
		{
			continue;
		}
		FileStorage fs(fullpath.string(),FileStorage::READ);
		Mat_<float> encoded,location;
		fs["encoded"] >> encoded;
		fs["location"] >> location;
		fs.release();

		Mat_<float> fr = feature.row(i);
		this->_extractFromEncoded(encoded,location,fr);
		//int x = cv::countNonZero(fr);
	}
}