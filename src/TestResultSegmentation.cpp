#include "GlobalConfig.h"
#include "AFeatureExtractor.h"
#include "UtilsBoost.hpp"
#include <opencv2/opencv.hpp>
#include "ASegmenter.h"
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argn, char* argc[])
{

    boost::log::core::get()->set_filter
    (
        boost::log::trivial::severity >= boost::log::trivial::info
    );

	string pathInput = argc[1];
	string pathOutput = argc[2];

	if ( !boost::filesystem::exists(boost::filesystem::path(pathInput)))
	{
		cout << "input path not exist" << endl;
		return 1;
	}
	if ( !boost::filesystem::exists(boost::filesystem::path(pathOutput)))
	{
		cout << "output path not exist" << endl;
		return 2;
	}

	GlobalConfig cfg;
	cfg.read("cfg_test.json");

	ASegmenter* pSegmenter = cfg.getSegmenter();

	vector<boost::filesystem::path> files;
	UtilsBoost::DirRecursive(boost::filesystem::path(pathInput),files);

	BOOST_LOG_TRIVIAL(info) << "begin processing, number: " << files.size();

	int total = (int) files.size();
	int cnt = 0;

#pragma omp parallel for
	for( int i=0;i<total;i++)
	{
#pragma omp critical
		{
			cnt++;
			BOOST_LOG_TRIVIAL(info) << "process file: " << files[i] << "(" << cnt << "/" << total << ")";
			//BOOST_LOG_TRIVIAL(info).flush();
		}
		Mat img = imread(files[i].string());
		Mat_<uchar> mask;
		if (!img.empty())
		{
			pSegmenter->segment(img,mask);

			mask = 255-mask;
			boost::filesystem::path savename(pathOutput);
			savename /= files[i].leaf();
			string saven = savename.string();
			boost::algorithm::replace_last(saven,".jpg","-profile.jpg");
			imwrite(saven,mask);
		}
	}

	BOOST_LOG_TRIVIAL(info) << "process FINISHED!!!";
}