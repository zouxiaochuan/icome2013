#include "GlobalConfig.h"
#include "AFeatureExtractor.h"
#include "UtilsBoost.hpp"
#include <opencv2/opencv.hpp>
#include "AClassifier.h"
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
	string fileOutput = argc[2];

	GlobalConfig cfg;
	cfg.read("cfg_test.json");

	AFeatureExtractor* pExtractor = cfg.getFeatureExtractor();

	vector<boost::filesystem::path> files;
	UtilsBoost::DirRecursive(boost::filesystem::path(pathInput),files);

	AClassifier* pClassifier = cfg.getClassifier();
	Mat_<int> label;
	label.create((int)files.size(),1);

	BOOST_LOG_TRIVIAL(info) << "begin processing, number: " << files.size();

	int total = (int) files.size();
	int cnt = 0;

	ofstream out;
	out.open(fileOutput.c_str());
	if (out.fail() || out.bad())
	{
		cout << "cannot open output file : " << fileOutput << endl;
		exit(1);
	}
	out.close();

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
		if (!img.empty())
		{
			Mat_<float> feature;
			pExtractor->extractFromMat(img,feature);
			Mat_<int> r = label.row(i);
			pClassifier->predict(feature,r);
		}
		else
		{
			label(i) = 0;
		}

#pragma omp critical
		{
			out.open(fileOutput.c_str(),ios::app);
			out << files[i].leaf().string() << " " << label(i) << endl;
			out.close();
		}
	}

	BOOST_LOG_TRIVIAL(info) << "process FINISHED!!!";
}