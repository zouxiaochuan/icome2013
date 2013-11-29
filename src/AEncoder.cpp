#include "AEncoder.h"
#include "UtilsBoost.hpp"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <iostream>

using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace boost;

void AEncoder::encodeFromDirectory(const std::string& pathImage, ALocalExtractor& extractor, const std::string& pathOutput)
{
	vector<path> files;
	UtilsBoost::DirRecursive(path(pathImage),files);

#pragma omp parallel for
	for(int i=0;i<files.size();i++)
	{
		path cfile = files[i];

		if ( cfile.extension().string() == ".jpg")
		{
			Mat img = imread(cfile.string());
			Mat_<float> d,l;
			extractor.extractFromMat(img,d,l);
			Mat_<float> encoded;
			this->encode(d,encoded);
			string rawname = cfile.string();
			replace_first(rawname,path(pathImage).string(),"");
			path dstFilename(pathOutput);
			dstFilename /= rawname;
			dstFilename.replace_extension(".xml.gz");
			if (!exists(dstFilename.parent_path()))
			{
				boost::filesystem::create_directory(dstFilename.parent_path());
			}

			cv::FileStorage fs(dstFilename.string(),FileStorage::WRITE);
			fs << "encoded" << encoded;
			fs << "location" << l;
			fs.release();
		}
	}

}