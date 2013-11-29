#include "CodebookVQ.h"
#include <string>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

CodebookVQ::CodebookVQ()
{
}

CodebookVQ::CodebookVQ(CodebookVQ& other)
{
	this->m_matBasis = other.m_matBasis;
}

void CodebookVQ::save(const string& filename)
{
	string fname = filename;
	fname.append(".xml.gz");

	cv::FileStorage fs(fname,FileStorage::WRITE);
	fs << "basis" << this->m_matBasis;
	fs.release();
}

void CodebookVQ::load(const string& filename)
{
	string fname = filename;
	fname.append(".xml.gz");
	cv::FileStorage fs(fname,FileStorage::READ);
	fs["basis"] >> this->m_matBasis;
}