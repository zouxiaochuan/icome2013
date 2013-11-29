#include <opencv2/opencv.hpp>
#include "UtilsOpencv.hpp"

using namespace cv;

int main(int argn, char* argc[])
{
	string filename = argc[1];
	string objname = argc[2];
	string outname = argc[3];
	Mat mat;
	FileStorage fs(filename,FileStorage::READ);
	fs[objname] >> mat;

	if(mat.type() == CV_32F)
	{
		UtilsOpencv::WriteTxt<float>(outname,mat);
	}
}