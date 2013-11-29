#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include "UtilsOpencv.hpp"
#include "boost/filesystem.hpp"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

void test_ReadTxt()
{
	cout << "test_ReadTxt begin" << endl;
	Mat_<float> mat;
	//path pathTxt("D:\\work\\icome_challenge\\2013\\code\\data\\mat.txt");
	path pathTxt("../../data/mat.txt");
	UtilsOpencv::ReadTxt( pathTxt.string(), mat);

	if ( mat.rows != 3)
	{
		cout << "FAILED: mat.rows = " << mat.rows << endl;
		return;
	}
	if ( mat.cols != 5)
	{
		cout << "FAILED: mat.cols = " << mat.cols << endl;
		return;
	}
	if ( mat(0,3) != 4)
	{
		cout << "FAILED: mat(0,3) = " << mat(0,3) << endl;
		return;
	}
	if ( mat(2,4) != 15)
	{
		cout << "FAILED: mat(2,4) = " << mat(2,4) << endl;
		return;
	}

	cout << "test_ReadTxt end" << endl;
}

int main(int argn, char* argc[])
{

	test_ReadTxt();

	return 0;
}