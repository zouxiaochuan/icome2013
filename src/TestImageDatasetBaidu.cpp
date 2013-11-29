#include "ImageDatasetBaidu.h"

using namespace std;
using namespace cv;

int main(int argn, char* argc[])
{
	string root = argc[1];

	ImageDatasetBaidu dataset(root);
	
	int size = dataset.getSize();
	if ( size != 29968)
	{
		cout << "FAIL: size is " << size << ", not 29968" << endl;
		return 1;
	}

	Mat_<int> label = dataset.getLabels();
	if (label(1) != 0)
	{
		cout << "FAIL: label(1) is not 0" << endl;
		return 1;
	}
	if (label(20903) != 1)
	{
		cout << "FAIL: label(20903) is not 1" << endl;
		return 1;
	}


	Mat img = dataset.getImage(29967);
	if ( (img.rows != 569) || ( img.cols != 791))
	{
		cout << "FAIL: image(29967).size is [" << img.rows << "," << img.cols << "]" << ", is not [569,791]" << endl;
		return 1;
	}

	img = dataset.getImage(3555);
	if( (img.rows != 333) || ( img.cols != 500))
	{
		cout << "FAIL: image(3555).size is [" << img.rows << "," << img.cols << "]" << ", is not [333,500]" << endl;
		return 1;
	}

	return 0;
}
