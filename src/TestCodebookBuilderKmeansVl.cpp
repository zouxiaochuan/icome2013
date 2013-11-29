#include "CodebookBuilderKmeansVl.h"
#include "UtilsOpencv.hpp"
#include "CodebookVQ.h"

#include <boost/filesystem.hpp>
#include <iostream>

using namespace boost::filesystem;
using namespace std;

int main(int argn, char* argc[])
{
	path pathDesc("../../data/test_descs.txt");
	Mat_<float> data;

	UtilsOpencv::ReadTxt(pathDesc.string(),data);

	CodebookBuilderKmeansVl cbBuilder(200,50,1);

	CodebookVQ cb;
	cbBuilder.build(data,cb);
	//cout << cb.getBasis() << endl;
}