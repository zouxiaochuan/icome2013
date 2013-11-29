#include "GlobalConfig.h"
#include "ImageDatasetBaidu.h"
#include <fstream>

using namespace cv;
using namespace std;

void checkPoint( Point2f point, vector<Point2f>& border)
{
	border[0].x = min(point.x, border[0].x);
	border[0].y = min(point.y, border[0].y);
	border[1].x = max(point.x, border[1].x);
	border[1].y = min(point.y, border[1].y);
	border[2].x = max(point.x, border[2].x);
	border[2].y = max(point.y, border[2].y);
	border[3].x = min(point.x, border[3].x);
	border[3].y = max(point.y, border[3].y);
}

void outputBorder( ostream& out, vector<Point2f> border)
{
	if ( (border[0].x != border[3].x) || (border[0].y != border[1].y) || ( border[1].x != border[2].x) || ( border[2].y != border[3].y))
	{
		cout << "logic error" << endl;
	}

	out << border[0].x << " " <<  border[0].y << " " << border[1].x-border[0].x << " "
		<< border[2].y-border[0].y << endl;
}
int main(int argn, char* argv[])
{
	GlobalConfig cfg;
	cfg.read(argv[1]);

	ImageDatasetBaidu* pDataset = dynamic_cast<ImageDatasetBaidu*>(cfg.getDataset());


	int size = pDataset->getSize();

	vector<Point2f> upleft(4);
	vector<Point2f> upright(4);
	vector<Point2f> boleft(4);
	vector<Point2f> boright(4);

	upleft[0].x = 1.0f;
	upleft[0].y = 1.0f;
	upleft[1].x = 0.0f;
	upleft[1].y = 1.0f;
	upleft[2].x = 0.0f;
	upleft[2].y = 0.0f;
	upleft[3].x = 1.0f;
	upleft[3].y = 0.0f;

	upright = boleft = boright = upleft;
	
	float minw = 1.0f;
	float maxw = 0.0f;
	float minh = 1.0f;
	float maxh = 0.0f;

#pragma omp parallel for
	for(int i=0;i<size;i++)
	{
		vector<Rect> boxes;
		pDataset->getBoundingBox(i,boxes);

		if ( boxes.size() == 0)
		{
			continue;
		}

		Mat img;
		img = pDataset->getImage(i);

		if ( img.empty())
		{
			continue;
		}

#pragma omp critical
		{
		for(int j=0;j<boxes.size();j++)
		{
			float lx = (float) boxes[j].x;
			float rx = (float) boxes[j].x+boxes[j].width;
			float uy = (float) boxes[j].y;
			float by = (float) boxes[j].y+boxes[j].height;

			lx /= img.cols;
			rx /= img.cols;
			uy /= img.rows;
			by /= img.rows;

			float width = rx - lx;
			float height = by - uy;

			checkPoint(Point2f(lx,uy),upleft);
			checkPoint(Point2f(rx,uy),upright);
			checkPoint(Point2f(rx,by),boright);
			checkPoint(Point2f(lx,by),boleft);

			minw = min(width,minw);
			maxw = max(width,maxw);
			minh = min(height,minh);
			maxh = max(height,maxh);
		}
		}
	}

	ofstream out(argv[2]);

	outputBorder(out,upleft);
	outputBorder(out,upright);
	outputBorder(out,boright);
	outputBorder(out,boleft);

	out << minw << " " << maxw << endl;
	out << minh << " " << maxh << endl;

	out.close();
}