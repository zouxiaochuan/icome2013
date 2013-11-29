#include "UtilsSegmentation.h"
#include "UtilsSuperpixel.h"
#include "graph.h"
#include <stdexcept>

using namespace cv;
using namespace std;

void errfunc(const char* msg)
{
	throw std::runtime_error(msg);
}

void UtilsSegmentation::MaxFlowSuperpixel(std::vector<SuperpixelStatistic>& spstat, const Mat_<float>& fgdEnergy,
		const Mat_<float>& bgdEnergy, float gamma, Mat_<int>& label)
{
	//::Graph<float,float,float> graph(nNode,nEdge,errfunc);
	//graph
	int nEdge = UtilsSuperpixel::CountEdge(spstat);
	Mat_<int> edgeVertex(nEdge,2);
	Mat_<float> edgeWeight(nEdge,1);
	Mat_<float> edgeLen(nEdge,1);

	int idx = 0;
	for(int i=0;i<spstat.size();i++)
	{
		SuperpixelStatistic& sp = spstat[i];
		for( set<int>::iterator j=sp.conn.begin();
			j!= sp.conn.end();
			j++)
		{
			int d = (*j);
			SuperpixelStatistic& dsp = spstat[d];
			if ( i != d)
			{
				edgeVertex(idx,0) = min(i,d);
				edgeVertex(idx,1) = max(i,d);
				float diff = (float) norm(sp.mean_color_ - dsp.mean_color_);
				edgeWeight(idx) = diff*diff;
				edgeLen(idx) = (float) cv::norm(sp.mean_position_-dsp.mean_position_);
				idx++;
			}
		}
	}

	float beta = (float) cv::mean(edgeWeight)[0];

	Graph<float,float,float> graph((int)spstat.size(), nEdge, errfunc);

	graph.add_node((int)spstat.size());

	for(int i=0;i<fgdEnergy.total();i++)
	{
		graph.add_tweights(i,bgdEnergy(i),fgdEnergy(i));
	}

	edgeWeight = - edgeWeight / beta;
	cv::exp(edgeWeight,edgeWeight);
	edgeWeight *= gamma;
	cv::divide(edgeWeight, edgeLen,edgeWeight);

	for(int i=0;i<nEdge;i++)
	{
		float w = edgeWeight(i);
		graph.add_edge(edgeVertex(i,0),edgeVertex(i,1),w,w);
	}

	graph.maxflow();

	label.create((int)spstat.size(),1);
	for(int i=0;i<spstat.size();i++)
	{
		if ( graph.what_segment(i) == Graph<float,float,float>::SOURCE)
		{
			label(i) = 1;
		}
		else
		{
			label(i) = 0;
		}
	}
}


float UtilsSegmentation::CompareTwoMask(cv::Mat mask1, cv::Mat mask2)
{
	float cntTotal = 0.0f;
	float cntInter = 0.0f;

	for(int i=0;i<mask1.rows;i++)
	{
		for(int j=0;j<mask1.cols;j++)
		{
			uchar m1 = mask1.at<uchar>(i,j);
			uchar m2 = mask2.at<uchar>(i,j);

			if ( (m1==m2) && (m1==255))
			{
				cntInter++;
			}
			if ( (m1==255) || (m2==255))
			{
				cntTotal++;
			}
		}
	}

	return cntInter / cntTotal;
}