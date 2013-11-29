#ifndef _UTILSOPENCV_H_
#define _UTILSOPENCV_H_

#include "opencv2/opencv.hpp"
#include <boost/algorithm/string.hpp>
#include <vector>
#include <queue>
#include <set>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace boost::algorithm;

class UtilsOpencv
{
public:
	template<class T>
	static void ReadTxt(const string& filename, Mat_<T>& mat);

	template<class T>
	static void ReadTxt(istream& in, Mat_<T>& mat, int rows, int cols);

	template<class T>
	static void WriteTxt(const string& filename, const Mat_<T>& mat);

	template<class T>
	static void WriteTxt(ostream& out, const Mat_<T>& mat);

	template<class T>
	static void ReadSvm(const string& filename, Mat_<T>& data, Mat_<int>& label);

	template<class T>
	static void WriteSvm(const string& filename, Mat_<T>& data, Mat_<int>& label);

	template<class T>
	static void MergeRows(vector<Mat_<T> >& rows, Mat_<T>& mat);

	template<class T>
	static void MergeRows(queue<Mat_<T> >& rows, int size, Mat_<T>& mat);

	static void SelectRows(const Mat& src, const Mat_<int>& idx, Mat& dst)
	{
		int size = idx.rows*idx.cols;
		dst.create(size,src.cols, src.type());

		for(int i=0;i<size;i++)
		{
			src.row(idx(i)).copyTo(dst.row(i));
		}
	}

	static void SelectRows(const Mat& src, const vector<int>& idx, Mat& dst)
	{
		Mat_<int> matIdx(idx);
		SelectRows(src,matIdx,dst);
	}

	template<class T>
	static void FindValue(const Mat_<T> values, T val, Mat_<int>& index);

	template<class T>
	static void ConcatenateRows(const vector<Mat_<T> > rows, Mat_<T>& mat);

	template<class T>
	static void Bool2Index(const Mat& boolean, Mat_<T>&idx);

	template<class T>
	static void RandPerm(int n, Mat_<T>& perm);

	template<class T>
	static bool IsZero(const Mat_<T> mat);

	template<class T>
	static void Unique(const Mat_<T>& mat, Mat_<T>& umat);

	static Rect BoundingBoxFromMask(const Mat& mask)
	{
		int left,right,top,bottom;
		//left
		for(int i=0;i<mask.cols;i++)
		{
			left = i;
			if ( countNonZero(mask.col(i)) > 0)
			{
				break;
			}
		}

		//right
		for(int i=mask.cols-1;i>=0;i--)
		{
			right = i;
			if ( countNonZero(mask.col(i)) > 0)
			{
				break;
			}
		}

		//top
		for(int i=0;i<mask.rows;i++)
		{
			top = i;
			if ( countNonZero(mask.row(i)) > 0)
			{
				break;
			}
		}

		//bottom
		for(int i=mask.rows-1;i>=0;i--)
		{
			bottom = i;
			if( countNonZero(mask.row(i)) > 0)
			{
				break;
			}
		}

		return Rect(left,top,right-left,bottom-top);
	}

	static Rect ScaleRect(Rect rect, Rect border, double leftFactor, double upFactor, double rightFactor, double downFactor)
	{
		double borderXs = border.x;
		double borderXe = border.x+border.width;
		double borderYs = border.y;
		double borderYe = border.y+border.height;

		Point cent;
		cent.x = rect.x + rect.width/2;
		cent.y = rect.y + rect.height/2;

		double width = rect.width;
		double height = rect.height;

		int startX = (int) max(borderXs,cent.x-width*leftFactor/2);
		int endX = (int) min(borderXe,cent.x+width*rightFactor/2);
		int startY = (int) max(borderYs,cent.y-height*upFactor/2);
		int endY = (int) min(borderYe,cent.y+height*downFactor/2);

		return cv::Rect(startX,startY,endX-startX,endY-startY);
	}

	static double RectDistance(Rect r1, Rect r2)
	{
		Point ul1(r1.x,r1.y);
		Point ur1(r1.x+r1.width,r1.y);
		Point bl1(r1.x,r1.y+r1.height);
		Point br1(r1.x+r1.width,r1.y+r1.height);

		Point ul2(r2.x,r2.y);
		Point ur2(r2.x+r2.width,r2.y);
		Point bl2(r2.x,r2.y+r2.height);
		Point br2(r2.x+r2.width,r2.y+r2.height);
		
		return norm(ul1-ul2)+norm(ur1-ur2)+norm(bl1-bl2)+norm(br1-br2);
	}

	static Rect FindNearestRect(Rect r, const vector<Rect>& rects)
	{
		Rect ret;
		double mindist =  1e10;
		for(int i=0;i<rects.size();i++)
		{
			double dist = UtilsOpencv::RectDistance(r,rects[i]);
			if ( dist < mindist)
			{
				mindist = dist;
				ret = rects[i];
			}
		}

		return ret;
	}
};


template< class T>
void UtilsOpencv::ReadTxt(const string& filename, Mat_<T>& mat)
{
	ifstream infile(filename.c_str());

	string sline = "";

	int rows = 0;
	while( !infile.eof())
	{
		string line;
		getline(infile,line);
		if (line.length() > 0)
		{
			rows++;
			if (sline.length() == 0)
			{
				sline = line;
			}
		}
	}

	sline.append(" ");
	istringstream lineIn(sline);
	int cols = 0;
	T tmp;
	while( !lineIn.eof())
	{
		lineIn >> tmp;
		
		if (lineIn.good())
		{
			cols++;
		}
	}

	mat.create(rows,cols);

	infile.close();
	infile.open(filename.c_str());
	int count = rows*cols;
	int i= 0;
	while(i < count)
	{
		infile >> tmp;
		mat(i++) = tmp;
	}
}

template< class T>
void UtilsOpencv::ReadTxt(istream& in, Mat_<T>& mat, int rows, int cols)
{
	mat.create(rows,cols);
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<cols;j++)
		{
			in >> mat(i,j);
		}
	}
}

template<class T>
void UtilsOpencv::WriteTxt(const string& filename, const Mat_<T>& mat)
{
	ofstream out(filename.c_str());

	for(int i=0;i<mat.rows;i++)
	{
		for(int j=0;j<mat.cols;j++)
		{
			out << mat(i,j) << " ";
		}
		out << endl;
	}

	out.close();
}

template<class T>
void UtilsOpencv::WriteTxt(ostream& out, const Mat_<T>& mat)
{
	for(int i=0;i<mat.rows;i++)
	{
		for(int j=0;j<mat.cols;j++)
		{
			out << mat(i,j) << " ";
		}
		out << endl;
	}
}

template<class T>
void UtilsOpencv::ReadSvm(const string& filename, Mat_<T>& data, Mat_<int>& label)
{
	ifstream in;
	in.open(filename.c_str());

	if ( in.fail() || in.bad())
	{
		return;
	}

	int nline = 0;
	int maxIndex = 0;
	while( !in.eof())
	{
		string line;
		getline(in,line);
		boost::algorithm::trim(line);
		if ( line.length() > 0)
		{
			nline++;
			vector<string> pairs;
			boost::algorithm::split(pairs,line,boost::algorithm::is_any_of("\t "),boost::algorithm::token_compress_on);

			if ( pairs.size() >1)
			{
				vector<string> idval;
				boost::algorithm::split(idval,pairs[pairs.size()-1],boost::algorithm::is_any_of(":"));
				int index = atoi(idval[0].c_str());
				maxIndex = max(maxIndex,index);
			}
		}
	}
	in.close();

	data.create(nline,maxIndex);
	data.setTo(0);
	label.create(nline,1);

	in.open(filename.c_str());
	int iline = 0;
	while( !in.eof())
	{
		string line;
		getline(in,line);
		boost::algorithm::trim(line);
		if ( line.length() > 0)
		{
			vector<string> pairs;
			boost::algorithm::split(pairs,line,boost::algorithm::is_any_of("\t "),boost::algorithm::token_compress_on);
			label(iline) = atoi(pairs[0].c_str());

			if ( pairs.size() >1)
			{
				for(int i=1;i<pairs.size();i++)
				{
					vector<string> idval;
					boost::algorithm::split(idval,pairs[i],boost::algorithm::is_any_of(":"));
					int id = atoi(idval[0].c_str());
					double val = atof(idval[1].c_str());
					data(iline,id-1) = (T) val;
				}
			}

			iline++;
		}
	}
	in.close();

}

template<class T>
void UtilsOpencv::WriteSvm(const string& filename, Mat_<T>& data, Mat_<int>& label)
{
	if ( data.rows != label.total())
	{
		throw runtime_error("data,label size not match");
	}

	int size = data.rows;

	ofstream out(filename.c_str());

	for(int i=0;i<size;i++)
	{
		out << label(i);

		for (int j=0;j<data.cols;j++)
		{
			T val = data(i,j);
			if( val != 0)
			{
				out << " " << j+1 << ":" << val;
			}
		}

		out << endl;
	}

	out.close();
}

template< class T>
void UtilsOpencv::MergeRows(vector<Mat_<T> >& rows, Mat_<T>& mat)
{
	if (rows.size() == 0)
	{
		mat = Mat_<T>();
		return;
	}
	int nRow = 0;
	int nCol = rows[0].cols;
	for ( int i=0;i<rows.size();i++)
	{
		nRow += rows[i].rows;
	}

	mat.create(nRow,nCol);
	int rowIdx = 0;
	for(int i=0;i<rows.size();i++)
	{
		if ( rows[i].rows == 0)
		{
			continue;
		}
		rows[i].copyTo(mat.rowRange(rowIdx,rowIdx+rows[i].rows));
		rowIdx += rows[i].rows;
		//free memory
		rows[i] = Mat_<T>();
	}
}

template< class T>
void UtilsOpencv::MergeRows(queue<Mat_<T> >& rows, int size, Mat_<T>& mat)
{
	
	if (rows.size() == 0)
	{
		mat = Mat_<T>();
		return;
	}

	int rowIdx = 0;
	Mat_<T> val = rows.front();
	
	mat.create(size,val.cols);
	
	while( !rows.empty())
	{
		val = rows.front();
		rows.pop();
		val.copyTo(mat.rowRange(rowIdx+val.rows));
		rowIdx += val.rows;
	}
}

template<class T>
void UtilsOpencv::Bool2Index(const Mat& boolean, Mat_<T>&idx)
{
	int nnz = cv::countNonZero(boolean);
	idx.create(nnz,1);
	int size = boolean.rows*boolean.cols;

	int iidex = 0;
	for(int i=0;i<size;i++)
	{
		if (boolean.at<uchar>(i) > 0)
		{
			idx(iidex++) = i;
		}
	}
}

template<class T>
void UtilsOpencv::RandPerm(int n, Mat_<T>& perm)
{
	perm.create(n,1);
	for(int i=0;i<n;i++)
	{
		perm(i) = i;
	}
	cv::randShuffle(perm,1.0);
}

template<class T>
bool UtilsOpencv::IsZero(const Mat_<T> mat)
{
	if( mat.empty())
	{
		return false;
	}

	int size = mat.rows*mat.cols;
	for(int i=0;i<size;i++)
	{
		if (mat(i) != 0)
		{
			return false;
		}
	}

	return true;
}

template<class T>
void UtilsOpencv::ConcatenateRows(const vector<Mat_<T> > rows, Mat_<T>& mat)
{
	int cols = 0;
	for(int i=0;i<rows.size();i++)
	{
		cols += (int) rows[i].total();
	}

	mat.create(1,cols);
	int colIdx = 0;
	for(int i=0;i<rows.size();i++)
	{
		rows[i].copyTo(mat.colRange(colIdx,colIdx+(int)rows[i].total()));
		colIdx += (int)rows[i].total();
	}
}

template<class T>
void UtilsOpencv::Unique(const Mat_<T>& mat, Mat_<T>& umat)
{
	set<T> uset;
	int size = (int) mat.total();

	for (int i=0;i<size;i++)
	{
		uset.insert(mat(i));
	}

	umat.create((int)uset.size(),1);
	int idx = 0;
	typename set<T>::iterator iter = uset.begin();
	while( iter != uset.end())
	{
		umat(idx++) = (*iter);
		iter++;
	}
}

template<class T>
void UtilsOpencv::FindValue(const Mat_<T> values, T val, Mat_<int>& index)
{
	vector<int> is;

	for(int i=0;i<values.total();i++)
	{
		if ( values(i) == val)
		{
			is.push_back(i);
		}
	}

	index.create((int) is.size(),1);
	for(int i=0;i<is.size();i++)
	{
		index(i) = is[i];
	}
}

#endif