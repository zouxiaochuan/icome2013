#ifndef _UTILSBOOST_HPP_
#define _UTILSBOOST_HPP_

#include <boost/property_tree/ptree.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sstream>

class UtilsBoost
{
public:
	template <class T>
	static void GetVectorFromPtree( boost::property_tree::ptree& pt, std::vector<T>& vec);

	static void DirRecursive(const boost::filesystem::path& root, std::vector<boost::filesystem::path>& files)
	{
		boost::filesystem::recursive_directory_iterator end,iter(root);

		while( iter != end)
		{
			if ( !boost::filesystem::is_directory(iter->path()))
			{
				files.push_back(iter->path());
			}
			iter++;
		}
	}

	template <class T>
	static void GetMatFromPtree( boost::property_tree::ptree& pt, cv::Mat_<T>& mat);

	static std::string FormatPTime( const boost::posix_time::ptime& t, const std::string& format)
	{
		using namespace boost::posix_time;
		std::locale loc(std::cout.getloc(),
			new time_facet(format.c_str()));

		std::ostringstream ss;
		ss.imbue(loc);
		ss << t;
		return ss.str();
	}
};


template<class T>
void UtilsBoost::GetVectorFromPtree( boost::property_tree::ptree& pt, std::vector<T>& vec)
{
    BOOST_FOREACH(boost::property_tree::ptree::value_type& v,pt) 
	{
		vec.push_back(v.second.get_value<T>());
    }
}

template <class T>
void UtilsBoost::GetMatFromPtree( boost::property_tree::ptree& pt, cv::Mat_<T>& mat)
{
	int rows = (int) pt.size();
	int cols = (int) pt.begin()->second.size();

	mat.create(rows,cols);

	int i=0;

    BOOST_FOREACH(boost::property_tree::ptree::value_type& v,pt) 
	{
		BOOST_FOREACH(boost::property_tree::ptree::value_type& vv,v.second) 
		{
			mat(i++) = vv.second.get_value<T>();
		}
    }
}


#endif