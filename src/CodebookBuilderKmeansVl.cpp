#include "CodebookBuilderKmeansVl.h"
#include "CodebookVQ.h"
#include "kmeans.h"
#include <boost/log/trivial.hpp>

using namespace std;

void CodebookBuilderKmeansVl::build(const Mat_<float> data, ACodebook& cb)
{
	float* pdata = (float*) data.data;
	int ndim = data.cols;
	int ndata = data.rows;

	BOOST_LOG_TRIVIAL(info)  << "begin build codebook: " << endl;
	BOOST_LOG_TRIVIAL(info)  << "number of data: " << ndata << endl;
	BOOST_LOG_TRIVIAL(info)  << "number of dimension: " << ndim << endl;

	VlKMeans * kmeans = vl_kmeans_new(VL_TYPE_FLOAT,VlDistanceL2);
	vl_kmeans_set_verbosity (kmeans, 1) ;
	vl_kmeans_set_num_repetitions (kmeans, this->_m_nRun) ;
	vl_kmeans_set_algorithm (kmeans, VlKMeansLloyd) ;
	vl_kmeans_set_initialization (kmeans, VlKMeansPlusPlus) ;
	vl_kmeans_set_max_num_iterations (kmeans, this->_m_MaxIteration) ;

	double energy = vl_kmeans_cluster(kmeans, pdata, ndim, ndata, this->_m_nCent) ;

	Mat_<float> cents;
	cents.create(this->_m_nCent,ndim);

	memcpy (cents.data,
          vl_kmeans_get_centers (kmeans),
          vl_get_type_size (VL_TYPE_FLOAT) * ndim * vl_kmeans_get_num_centers(kmeans)) ;

	CodebookVQ& codebook = static_cast<CodebookVQ&>(cb);
	codebook.setBasis(cents);
}