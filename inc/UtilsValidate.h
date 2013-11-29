#include <opencv2/opencv.hpp>
#include "AClassifier.h"

class UtilsValidate
{
public:
	static void SplitRandomBalance(const cv::Mat_<int>& label, int ntest, int nrun, cv::Mat_<uchar>& splits);
	static void ValidateRandomBalance(const cv::Mat_<int>& label, const cv::Mat& data, AClassifier* pClassifier, cv::Mat_<double>& result,
		int ntest, int nrun);

	static double GetLoss(const cv::Mat_<int>& label, const cv::Mat& data, AClassifier* pClassifier, const cv::Mat_<uchar>& split);
};