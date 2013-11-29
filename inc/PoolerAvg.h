#ifndef _POOLERAVG_H_
#define _POOLERAVG_H_

#include "APooler.h"

class PoolerAvg : public APooler
{
public:
	virtual void pool(const cv::Mat& code, cv::Mat& pooled);
};

#endif