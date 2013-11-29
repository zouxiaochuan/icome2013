#ifndef _POOLERMAX_H_
#define _POOLERMAX_H_

#include "APooler.h"

class PoolerMax : public APooler
{
public:
	virtual void pool(const cv::Mat& code, cv::Mat& pooled);
};

#endif