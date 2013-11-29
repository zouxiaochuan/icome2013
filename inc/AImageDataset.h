#ifndef _AIMAGEDATASET_H_
#define _AIMAGEDATASET_H_

#include <opencv2/opencv.hpp>
#include <string>

class AImageDataset
{
public:
	virtual cv::Mat getImage(int idx) const = 0;
	virtual int getSize() const = 0;
	virtual std::string getRawName(int idx) const = 0;

	virtual ~AImageDataset(){};
};


#endif