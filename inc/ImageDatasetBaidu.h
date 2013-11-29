#ifndef _IMAGEDATASETBAIDU_H_
#define _IMAGEDATASETBAIDU_H_

#include "AImageDatasetForClassification.h"
#include "AImageDatasetForDetection.h"
#include "AImageDatasetForSegmentation.h"

#include <string>


class ImageDatasetBaidu :	public AImageDatasetForClassification,
							public AImageDatasetForDetection,
							public AImageDatasetForSegmentation
{
public:
	virtual cv::Mat getImage(int idx) const;
	virtual int getSize() const;
	virtual std::string getRawName(int idx) const;

	virtual cv::Mat_<int> getLabels();
	virtual void getBoundingBox(int idx, std::vector<cv::Rect>& boxes);
	virtual cv::Mat getMask(int idx);

public:
	ImageDatasetBaidu(const std::string& root);
	virtual ~ImageDatasetBaidu(){};

private:
	std::string _m_pathImage;
	std::string _m_pathMask;
	cv::Mat_<int> _m_matLabels;
	std::vector<std::string> _m_vecFilenames;
};

#endif
