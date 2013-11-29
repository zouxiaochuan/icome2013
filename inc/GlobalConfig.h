#ifndef _GLOBALCONFIG_H_
#define _GLOBALCONFIG_H_

#include "ALocalExtractor.h"
#include "ACodebookBuilder.h"
#include "AEncoder.h"
#include "AFeatureExtractor.h"
#include "APooler.h"
#include "AImageDataset.h"
#include "AClassifier.h"
#include "ADetector.h"
#include "ASegmenter.h"

#include <boost/property_tree/json_parser.hpp>
#include <string>

class GlobalConfig
{
public:
	GlobalConfig();
	virtual ~GlobalConfig();

	boost::property_tree::ptree getRoot();

	void read(const std::string& filename);
	ALocalExtractor* getLocalExtractor();
	ACodebookBuilder* getCodebookBuilder();
	ACodebook* getCodebook();
	AEncoder* getEncoder();
	AFeatureExtractor* getFeatureExtractor();
	APooler* getPooler();
	AImageDataset* getDataset();
	AClassifier* getClassifier();
	ADetector* getDetector();
	ASegmenter* getSegmenter();

	string getCodebookPath();
	int getCodebookMaxTrainingNum();

	string getImagePath();
	string getEncodedPath();
	string getFeaturePath();

	string getClassifierModelPath();
	string getDetectorModelPath();

private:
	void _parse( boost::property_tree::ptree& root);
	void _parseLocalExtractor( boost::property_tree::ptree& pt);
	void _parseCodebookBuilder( boost::property_tree::ptree& pt);
	void _parseCodebook( boost::property_tree::ptree& pt);
	void _parseEncoder( boost::property_tree::ptree& pt);
	void _parseFeatureExtractor( boost::property_tree::ptree& pt);
	void _parsePooler( boost::property_tree::ptree& pt);
	void _parseDataset(boost::property_tree::ptree& pt);
	void _parseClassifier( boost::property_tree::ptree& pt);
	void _parseDetector( boost::property_tree::ptree& pt);
	void _parseSegmenter( boost::property_tree::ptree& pt);

private:
	boost::property_tree::ptree _m_root;

	ALocalExtractor* _m_pLocalExtractor;
	ACodebookBuilder* _m_pCodebookBuilder;
	AEncoder* _m_pEncoder;
	AFeatureExtractor* _m_pFeatureExtractor;
	APooler* _m_pPooler;
	string _m_strCodebookPath;
	int _m_numCodebookMaxTraining;
	ACodebook* _m_pCodebook;
	AImageDataset* _m_pDataset;
	AClassifier* _m_pClassifier;
	ADetector* _m_pDetector;
	ASegmenter* _m_pSegmenter;

	string _m_strImagePath;
	string _m_strEncodedPath;
	string _m_strFeaturePath;
	string _m_strClassifierModelPath;
	string _m_strDetectorModelPath;
};

#endif