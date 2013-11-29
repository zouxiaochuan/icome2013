#include "GlobalConfig.h"
#include "UtilsBoost.hpp"
#include "LocalExtractorDenseSiftMultiScale.h"
#include "CodebookBuilderKmeansVl.h"
#include "EncoderBoFSoft.h"
#include "CodebookVQ.h"
#include "FeatureExtractorSPM.h"
#include "ImageDatasetBaidu.h"
#include "PoolerMax.h"
#include "ClassifierLiblinear.h"
#include "DetectorHoGTemplate.h"
#include "DetectorLatentSVMOpencv.h"
#include "DetectorHumanBaidu.h"
#include "SegmenterHumanSimple.h"
#include "SegmenterHumanFaceTemplate.h"
#include "SegmenterHumanFaceTemplateGrabcut.h"
#include "SegmenterHumanDet.h"
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/filesystem.hpp>
#include <iostream>

using namespace boost::property_tree;
using namespace std;

GlobalConfig::GlobalConfig()
{
	this->_m_pCodebookBuilder = NULL;
	this->_m_pLocalExtractor = NULL;
	this->_m_pCodebook = NULL;
	this->_m_pEncoder = NULL;
	this->_m_pPooler = NULL;
	this->_m_pFeatureExtractor = NULL;
	this->_m_pDataset = NULL;
	this->_m_pClassifier = NULL;
	this->_m_pDetector = NULL;
	this->_m_pSegmenter = NULL;

    boost::shared_ptr< boost::log::core > core = boost::log::core::get();
	core->add_global_attribute("TimeStamp", boost::log::attributes::local_clock());
	//core->add_global_attribute("Severity", );
	//BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", boost::log::trivial::severity_level);
	boost::log::add_file_log("log.log"
		/*
		, boost::log::keywords::format = (
			boost::log::expressions::stream
				//<< boost::log::expressions::format_date_time<boost::posix_time::ptime>("TimeStamp", "%Y-%m-%d %H:%M:%S")
				<< boost::log::expressions::format_date_time(boost::log::expressions::attr<boost::posix_time::ptime>("TimeStamp"),"%Y-%m-%d %H:%M:%S")
                << ": <" << boost::log::trivial::severity
                << "> " << boost::log::expressions::smessage
		)
		*/
		, boost::log::keywords::format = "%TimeStamp% : <%Severity%> %Message%"
		, boost::log::keywords::auto_flush = true
	);
}

GlobalConfig::~GlobalConfig()
{
	if (this->_m_pLocalExtractor)
	{
		delete this->_m_pLocalExtractor;
	}
	if (this->_m_pCodebookBuilder)
	{
		delete this->_m_pCodebookBuilder;
	}
	if (this->_m_pCodebook)
	{
		delete this->_m_pCodebook;
	}
	if (this->_m_pEncoder)
	{
		delete this->_m_pEncoder;
	}
	if (this->_m_pPooler)
	{
		delete this->_m_pPooler;
	}
	if (this->_m_pFeatureExtractor)
	{
		delete this->_m_pFeatureExtractor;
	}
	if (this->_m_pDataset)
	{
		delete this->_m_pDataset;
	}
	if (this->_m_pClassifier)
	{
		delete this->_m_pClassifier;
	}
	if (this->_m_pDetector)
	{
		delete this->_m_pDetector;
	}
	if (this->_m_pSegmenter)
	{
		delete _m_pSegmenter;
	}
}

ptree GlobalConfig::getRoot()
{
	return this->_m_root;
}

ACodebook* GlobalConfig::getCodebook()
{
	return this->_m_pCodebook;
}

ALocalExtractor* GlobalConfig::getLocalExtractor()
{
	return this->_m_pLocalExtractor;
}

ACodebookBuilder* GlobalConfig::getCodebookBuilder()
{
	return this->_m_pCodebookBuilder;
}
int GlobalConfig::getCodebookMaxTrainingNum()
{
	return this->_m_numCodebookMaxTraining;
}
string GlobalConfig::getCodebookPath()
{
	return this->_m_strCodebookPath;
}
string GlobalConfig::getImagePath()
{
	return this->_m_strImagePath;
}
string GlobalConfig::getEncodedPath()
{
	return this->_m_strEncodedPath;
}
string GlobalConfig::getFeaturePath()
{
	return this->_m_strFeaturePath;
}
AEncoder* GlobalConfig::getEncoder()
{
	return this->_m_pEncoder;
}
APooler* GlobalConfig::getPooler()
{
	return this->_m_pPooler;
}
AFeatureExtractor* GlobalConfig::getFeatureExtractor()
{
	return this->_m_pFeatureExtractor;
}
AImageDataset* GlobalConfig::getDataset()
{
	return this->_m_pDataset;
}
AClassifier* GlobalConfig::getClassifier()
{
	return this->_m_pClassifier;
}
ADetector* GlobalConfig::getDetector()
{
	return this->_m_pDetector;
}
ASegmenter* GlobalConfig::getSegmenter()
{
	return this->_m_pSegmenter;
}
string GlobalConfig::getClassifierModelPath()
{
	return this->_m_strClassifierModelPath;
}
string GlobalConfig::getDetectorModelPath()
{
	return this->_m_strDetectorModelPath;
}
void GlobalConfig::read(const std::string& filename)
{
	read_json(filename,this->_m_root);
	this->_parse(this->_m_root);
}


void GlobalConfig::_parse( ptree& root)
{
	_parseLocalExtractor(root.get_child("LocalExtractor"));
	_parseCodebookBuilder(root.get_child("CodebookBuilder"));
	_parseCodebook(root.get_child("Codebook"));
	_parseEncoder(root.get_child("Encoder"));
	_parsePooler(root.get_child("Pooler"));
	_parseFeatureExtractor(root.get_child("FeatureExtractor"));
	_parseDataset(root.get_child("Dataset"));
	_parseClassifier(root.get_child("Classifier"));
	_parseDetector(root.get_child("Detector"));
	_parseSegmenter(root.get_child("Segmenter"));

	this->_m_strImagePath = root.get_child("ImagePath").get_value<string>();
	this->_m_strEncodedPath = root.get_child("EncodedPath").get_value<string>();
	this->_m_strFeaturePath = root.get_child("FeaturePath").get_value<string>();

	BOOST_LOG_TRIVIAL(debug) << "encoded path: " << this->_m_strEncodedPath << endl;
}

void GlobalConfig::_parseLocalExtractor( ptree& pt)
{
	string type = pt.get_child("Type").get_value<string>();
	string lib = pt.get_child("Lib").get_value<string>();
	int step = pt.get_child("Step").get_value<int>();
	vector<int> sizes;
	UtilsBoost::GetVectorFromPtree(pt.get_child("Sizes"),sizes);
	int maxsize = pt.get_child("MaxSize").get_value<int>(-1);

	BOOST_LOG_TRIVIAL(debug)  << "config local extractor type: " << type;
	BOOST_LOG_TRIVIAL(debug)  << "config local extractor lib: " << lib;
	BOOST_LOG_TRIVIAL(debug)  << "config local extractor step: " << step;
	BOOST_LOG_TRIVIAL(debug)  << "config local extractor sizes [";

	for(int i=0;i<sizes.size();i++)
	{
		BOOST_LOG_TRIVIAL(debug)  << sizes[i] << ",";
	}
	BOOST_LOG_TRIVIAL(debug)  << "]";

	if (type=="densesift")
	{
		this->_m_pLocalExtractor = new LocalExtractorDenseSiftMultiScale(lib,step,sizes,maxsize);
	}
}

void GlobalConfig::_parseCodebookBuilder( ptree& pt)
{
	string type = pt.get_child("Type").get_value<string>();
	string lib = pt.get_child("Lib").get_value<string>();
	int numSample = pt.get_child("MaxTrainingNum").get_value<int>();
	int numCode = pt.get_child("NumCode").get_value<int>();
	int maxIter = pt.get_child("MaxIteration").get_value<int>();

	this->_m_numCodebookMaxTraining = numSample;

	BOOST_LOG_TRIVIAL(debug)  << "codebook builder type: " << type;
	BOOST_LOG_TRIVIAL(debug)  << "codebook builder lib: " << lib;
	BOOST_LOG_TRIVIAL(debug)  << "codebook builder number of code: " << numCode;
	BOOST_LOG_TRIVIAL(debug)  << "codebook builder max iteration: " << maxIter ;
	BOOST_LOG_TRIVIAL(debug)  << "codebook builder max training number: " << numSample;

	if (type=="kmeans" && lib=="vlfeat")
	{
		this->_m_pCodebookBuilder = new CodebookBuilderKmeansVl(numCode,maxIter,1);
	}
}

void GlobalConfig::_parseCodebook( boost::property_tree::ptree& pt)
{
	string path = pt.get_child("Path").get_value<string>();
	string type = pt.get_child("Type").get_value<string>();

	this->_m_strCodebookPath = path;

	if (type == "kmeans")
	{
		this->_m_pCodebook = new CodebookVQ();
	}
	string fullname = path;
	fullname.append(".xml.gz");
	if (boost::filesystem::exists(boost::filesystem::path(fullname)))
	{
		this->_m_pCodebook->load(path);
	}
	
}

void GlobalConfig::_parseEncoder( ptree& pt)
{
	string type = pt.get_child("Type").get_value<string>();

	BOOST_LOG_TRIVIAL(debug)  << "ecoder type: " << type;

	if ( type == "bofsoft")
	{
		int nnz = pt.get_child("NumNonZero").get_value<int>();

		BOOST_LOG_TRIVIAL(debug)  << "encoder number of nonzeros: " << nnz;

		this->_m_pEncoder = new EncoderBoFSoft(dynamic_cast<CodebookVQ&>(*this->getCodebook()),nnz);
	}
	
}

void GlobalConfig::_parseDataset( ptree& pt)
{
	string type = pt.get_child("Type").get_value<string>();
	BOOST_LOG_TRIVIAL(debug)  << "dataset type: " << type;

	if ( type == "baidu")
	{
		string root = pt.get_child("Path").get_value<string>();
		try
		{
			this->_m_pDataset = new ImageDatasetBaidu(root);
		}
		catch( runtime_error& )
		{
			this->_m_pDataset = NULL;
		}
	}
}
void GlobalConfig::_parseFeatureExtractor( boost::property_tree::ptree& pt)
{
	string type = pt.get_child("Type").get_value<string>();
	BOOST_LOG_TRIVIAL(debug)  << "feature extractor type: " << type;

	if (type == "spm")
	{
		Mat_<int> spm;
		UtilsBoost::GetMatFromPtree(pt.get_child("SPM"),spm);
		BOOST_LOG_TRIVIAL(debug) << "spm:" << spm;

		//this->_m_pFeatureExtractor = new FeatureExtractorSPM(spm,this->getPooler());
		this->_m_pFeatureExtractor = new FeatureExtractorSPM(spm, this->getPooler(), this->getLocalExtractor(), this->getEncoder());
	}
}

void GlobalConfig::_parsePooler( boost::property_tree::ptree& pt)
{
	string type = pt.get_child("Type").get_value<string>();
	BOOST_LOG_TRIVIAL(debug)  << "pooler type: " << type;
	

	if ( type == "max")
	{
		this->_m_pPooler = new PoolerMax();
	}
}

void GlobalConfig::_parseClassifier( boost::property_tree::ptree& pt)
{
	string type = pt.get_child("Type").get_value<string>();
	BOOST_LOG_TRIVIAL(debug)  << "classifier type: " << type;

	if ( type == "liblinear")
	{
		int solvertype = pt.get_child("SolverType").get_value<int>();
		double c = pt.get_child("C").get_value<double>();
		double bias = pt.get_child("Bias").get_value<double>();

		this->_m_pClassifier = new ClassifierLiblinear(solvertype,c,bias);

		BOOST_LOG_TRIVIAL(debug)  << "classifier solver type: " << solvertype;
		BOOST_LOG_TRIVIAL(debug)  << "classifier c: " << c;
		BOOST_LOG_TRIVIAL(debug)  << "classifier bias: " << bias;
	}

	string path = pt.get_child("Path").get_value<string>();

	if (boost::filesystem::exists(boost::filesystem::path(path)))
	{
		this->_m_pClassifier->load(path);
	}
	this->_m_strClassifierModelPath = path;

	BOOST_LOG_TRIVIAL(debug)  << "classifier path: " << path;
}

void GlobalConfig::_parseDetector( boost::property_tree::ptree& pt)
{
	string type = pt.get_child("Type").get_value<string>();
	BOOST_LOG_TRIVIAL(debug)  << "detector type: " << type;

	if (type == "hogtemplate")
	{
		int cellSize = pt.get_child("CellSize").get_value<int>();
		int wndSize = pt.get_child("WndSize").get_value<int>();

		this->_m_pDetector = new DetectorHoGTemplate(cellSize,wndSize);

		BOOST_LOG_TRIVIAL(debug)  << "detector cell size: " << cellSize;
		BOOST_LOG_TRIVIAL(debug)  << "detector window size: " << wndSize;

		string path = pt.get_child("Path").get_value<string>();
		this->_m_strDetectorModelPath = path;
		path.append("_p");

		if (boost::filesystem::exists(boost::filesystem::path(path)))
		{
			this->_m_pDetector->load(this->_m_strDetectorModelPath);
		}

		BOOST_LOG_TRIVIAL(debug)  << "detector path: " << path;
	}

	if ( type == "latentsvm")
	{
		int maxsize = pt.get_child("MaxSize").get_value<int>();
		string modelpath = pt.get_child("ModelPath").get_value<string>();
		float thresh = pt.get_child("Thresh").get_value<float>();

		ADetector* p = new DetectorLatentSVMOpencv(modelpath,maxsize,thresh);

		BOOST_LOG_TRIVIAL(debug)  << "detector max size: " << maxsize;
		BOOST_LOG_TRIVIAL(debug)  << "detector model path: " << modelpath;
		BOOST_LOG_TRIVIAL(debug)  << "detector threshold: " << thresh;

		string modelhuman = pt.get_child("ModelPathBaidu").get_value<string>();
		this->_m_pDetector = new DetectorHumanBaidu(modelhuman,p);

		BOOST_LOG_TRIVIAL(debug)  << "detector human path: " << modelhuman;
	}
}

void GlobalConfig::_parseSegmenter( boost::property_tree::ptree& pt)
{
	string type = pt.get_child("Type").get_value<string>();

	BOOST_LOG_TRIVIAL(debug)  << "segmenter type: " << type;
	if ( type == "simple")
	{
		string faceModel = pt.get_child("FaceModel").get_value<string>();
		this->_m_pSegmenter = new SegmenterHumanSimple(faceModel);

		BOOST_LOG_TRIVIAL(debug)  << "segmenter face model: " << faceModel;
	}
	if ( type == "facetemplate")
	{
		string faceModel = pt.get_child("FaceModel").get_value<string>();
		string templateFile = pt.get_child("TemplateFile").get_value<string>();
		this->_m_pSegmenter = new SegmenterHumanFaceTemplate(faceModel,templateFile);

		BOOST_LOG_TRIVIAL(debug)  << "segmenter face model: " << faceModel;
		BOOST_LOG_TRIVIAL(debug)  << "segmenter template file: " << templateFile;
	}
	if ( type == "det")
	{
		string templateFile = pt.get_child("Template").get_value<string>();
		this->_m_pSegmenter = new SegmenterHumanDet(this->_m_pDetector, templateFile);
		BOOST_LOG_TRIVIAL(debug)  << "segmenter template file: " << templateFile;
	}
}