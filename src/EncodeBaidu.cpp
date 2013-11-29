#include "GlobalConfig.h"
#include "AEncoder.h"
#include <string>
#include <iostream>

using namespace std;

int main(int argn, char* argc[])
{
	string filenameCfg = argc[1];
	GlobalConfig cfg;
	cfg.read(filenameCfg);

	AEncoder* pEncoder = cfg.getEncoder();

	pEncoder->encodeFromDirectory(cfg.getImagePath(),*(cfg.getLocalExtractor()),cfg.getEncodedPath());
}