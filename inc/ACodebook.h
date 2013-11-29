#ifndef _ACODEBOOK_H_
#define _ACODEBOOK_H_

#include <string>

class ACodebook
{
public:
	virtual void save(const std::string& filename) = 0;
	virtual void load(const std::string& filename) = 0;

};

#endif