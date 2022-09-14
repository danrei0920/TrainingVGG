#pragma once
#include "opencv2/opencv.hpp"


#define CIFAT_10_OENFILE_DATANUM 10000
#define CIFAT_10_FILENUM 5
#define CIFAT_10_TOTAL_DATANUM (CIFAT_10_OENFILE_DATANUM*CIFAT_10_FILENUM)


class util
{
public:
	void read_cifar_to_file(char* bin_path);
	//void trains_bin();

};
