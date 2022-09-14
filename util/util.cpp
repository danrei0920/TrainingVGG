#include "util.h"



void util::read_cifar_to_file(char* bin_path)
{
    const int img_num = CIFAT_10_OENFILE_DATANUM;
    const int img_size = 3073; 
    const int img_size_1 = 1024;
    const int data_size = img_num * img_size;
    const int row = 32;
    const int col = 32;

    uchar* labels = (uchar*)malloc(CIFAT_10_TOTAL_DATANUM);
    uchar* cifar_data = (uchar*)malloc(data_size);

    for (int i = 0; i < CIFAT_10_FILENUM; i++) 
    {
        char str[200] = { 0 };
        sprintf(str, bin_path, i + 1);

        FILE* fp = fopen(str, "rb");
        if (fp == nullptr) {
            std::cout << "Read file error" << std::endl;
            return;
        }
        fread(cifar_data, 1, data_size, fp);  

  
        for (int j = 0; j < CIFAT_10_OENFILE_DATANUM; j++)
        {
            long int offset = j * img_size;
            long int offset0 = offset + 1;    
            long int offset1 = offset0 + img_size_1;    
            long int offset2 = offset1 + img_size_1;  

            long int idx = i * CIFAT_10_OENFILE_DATANUM + j;
            labels[idx] = cifar_data[offset];  

            cv::Mat img(row, col, CV_8UC3);
            for (int y = 0; y < row; y++)
            {
                for (int x = 0; x < col; x++)
                {
                    int index = y * col + x;
                    //
                    img.at<cv::Vec3b>(y, x) = cv::Vec3b(cifar_data[offset2 + index], cifar_data[offset1 + index], cifar_data[offset0 + index]);    //BGR
                }
            }
            //
            char str1[200] = { 0 };
            sprintf(str1, "C:/LIBS/datasets/cifar-10-binary/cifar-10-batches-bin/image/%d.tif", idx);
            cv::imwrite(str1, img);
        }

        fclose(fp);
    }

    cv::Mat label_mat(100, 500, CV_8UC1, labels);   //
    cv::imwrite("C:/LIBS/datasets/cifar-10-binary/cifar-10-batches-bin/label.tif", label_mat);

    free(labels);
    free(cifar_data);
    
}
