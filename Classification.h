#ifndef CLASSIFICATION_H
#define CLASSIFICATION_H

#include <opencv2/opencv.hpp>
#undef slots
#include <torch/script.h>
#include <torch/torch.h>
#define slots Q_SLOTS
#include <model/vgg.h>
#include <model/resnet.h>

class Classifier
{
private:
    torch::Device device = torch::Device(torch::kCPU);
    //VGG vgg_train = VGG{nullptr};
    ResNet resnet_train = ResNet{ nullptr };
    void load_traindata();
public:
    Classifier(int gpu_id = -1);
    void Initialize(int num_epochs, int batch_size, float learning_rate, std::string train_val_dir, std::string image_type);
    void Train(int epochs, int batch_size, float learning_rate, std::string train_val_dir, std::string image_type, std::string save_path);
    int Predict(cv::Mat &image);
    void LoadWeight(std::string weight);

};

#endif // CLASSIFICATION_H
