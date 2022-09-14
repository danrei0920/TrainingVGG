#ifndef VGG_H
#define VGG_H

#undef slots
#include<torch/script.h>
#include<torch/torch.h>
#define slots Q_SLOTS





class VGGImpl: public torch::nn::Module
{
public:
    torch::nn::Sequential _features{ nullptr };
    torch::nn::AdaptiveAvgPool2d _avgpool{ nullptr };
    torch::nn::Sequential _classifier;

    VGGImpl(std::vector<int> &cfg, int num_classes , bool batch_norm = true, int in_channels = 3);

    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(VGG);


VGG vgg16bn(int num_classes); 
VGG vgg19bn(int num_classes);

#endif // VGG16_H
