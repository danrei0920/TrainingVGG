#include "modelcom.h"




inline torch::nn::Conv2dOptions mcom::conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
    int64_t stride, int64_t padding, int groups, bool with_bias) {
    torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
    conv_options.stride(stride);
    conv_options.padding(padding);
    conv_options.bias(with_bias);
    conv_options.groups(groups);
    return conv_options;
}

inline torch::nn::MaxPool2dOptions mcom::maxpool_options(int kernel_size, int stride) {
    torch::nn::MaxPool2dOptions maxpool_options(kernel_size);
    maxpool_options.stride(stride);
    return maxpool_options;
}

torch::nn::Sequential mcom::make_features(std::vector<int>& cfg, bool batch_norm, int in_channels) {
    torch::nn::Sequential features;
    //int in_channels = 3;
    for (auto v : cfg) {
        if (v == -1) {
            features->push_back(torch::nn::MaxPool2d(maxpool_options(2, 2)));
        }
        else {
            auto conv2d = torch::nn::Conv2d(conv_options(in_channels, v, 3, 1, 1 ));
            features->push_back(conv2d);
            if (batch_norm) {
                features->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(v)));
            }
            features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
            in_channels = v;
        }
    }
    return features;


}
 


