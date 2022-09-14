#include "vgg.h"
#include "modelcom.h"


VGGImpl::VGGImpl(std::vector<int> &cfg, int num_classes, bool batch_norm, int in_channels){
    _features = mcom::make_features(cfg, batch_norm, in_channels);
    _avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
    _classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)));
    _classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    _classifier->push_back(torch::nn::Dropout());
    _classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)));
    _classifier->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
    _classifier->push_back(torch::nn::Dropout());
    _classifier->push_back(torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes)));

    _features = register_module("features",_features);
    _classifier = register_module("classifier",_classifier);
}

torch::Tensor VGGImpl::forward(torch::Tensor x){
    x = _features->forward(x);
    x = _avgpool(x);
    x = torch::flatten(x,1);

    x = _classifier->forward(x);
    return torch::log_softmax(x, 1);

}

VGG vgg16bn(int num_classes) {
    std::vector<int> cfg_dd = { 64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1 };
    VGG vgg = VGG(cfg_dd, num_classes, true, 3);
    return vgg;
}

VGG vgg19bn(int num_classes) {
    std::vector<int> cfg_dd = { 64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1 };
    VGG vgg = VGG(cfg_dd, num_classes, true, 3);
    return vgg;
}