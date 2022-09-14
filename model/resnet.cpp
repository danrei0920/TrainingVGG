#include "resnet.h"
#include "modelcom.h"


BlockImpl::BlockImpl(int64_t inplanes, int64_t planes, int64_t stride_,
    torch::nn::Sequential downsample_, int groups, int base_width, bool _is_basic)
{
    downsample = downsample_;
    stride = stride_;
    int width = int(planes * (base_width / 64.)) * groups;

    conv1 = torch::nn::Conv2d(mcom::conv_options(inplanes, width, 3, stride_, 1, groups, false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
    conv2 = torch::nn::Conv2d(mcom::conv_options(width, width, 3, 1, 1, groups, false));
    bn2 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(width));
    is_basic = _is_basic;
    if (!is_basic) {
        conv1 = torch::nn::Conv2d(mcom::conv_options(inplanes, width, 1, 1, 0, 1, false));
        conv2 = torch::nn::Conv2d(mcom::conv_options(width, width, 3, stride_, 1, groups, false));
        conv3 = torch::nn::Conv2d(mcom::conv_options(width, planes * 4, 1, 1, 0, 1, false));
        bn3 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(planes * 4));
    }

    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("conv2", conv2);
    register_module("bn2", bn2);
    if (!is_basic) {
        register_module("conv3", conv3);
        register_module("bn3", bn3);
    }

    if (!downsample->is_empty()) {
        register_module("downsample", downsample);
    }
}

torch::Tensor BlockImpl::forward(torch::Tensor x) {
    torch::Tensor residual = x.clone();

    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);

    x = conv2->forward(x);
    x = bn2->forward(x);

    if (!is_basic) {
        x = torch::relu(x);
        x = conv3->forward(x);
        x = bn3->forward(x);
    }

    if (!downsample->is_empty()) {
        residual = downsample->forward(residual);
    }

    x += residual;
    x = torch::relu(x);

    return x;
}


torch::nn::Sequential ResNetImpl::_make_layer(int64_t planes, int64_t blocks, int64_t stride) {

    torch::nn::Sequential downsample;
    if (stride != 1 || inplanes != planes * expansion) {
        downsample = torch::nn::Sequential(
            torch::nn::Conv2d(mcom::conv_options(inplanes, planes * expansion, 1, stride, 0, 1, false)),
            torch::nn::BatchNorm2d(planes * expansion)
        );
    }
    torch::nn::Sequential layers;
    layers->push_back(Block(inplanes, planes, stride, downsample, groups, base_width, is_basic));
    inplanes = planes * expansion;
    for (int64_t i = 1; i < blocks; i++) {
        layers->push_back(Block(inplanes, planes, 1, torch::nn::Sequential(), groups, base_width, is_basic));
    }

    return layers;
}

ResNetImpl::ResNetImpl(std::vector<int> layers, int num_classes, std::string model_type, int _groups, int _width_per_group)
{
    if (model_type != "resnet18" && model_type != "resnet34") {
        expansion = 4;
        is_basic = false;
    }
    groups = _groups;
    base_width = _width_per_group;
    conv1 = torch::nn::Conv2d(mcom::conv_options(3, 64, 7, 2, 3, 1, false));
    bn1 = torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(64));
    layer1 = torch::nn::Sequential(_make_layer(64, layers[0]));
    layer2 = torch::nn::Sequential(_make_layer(128, layers[1], 2));
    layer3 = torch::nn::Sequential(_make_layer(256, layers[2], 2));
    layer4 = torch::nn::Sequential(_make_layer(512, layers[3], 2));

    fc = torch::nn::Linear(512 * expansion, num_classes);
    register_module("conv1", conv1);
    register_module("bn1", bn1);
    register_module("layer1", layer1);
    register_module("layer2", layer2);
    register_module("layer3", layer3);
    register_module("layer4", layer4);
    register_module("fc", fc);
}

torch::Tensor  ResNetImpl::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = bn1->forward(x);
    x = torch::relu(x);
    x = torch::max_pool2d(x, 3, 2, 1);

    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = layer4->forward(x);

    x = torch::avg_pool2d(x, 7, 1);
    x = x.view({ x.sizes()[0], -1 });
    x = fc->forward(x);

    return torch::log_softmax(x, 1);
}

ResNet resnet34(int num_classes) {
    std::vector<int> cfg_dd = { 3, 4, 6, 3 };
    ResNet resnet = ResNet(cfg_dd, num_classes, "resnet34",1, 64);
    return resnet;
}

ResNet resnet50(int num_classes) {
    std::vector<int> cfg_dd = { 3, 4, 6, 3 };
    ResNet resnet = ResNet(cfg_dd, num_classes, "resnet50",1, 64);
    return resnet;
}