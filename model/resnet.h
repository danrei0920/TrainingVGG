#pragma once

#undef slots
#include<torch/script.h>
#include<torch/torch.h>
#define slots Q_SLOTS

#include <vector>


class BlockImpl : public torch::nn::Module {
public:
	BlockImpl(int64_t inplanes, int64_t planes, int64_t stride_ = 1,
		torch::nn::Sequential downsample_ = nullptr, int groups = 1, int base_width = 64, bool is_basic = true);
	torch::Tensor forward(torch::Tensor x);
	torch::nn::Sequential downsample{ nullptr };
private:
	bool is_basic = true;
	int64_t stride = 1;
	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::BatchNorm2d bn1{ nullptr };
	torch::nn::Conv2d conv2{ nullptr };
	torch::nn::BatchNorm2d bn2{ nullptr };
	torch::nn::Conv2d conv3{ nullptr };
	torch::nn::BatchNorm2d bn3{ nullptr };
};
TORCH_MODULE(Block);

class ResNetImpl : public torch::nn::Module {
public:
    ResNetImpl(std::vector<int> layers, int num_classes = 1000, std::string model_type = "resnet18",
        int groups = 1, int width_per_group = 64);
    torch::Tensor forward(torch::Tensor x);
    //std::vector<torch::Tensor> features(torch::Tensor x);
    torch::nn::Sequential _make_layer(int64_t planes, int64_t blocks, int64_t stride = 1);

private:
    int expansion = 1; bool is_basic = true;
    int64_t inplanes = 64; int groups = 1; int base_width = 64;
    torch::nn::Conv2d conv1{ nullptr };
    torch::nn::BatchNorm2d bn1{ nullptr };
    torch::nn::Sequential layer1{ nullptr };
    torch::nn::Sequential layer2{ nullptr };
    torch::nn::Sequential layer3{ nullptr };
    torch::nn::Sequential layer4{ nullptr };
    torch::nn::Linear fc{ nullptr };
};
TORCH_MODULE(ResNet);


ResNet resnet34(int num_classes);
ResNet resnet50(int num_classes);