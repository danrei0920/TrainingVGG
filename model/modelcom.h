#ifndef MODELCOM_H
#define MODELCOM_H



//#pragma once

#undef slots
#include<torch/script.h>
#include<torch/torch.h>
#define slots Q_SLOTS



namespace mcom {

	inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
		int64_t stride = 1, int64_t padding = 0, int groups = 1, bool with_bias = true);

	inline torch::nn::MaxPool2dOptions maxpool_options(int kernel_size, int stride);

	torch::nn::Sequential make_features(std::vector<int>& cfg, bool batch_norm, int in_channels);



};



#endif