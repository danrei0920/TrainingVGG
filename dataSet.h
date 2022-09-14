#ifndef DATASET_H
#define DATASET_H


#undef slots
#include <torch/script.h>
#include <torch/torch.h>
#define slots Q_SLOTS

#include <vector>
#include <string>
#include <io.h>

#include <opencv2/opencv.hpp>


class dataSetClc:public torch::data::Dataset<dataSetClc>{
public:
    explicit dataSetClc(std::string image_dir, std::string type);
    torch::data::Example<> get(size_t index) override; 
    torch::optional<size_t> size() const override;

private:
    void load_data_from_folder(std::string image_dir, std::string type, int label);
    std::vector<std::string> _list_images;
    std::vector<int> _list_labels;
};



#endif // DATASET_H
