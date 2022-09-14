#include<dataSet.h>
#include <memory>
#include <exception>
dataSetClc::dataSetClc(std::string image_dir, std::string type) {
    load_data_from_folder(image_dir, std::string(type), 0);
}

void dataSetClc::load_data_from_folder(std::string path, std::string type, int label) {
    long long hFile = 0; 
    struct _finddata_t fileInfo;
    std::string pathName;
    if ((hFile = _findfirst(pathName.assign(path).append("/*.*").c_str(), &fileInfo)) == -1) {
        return;
    }
    do {
        const char* s = fileInfo.name;
        const char* t = type.data();

        if (fileInfo.attrib&_A_SUBDIR) {
            if (strcmp(s, ".") == 0 || strcmp(s, "..") == 0) { continue; }
            std::string sub_path = path + "/" + fileInfo.name;
            ++label;
            load_data_from_folder(sub_path, type, label);

        }
        else {
            if (strstr(s, t)) {
                std::string image_path = path + "/" + fileInfo.name;
                _list_images.push_back(image_path);
                _list_labels.push_back(label);
            }
        }
    } while (_findnext(hFile, &fileInfo) == 0);  
    return;
}

torch::data::Example<> dataSetClc::get(size_t index) {
  
    try {
        cv::Mat image = cv::imread(_list_images.at(index), 1);
        if (!image.data) { 

            std::ifstream fin(_list_images.at(index));
            if (!fin) {
                throw("image file is not exist."); 
            }
            cv::VideoCapture capture;
            cv::Mat imggif;
            imggif = capture.open(_list_images.at(index));
            if (!capture.isOpened()) {
                throw("read image error, please check image format.");
            }
            std::cout << "try gif format image reading." << std::endl;
            while (capture.read(imggif))
            {
                image = imggif.clone();
            }
            capture.release();
            if (image.empty()) {
                throw("read image error, please check image format.");
            }
        }


        cv::resize(image, image, cv::Size(224, 224), cv::INTER_CUBIC);


        std::vector<double> norm_mean = { 0.485,0.456,0.406 };
        std::vector<double> norm_std = { 0.229,0.224,0.225 };

        torch::Tensor img_tensor = torch::from_blob(image.data, { 224, 224, 3}, torch::kByte).permute({2, 0, 1}).toType(torch::kFloat).div(255);
        img_tensor = torch::data::transforms::Normalize<>(norm_mean, norm_std)(img_tensor);
        //label
        torch::Tensor label_tensor = torch::full({ 1 }, _list_labels.at(index));
        
        //return { img_tensor.clone(), label_tensor.clone()};
        return { std::move(img_tensor), std::move(label_tensor)};
    }
    
    catch (std::string& e) {
        std::cout << e << std::endl;
    }
    catch (std::exception& e){
        std::cout << e.what() << std::endl;
    }

}

torch::optional<size_t> dataSetClc::size() const 
{
    return _list_images.size();
}