#include "Classification.h"
#include <vector>
#include <dataSet.h>

void Classifier::load_traindata()
{
    //cv::Mat image;
    //image = cv::imread("C:/LIBS/datasets/kagglecatsanddogs_3367a/PetImages/Cat/2.jpg");// , CV_LOAD_IMAGE_COLOR);
    //// 2. convert color space, opencv read the image in BGR
    //cv::cvtColor(image, image, CV_BGR2RGB);
    //cv::Mat img_float;
    //// convert to float format
    //image.convertTo(img_float, CV_32F, 1.0 / 255);
    //// 3. resize the image for resnet101 model
    //cv::resize(img_float, img_float, cv::Size(224, 224), cv::INTER_AREA);
    //// 4. transform to tensor
    //auto img_tensor = torch::from_blob(img_float.data, { 1,224,224,3 }, torch::kFloat32);
    //// in pytorch, batch first, then channel
    //img_tensor = img_tensor.permute({ 0,3,1,2 });
    //// 5. Removing mean values of the RGB channels
    //// the values are from following link.
    //// https://github.com/pytorch/examples/blob/master/imagenet/main.py#L202
    //img_tensor[0][0] = img_tensor[0][0].sub_(0.485).div_(0.229);
    //img_tensor[0][1] = img_tensor[0][1].sub_(0.456).div_(0.224);
    //img_tensor[0][2] = img_tensor[0][2].sub_(0.406).div_(0.225);

    //// Create vectors of inputs.
    //std::vector<torch::jit::IValue> inputs1, inputs2;
    //inputs1.push_back(torch::ones({ 1,3,224,224 }));
    //inputs2.push_back(img_tensor);


}

Classifier::Classifier(int gpu_id){
    if (gpu_id >= 0) {
        device = torch::Device(torch::kCUDA, gpu_id);
    }
    else {
        device = torch::Device(torch::kCPU);
    }
}

void Classifier::Initialize(int num_epochs, int batch_size, float learning_rate, std::string train_val_dir, std::string image_type){
    try {
        auto custom_dataset_train = dataSetClc(train_val_dir, image_type);
        //std::vector<int> cfg_d = { 64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1 };
        //vgg_train = VGG(cfg_d, _num_classes, true);
        //
        //if (train) {
        //    torch::jit::script::Module vgg_pretrain;
        //    vgg_pretrain = torch::jit::load("C:/LIBS/CNN/resnet34_jt.pt");
        //}

       
        //// 6. Execute the model and turn its output into a tensor
        //at::Tensor output = vgg_pretrain.forward(inputs2).toTensor();
        //std::cout << output.sizes() << std::endl;
        //std::cout << output.slice(/*dim=*/1,/*start=*/0,/*end=*/3) << '\n';

        //// 7. Load labels
        //std::string label_file = "C:/LIBS/datasets/synset_words.txt";
        //std::ifstream rf(label_file.c_str());
        //CHECK(rf) << "Unable to open labels file" << label_file;
        //std::string line;
        //std::vector<std::string> labels;
        //while (std::getline(rf, line)) { labels.push_back(line); }

        //// 8. print predicted top-3 labels
        //std::tuple<torch::Tensor, torch::Tensor> result = output.sort(-1, true);
        //torch::Tensor top_scores = std::get<0>(result)[0];
        //torch::Tensor top_idxs = std::get<1>(result)[0].toType(torch::kInt32);

        //auto top_scores_a = top_scores.accessor<float, 1>();
        //auto top_idxs_a = top_idxs.accessor<int, 1>();
        //for (int i = 0; i < 3; i++) {
        //    int idx = top_idxs_a[i];
        //    std::cout << "top-" << i + 1 << " label: ";
        //    std::cout << labels[idx] << ",score: " << top_scores_a[i] << std::endl;
        //}

    }
    catch (const c10::Error& e) {
        std::cout << e.msg() << std::endl;
    }
    return;
}

void Classifier::Train(int num_epochs, int batch_size, float learning_rate, std::string train_val_dir, std::string image_type, std::string save_path){

    try
    {
        //vgg_train = vgg16bn(3);
        //auto dict16bn = vgg_train->named_parameters();
        resnet_train = resnet34(3);
        auto resnet34 = resnet_train->named_parameters();
        
        for (auto n = resnet34.begin(); n != resnet34.end(); n++)
        {
            std::cout << (*n).key() << std::endl;
        }
        auto custom_dataset_train = dataSetClc(train_val_dir, image_type).map(torch::data::transforms::Stack<>());
        auto data_loader_train = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset_train), batch_size);    
   
        float loss_train = 0; float loss_val = 0;
        float acc_train = 0.0; float acc_val = 0.0; float best_acc = 0.0;

        for (size_t epoch = 1; epoch <= num_epochs; ++epoch) {
            size_t batch_index_train = 0;
            size_t batch_index_val = 0;
            if (epoch == int(num_epochs / 2)) { learning_rate /= 10; }
            torch::optim::Adam optimizer(resnet_train->parameters(), learning_rate); // Learning Rate

 /*           if (epoch < int(num_epochs / 8)) {
                for (auto mm : vgg_train->named_parameters()) {
                    if (strstr(mm.key().data(), "classifier")) {
                        mm.value().set_requires_grad(true);
                    }
                    else {
                        mm.value().set_requires_grad(false);
                    }
                }
            }
            else {
                for (auto mm : vgg_train->named_parameters()) {
                    mm.value().set_requires_grad(true);
                }
            }*/
      
    
            for (auto& batch : *data_loader_train) {
                auto data = batch.data;
                auto target = batch.target;
           
                data = data.to(torch::kF32).to(device);
                target = target.squeeze_().to(torch::kInt64).to(device);
                optimizer.zero_grad();

                // Execute the model
                torch::Tensor prediction = resnet_train->forward(data);
                //std::cout << prediction.argmax(1).eq(target) << std::endl;
                auto acc = prediction.argmax(1).eq(target).sum().item<float>();
                //std::cout << acc << std::endl;
                //acc_train += acc.template item<float>() / batch_size;

                acc_train += acc / batch_size;
                // Compute loss value

                //std::cout << prediction << std::endl;
                //std::cout << target << std::endl;
           

                torch::Tensor loss = torch::nll_loss(prediction, target.squeeze());

                //std::cout << loss.item<float>() << std::endl;
                //torch::Tensor loss = torch::nll_loss(prediction, b);
                // Compute gradients
                loss.backward();
                
                // Update the parameters
                optimizer.step();
                loss_train = loss.item<float>();
                batch_index_train++;
                std::cout << "Epoch: " << epoch << " |Train Loss: " <<  loss_train  << " |Train Acc:" << acc_train / batch_index_train << "\r";
            }
            std::cout << std::endl;

            //validation part
            //vgg->eval();
            //for (auto& batch : *data_loader_val) {
            //    auto data = batch.data;
            //    auto target = batch.target.squeeze();
            //    data = data.to(torch::kF32).to(device).div(255.0);
            //    target = target.to(torch::kInt64).to(device);
            //    torch::Tensor prediction = vgg->forward(data);
            //    // Compute loss value
            //    torch::Tensor loss = torch::nll_loss(prediction, target);
            //    auto acc = prediction.argmax(1).eq(target).sum();
            //    acc_val += acc.template item<float>() / batch_size;
            //    loss_val += loss.item<float>();
            //    batch_index_val++;
            //    std::cout << "Epoch: " << epoch << " |Val Loss: " << loss_val / batch_index_val << " |Valid Acc:" << acc_val / batch_index_val << "\r";
            //}
            //std::cout << std::endl;


            //if (acc_val > best_acc) {
            //    torch::save(vgg, save_path);
            //    best_acc = acc_val;
            //}
            loss_train = 0; loss_val = 0; acc_train = 0; acc_val = 0; batch_index_train = 0; batch_index_val = 0;
        }
    }
    catch (const c10::Error& e)
    {
        std::cout << e.msg() << std::endl;
    }
}

int Classifier::Predict(cv::Mat& image){
    //cv::resize(image, image, cv::Size(448, 448));
    //torch::Tensor img_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 });
    //img_tensor = img_tensor.to(device).unsqueeze(0).to(torch::kF32).div(255.0);
    //auto prediction = vgg->forward(img_tensor);
    //prediction = torch::softmax(prediction,1);
    //auto class_id = prediction.argmax(1);
    //std::cout<<prediction<<class_id;
    //int ans = int(class_id.item().toInt());
    //float prob = prediction[0][ans].item().toFloat();
    //return ans;

    return 0;
}

void Classifier::LoadWeight(std::string weight){
    //torch::load(vgg,weight);
    //vgg->eval();
    //return;
}
