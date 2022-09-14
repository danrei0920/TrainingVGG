#include "mainwindow.h"

#include <QApplication>
#include <Classification.h>
#include "util/util.h"


void do_train() {

    std::string train_val_dir = "C:/LIBS/datasets/kagglecatsanddogs_3367a/PetImages"; 
    Classifier classifier(-1);
    //classifier.Initialize(2,vgg_path);

    //predict
    //classifier.LoadWeight("classifer.pt");
    //cv::Mat image = cv::imread(train_val_dir+"C:\\0_MEC\\0_CODE_SOURCE\\10_CMakeProject\\TrainingVGG\\x64-Release\\kagglecatsanddogs_3367a\\PetImages");
    //classifier.Predict(image);
    classifier.Train(3,20,0.01,train_val_dir,".jpg","classifer.pt");

    //std::vector<int> cfg_a = {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
    
    //classifier.Initialize(2, 5, 0.01, train_val_dir, "jpg");
    //classifier.Train(2, 32, 0.01, train_val_dir, "jpg", train_val_dir);
    //std::vector<int> cfg_d = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    //auto vgg = VGG(cfg_d,2,true);
    //auto dict = vgg->named_parameters();
    //torch::load(vgg, vgg_path);




}


int main(int argc, char *argv[])
{

    do_train();
    //util _util;
    //util.read_cifar_to_file("C:/LIBS/datasets/cifar-10-binary/cifar-10-batches-bin/data_batch_%d.bin");
    //do_train();
    //auto pavgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(7));
    //auto inp = torch::rand({1,3,7,7});
    //auto outp = pavgpool->forward(inp);
    //std::cout<<outp.sizes();
    //std::vector<int> cfg_dd = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    //auto vgg_dd = VGG(cfg_dd,3,true,3);
    //auto in = torch::rand({1,3,224,224});
    //auto dictdd = vgg_dd->named_parameters();
    //vgg_dd->forward(in);
    //for (auto n = dictdd.begin(); n != dictdd.end(); n++)
    //{
    //    std::cout<<(*n).key()<< std::endl;
    //}

    //std::string vgg_path = "C:\\0_MEC\\0_CODE_SOURCE\\10_CMakeProject\\TrainingVGG\\x64-Release\\binvgg16_bn.pt";
    //std::string train_val_dir = "C:\\LIBS\\datasets\\kagglecatsanddogs_3367a\\PetImages";

    //Classifier classifier(-1);
    //classifier.Initialize(2,vgg_path);

    //predict
    //classifier.LoadWeight("classifer.pt");
    //cv::Mat image = cv::imread(train_val_dir+"C:\\0_MEC\\0_CODE_SOURCE\\10_CMakeProject\\TrainingVGG\\x64-Release\\kagglecatsanddogs_3367a\\PetImages");
    //classifier.Predict(image);
    //classifier.Train(300,4,0.0003,train_val_dir,".jpg","classifer.pt");
    //std::vector<int> cfg_a = {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1};
    //classifier.Train(2, 1, 0.01, train_val_dir, "jpg", train_val_dir);
    //std::vector<int> cfg_d = {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1};
    //auto vgg = VGG(cfg_d,2,true);
    //auto dict = vgg->named_parameters();
    //torch::load(vgg, vgg_path);

    


    QApplication a(argc, argv);
    MainWindow w;
    w.show();
    return a.exec();
}




