#include <iostream>
#include <opencv.hpp>
#include <opencv2/ml.hpp>



void main()
{
	char path[90];
	// 保存训练数据，行数为图片数量
	cv::Mat trainingMat;
	cv::Mat inputImg;
	cv::Mat p;
	int num = 1;

	// 加载75个正样本
	while (num <= 75)
	{
		sprintf(path, "C:\\Users\\Administrator\\Desktop\\CarNumber\\CarNumber\\Database_SVM\\posdata\\0000 (%d).bmp", num);
		inputImg = cv::imread(path, 0);
		if (inputImg.empty())
		{
			std::cout << "无法加载正样本！";
			return;
		}
		p = inputImg.reshape(1, 1);
		p.convertTo(p, CV_32FC1);
		trainingMat.push_back(p);
		++num;
	}

	// 加载负样本
	while (num <= 120)
	{
		sprintf(path, "C:\\Users\\Administrator\\Desktop\\CarNumber\\CarNumber\\Database_SVM\\negdata\\0000 (1).bmp", num);
		inputImg = cv::imread(path, 0);
		if (inputImg.empty())
		{
			std::cout << "无法加载正样本！";
			return;
		}
		p = inputImg.reshape(1, 1);
		p.convertTo(p, CV_32FC1);
		trainingMat.push_back(p);
		++num;
	}

	// 训练数据对应的标签
	cv::Mat label(120, 1, CV_32FC1);
	for (int i = 0 ; i < 120 ; ++i)
	{
		if (i < 75)
		{
			label.at<float>(i, 0) = 1.0;
		}
		else
		{
			label.at<float>(i, 0) = -1.0;
		}
	}

	CvSVM svm_classifier;
	CvSVMParams svm_params;
	// 使用线性划分
	svm_params.kernel_type = CvSVM::LINEAR;
	svm_classifier.train(trainingMat, label, cv::Mat(), cv::Mat(), svm_params); //SVM训练 
	svm_classifier.save("svm.xml");


	// 测试
	int result = svm_classifier.predict(trainingMat.row(10));
	std::cout << result << std::endl;

	result = svm_classifier.predict(trainingMat.row(100));
	std::cout << result << std::endl;

	// 加载已训练好的模型 
	CvSVM read_svm;
	cv::FileStorage svm_fs("svm.xml", cv::FileStorage::READ);
	if (svm_fs.isOpened())
	{
		read_svm.load("svm.xml");
	}

	result = read_svm.predict(trainingMat.row(10));
	std::cout << result << std::endl;

	result = read_svm.predict(trainingMat.row(110));
	std::cout << result << std::endl;

	result = read_svm.predict(trainingMat.row(70));
	std::cout << result << std::endl;
}