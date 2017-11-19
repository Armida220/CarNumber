#include <iostream>
#include <opencv.hpp>
#include <opencv2/ml.hpp>



void main()
{
	char path[90];
	// ����ѵ�����ݣ�����ΪͼƬ����
	cv::Mat trainingMat;
	cv::Mat inputImg;
	cv::Mat p;
	int num = 1;

	// ����75��������
	while (num <= 75)
	{
		sprintf(path, "C:\\Users\\Administrator\\Desktop\\CarNumber\\CarNumber\\Database_SVM\\posdata\\0000 (%d).bmp", num);
		inputImg = cv::imread(path, 0);
		if (inputImg.empty())
		{
			std::cout << "�޷�������������";
			return;
		}
		p = inputImg.reshape(1, 1);
		p.convertTo(p, CV_32FC1);
		trainingMat.push_back(p);
		++num;
	}

	// ���ظ�����
	while (num <= 120)
	{
		sprintf(path, "C:\\Users\\Administrator\\Desktop\\CarNumber\\CarNumber\\Database_SVM\\negdata\\0000 (1).bmp", num);
		inputImg = cv::imread(path, 0);
		if (inputImg.empty())
		{
			std::cout << "�޷�������������";
			return;
		}
		p = inputImg.reshape(1, 1);
		p.convertTo(p, CV_32FC1);
		trainingMat.push_back(p);
		++num;
	}

	// ѵ�����ݶ�Ӧ�ı�ǩ
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
	// ʹ�����Ի���
	svm_params.kernel_type = CvSVM::LINEAR;
	svm_classifier.train(trainingMat, label, cv::Mat(), cv::Mat(), svm_params); //SVMѵ�� 
	svm_classifier.save("svm.xml");


	// ����
	int result = svm_classifier.predict(trainingMat.row(10));
	std::cout << result << std::endl;

	result = svm_classifier.predict(trainingMat.row(100));
	std::cout << result << std::endl;

	// ������ѵ���õ�ģ�� 
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