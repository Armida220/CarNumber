#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;

#include <iostream>

int main()
{
	int num = 1;
	char path[90];

	Mat trainningMat; //行数为样本数170，每个样本含144*33=4752个数，也即有4752列
	Mat inputImg;
	Mat p;

	//先加载100个正样本
	while(num < 101)
	{
		sprintf( path ,"F:\\SVM_data\\posdata\\0000 (%d).bmp" , num);
		inputImg = imread(path,-1);
		if (inputImg.empty())
		{
			std::cerr<<"无法加载正样本！";
			return -1;
		}
		p = inputImg.reshape(1,1);
		p.convertTo(p, CV_32FC1);
		trainningMat.push_back(p);
		num++;
	}
	num = 1;

	//再加载70个负样本
	while(num < 71)
	{
		sprintf( path ,"F:\\SVM_data\\negdata\\0000 (%d).bmp" , num);
		inputImg = imread(path,-1);
		if (inputImg.empty())
		{
			std::cerr<<"无法加载负样本！";
			return -1;
		}
		p = inputImg.reshape(1,1);
		p.convertTo(p, CV_32FC1);
		trainningMat.push_back(p);
		num++;
	}

	Mat label(170,1,CV_32FC1);   //与训练数据相应的标签,当然在txt中写数据再读出来也可以
	for (int i = 0 ;i<label.rows ;++ i)
	{
		if( i < 100)
			label.at<float>(i,0) =  1.0;
		else
			label.at<float>(i,0) =  -1.0;
	}

	CvSVM classifier;
	CvSVMParams SVM_params;
	SVM_params.kernel_type = CvSVM::LINEAR; //使用线性划分

	classifier.train(trainningMat,label ,Mat(),Mat(),SVM_params); //SVM训练


	vector<Mat> testdata; //定义测试数据

	num = 1;
	while(num < 4)
	{
		sprintf( path ,"F:\\SVM_data\\testdata\\postest%d.bmp" , num);
		inputImg = imread(path,-1);
		if (inputImg.empty())
		{
			std::cerr<<"无法加载正测试样本！";
			return -1;
		}
		p = inputImg.reshape(1,1);
		p.convertTo(p, CV_32FC1);
		testdata.push_back(p);
		num++ ;
	}

	num = 1;
	while(num < 4)
	{
		sprintf( path ,"F:\\SVM_data\\testdata\\negtest%d.bmp" , num);
		inputImg = imread(path,-1);
		if (inputImg.empty())
		{
			std::cerr<<"无法加载负测试样本！";
			return -1;
		}
		p = inputImg.reshape(1,1);
		p.convertTo(p, CV_32FC1);
		testdata.push_back(p);
		num++ ;
	}

	for (int i = 0;i < testdata.size() ; ++i)
	{
	    std::cout<<"测试样本"<<i+1<<"的测试结果为："
			<<(int)classifier.predict( testdata[i] )<<"\n";
	}

	return  0;
}