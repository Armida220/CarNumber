#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;

#include <iostream>

int main()
{
	int num = 1;
	char path[90];

	Mat trainningMat; //����Ϊ������170��ÿ��������144*33=4752������Ҳ����4752��
	Mat inputImg;
	Mat p;

	//�ȼ���100��������
	while(num < 101)
	{
		sprintf( path ,"F:\\SVM_data\\posdata\\0000 (%d).bmp" , num);
		inputImg = imread(path,-1);
		if (inputImg.empty())
		{
			std::cerr<<"�޷�������������";
			return -1;
		}
		p = inputImg.reshape(1,1);
		p.convertTo(p, CV_32FC1);
		trainningMat.push_back(p);
		num++;
	}
	num = 1;

	//�ټ���70��������
	while(num < 71)
	{
		sprintf( path ,"F:\\SVM_data\\negdata\\0000 (%d).bmp" , num);
		inputImg = imread(path,-1);
		if (inputImg.empty())
		{
			std::cerr<<"�޷����ظ�������";
			return -1;
		}
		p = inputImg.reshape(1,1);
		p.convertTo(p, CV_32FC1);
		trainningMat.push_back(p);
		num++;
	}

	Mat label(170,1,CV_32FC1);   //��ѵ��������Ӧ�ı�ǩ,��Ȼ��txt��д�����ٶ�����Ҳ����
	for (int i = 0 ;i<label.rows ;++ i)
	{
		if( i < 100)
			label.at<float>(i,0) =  1.0;
		else
			label.at<float>(i,0) =  -1.0;
	}

	CvSVM classifier;
	CvSVMParams SVM_params;
	SVM_params.kernel_type = CvSVM::LINEAR; //ʹ�����Ի���

	classifier.train(trainningMat,label ,Mat(),Mat(),SVM_params); //SVMѵ��


	vector<Mat> testdata; //�����������

	num = 1;
	while(num < 4)
	{
		sprintf( path ,"F:\\SVM_data\\testdata\\postest%d.bmp" , num);
		inputImg = imread(path,-1);
		if (inputImg.empty())
		{
			std::cerr<<"�޷�����������������";
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
			std::cerr<<"�޷����ظ�����������";
			return -1;
		}
		p = inputImg.reshape(1,1);
		p.convertTo(p, CV_32FC1);
		testdata.push_back(p);
		num++ ;
	}

	for (int i = 0;i < testdata.size() ; ++i)
	{
	    std::cout<<"��������"<<i+1<<"�Ĳ��Խ��Ϊ��"
			<<(int)classifier.predict( testdata[i] )<<"\n";
	}

	return  0;
}