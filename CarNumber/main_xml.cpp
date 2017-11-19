#include "opencv2/opencv.hpp"

using namespace cv;
#include <iostream>

int main()
{
	FileStorage fs("SVM.xml", FileStorage::WRITE); ;
	if (!fs.isOpened())  
	{  
		std::cerr << "failed to open " << std::endl;  
	} 

	int num = 1;
	char path[90];
	Mat combineMat; 
	Mat img_input,p;

	while(num <76) //1-75张正图片
	{	
		sprintf( path ,"E:\\PracticeOfOpencv\\projectCarNumber3_2\\CarNumber\\Database_SVM\\posdata\\0000 (%d).bmp" , num);
		img_input = imread(path,-1);
		p = img_input.reshape(1,1);
		p.convertTo(p, CV_32FC1);
		combineMat.push_back(p);
		num++;
	}

	num = 1;
	while(num <121) //1-120张负图片
	{	
		sprintf( path ,"E:\\PracticeOfOpencv\\projectCarNumber3_2\\CarNumber\\Database_SVM\\negdata\\0000 (%d).bmp" , num);
		img_input = imread(path,-1);
		p = img_input.reshape(1,1);
		p.convertTo(p, CV_32FC1);
		combineMat.push_back(p);
		num++;
	}
	fs<<"TrainingData"<<combineMat;

	Mat label(195,1,CV_32FC1);
	for (int i = 0 ;i<label.rows ;++ i)
	{
		if( i < 75)
			label.at<float>(i,0) =  1.0;
		else
			label.at<float>(i,0) =  -1.0;
	}
	fs<<"classes"<<label;
	fs.release();


	return 0;
}