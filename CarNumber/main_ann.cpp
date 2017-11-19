#include "opencv2/opencv.hpp"
#include "carID_Detection.h"

using namespace cv;
#include <iostream>

int main()
{
	FileStorage fs("ann_xml.xml", FileStorage::WRITE); ;
	if (!fs.isOpened())  
	{  
		std::cerr << "failed to open " << std::endl;  
	} 

	Mat  trainData;
	Mat classes = Mat::zeros(1,1700,CV_8UC1);
	char path[90];
	Mat img_read;
	for (int i = 0;i<34 ;i++)  //第i类
	{
		for (int j=1 ; j< 51 ; ++j)  //i类中第j个
		{
			sprintf( path ,"E:\\PracticeOfOpencv\\projectCarNumber3_2\\CarNumber\\charSamples\\%d\\%d (%d).png" , i,i,j);
			img_read = imread(path , -1);

			Mat img_threshold;
			threshold(img_read ,img_threshold , 180,255 ,CV_THRESH_BINARY );

			Mat dst_mat;
			Mat train_mat(2,3,CV_32FC1);
			int length ;
			Point2f srcTri[3];  
			Point2f dstTri[3];

			srcTri[0] = Point2f( 0,0 );  
			srcTri[1] = Point2f( img_threshold.cols - 1, 0 );  
			srcTri[2] = Point2f( 0, img_threshold.rows - 1 );
			length = img_threshold.rows > img_threshold.cols?img_threshold.rows:img_threshold.cols;
			dstTri[0] = Point2f( 0.0, 0.0 );  
			dstTri[1] = Point2f( length, 0.0 );  
			dstTri[2] = Point2f( 0.0, length ); 
			train_mat = getAffineTransform( srcTri, dstTri );
			dst_mat = Mat::zeros(length,length,img_threshold.type());		
			warpAffine(img_threshold,dst_mat,train_mat,dst_mat.size(),INTER_LINEAR,BORDER_CONSTANT,Scalar(0));
			resize(dst_mat,dst_mat,Size(20,20));  //尺寸调整为20*20

			Mat dst_feature;
			features(dst_mat,dst_feature,5); //生成1*440特征向量

			trainData.push_back(dst_feature);
			classes.at<uchar>(i*50 + j -1) = i; 
		}
	}

	
	fs<<"TrainingData"<<trainData;
	fs<<"classes"<<classes;
	fs.release();
}