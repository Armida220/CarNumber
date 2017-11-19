#include "carID_Detection.h"


int main(int argc, char* argv[])
{
	Mat img_input = imread("testCarID.jpg");
	//如果读入图像失败
	if (img_input.empty())
	{
		fprintf(stderr, "Can not load image %s\n", "testCarID.jpg");
		return -1;
	}

	Mat hsvImg;
	cvtColor(img_input, hsvImg, CV_BGR2HSV);
	vector<Mat> planes;
	split(hsvImg, planes);
	Mat sImg;
	sImg = planes[1];  //获得红色分量

	blur(sImg, sImg, Size(3, 3));   //3*3高斯滤波
	vector <RotatedRect>  rects_sImg;
	posDetect(sImg, rects_sImg);


	Mat grayImg;
	RgbConvToGray(img_input, grayImg);
	medianBlur(grayImg, grayImg, 3);   //3*3中值滤波

	vector <RotatedRect>  rects_grayImg;
	posDetect(grayImg, rects_grayImg);

	vector <RotatedRect>  rects_closeImg;  //车牌区域较为贴近
	posDetect_closeImg(sImg, rects_closeImg);

	vector <RotatedRect>  rects_optimal;
	optimPosDetect(rects_sImg, rects_grayImg, rects_closeImg, rects_optimal);

	// 车牌裁剪归一化
	vector <Mat> output_area;
	normalPosArea(img_input, rects_optimal, output_area);  //获得144*33的候选车牌区域output_area

	CvSVM  svmClassifier;

	svm_train(svmClassifier);  //使用SVM对正负样本进行训练

	vector<Mat> plates_svm;   //需要把候选车牌区域output_area图像中每个像素点作为一行特征向量，后进行预测
	for (int i = 0; i < output_area.size(); ++i)
	{
		Mat img = output_area[i];
		Mat p = img.reshape(1, 1);
		p.convertTo(p, CV_32FC1);
		int response = (int)svmClassifier.predict(p);
		if (response == 1)
			plates_svm.push_back(output_area[i]);    //保存预测结果
	}

	if (plates_svm.size() != 0)
	{
		// 		imshow("Test", plates_svm[0]);     //正确预测的话，就只有一个结果plates_svm[0]
		// 		waitKey(0);
	}
	else
	{
		std::cout << "定位失败";
		return -1;
	}

	//从SVM预测获得得车牌区域中分割得字符区域
	vector <Mat> char_seg;
	char_segment(plates_svm[0], char_seg);

// 	for (int i = 0 ; i < char_seg.size() ; ++i)
// 	{
// 		char fname[40];
// 		sprintf(fname, "%d", i);
// 		cv::imshow(fname, char_seg[i]);
// 	}
// 	cv::waitKey(0);

	//获得7个字符矩阵的相应特征矩阵
	vector <Mat> char_feature;
	char_feature.resize(7);
	for (int i = 0; i < char_seg.size(); ++i)
		features(char_seg[i], char_feature[i], 5);

	//神经网络训练
	CvANN_MLP ann_classify;
	ann_train(ann_classify, 34, 48);   //34为样本的类别数，48为隐藏层的神经元数

	//字符预测
	vector<int>  char_result;
	classify(ann_classify, char_feature, char_result);


	//此函数等待按键，按键盘任意键就返回
	svmClassifier.clear();

	return 0;
}