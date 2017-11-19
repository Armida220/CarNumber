#include "carID_Detection.h"


int main(int argc, char* argv[])
{
	Mat img_input = imread("testCarID.jpg");
	//�������ͼ��ʧ��
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
	sImg = planes[1];  //��ú�ɫ����

	blur(sImg, sImg, Size(3, 3));   //3*3��˹�˲�
	vector <RotatedRect>  rects_sImg;
	posDetect(sImg, rects_sImg);


	Mat grayImg;
	RgbConvToGray(img_input, grayImg);
	medianBlur(grayImg, grayImg, 3);   //3*3��ֵ�˲�

	vector <RotatedRect>  rects_grayImg;
	posDetect(grayImg, rects_grayImg);

	vector <RotatedRect>  rects_closeImg;  //���������Ϊ����
	posDetect_closeImg(sImg, rects_closeImg);

	vector <RotatedRect>  rects_optimal;
	optimPosDetect(rects_sImg, rects_grayImg, rects_closeImg, rects_optimal);

	// ���Ʋü���һ��
	vector <Mat> output_area;
	normalPosArea(img_input, rects_optimal, output_area);  //���144*33�ĺ�ѡ��������output_area

	CvSVM  svmClassifier;

	svm_train(svmClassifier);  //ʹ��SVM��������������ѵ��

	vector<Mat> plates_svm;   //��Ҫ�Ѻ�ѡ��������output_areaͼ����ÿ�����ص���Ϊһ�����������������Ԥ��
	for (int i = 0; i < output_area.size(); ++i)
	{
		Mat img = output_area[i];
		Mat p = img.reshape(1, 1);
		p.convertTo(p, CV_32FC1);
		int response = (int)svmClassifier.predict(p);
		if (response == 1)
			plates_svm.push_back(output_area[i]);    //����Ԥ����
	}

	if (plates_svm.size() != 0)
	{
		// 		imshow("Test", plates_svm[0]);     //��ȷԤ��Ļ�����ֻ��һ�����plates_svm[0]
		// 		waitKey(0);
	}
	else
	{
		std::cout << "��λʧ��";
		return -1;
	}

	//��SVMԤ���õó��������зָ���ַ�����
	vector <Mat> char_seg;
	char_segment(plates_svm[0], char_seg);

// 	for (int i = 0 ; i < char_seg.size() ; ++i)
// 	{
// 		char fname[40];
// 		sprintf(fname, "%d", i);
// 		cv::imshow(fname, char_seg[i]);
// 	}
// 	cv::waitKey(0);

	//���7���ַ��������Ӧ��������
	vector <Mat> char_feature;
	char_feature.resize(7);
	for (int i = 0; i < char_seg.size(); ++i)
		features(char_seg[i], char_feature[i], 5);

	//������ѵ��
	CvANN_MLP ann_classify;
	ann_train(ann_classify, 34, 48);   //34Ϊ�������������48Ϊ���ز����Ԫ��

	//�ַ�Ԥ��
	vector<int>  char_result;
	classify(ann_classify, char_feature, char_result);


	//�˺����ȴ�������������������ͷ���
	svmClassifier.clear();

	return 0;
}