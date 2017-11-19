#include "opencv2/core/core.hpp"
#include "opencv2//highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <time.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

using namespace cv;


void RgbConvToGray(const Mat& inputImage,Mat & outpuImage); //rgbתΪ�Ҷ�


void posDetect(Mat &, vector <RotatedRect> &); //�ֲ�ѡȡ��ѡ��������
bool verifySizes(const RotatedRect & );  //����������Ҫ���������
void posDetect_closeImg(Mat &inputImage , vector <RotatedRect> & rects  ) ;  //���ǵ����ƾ���ǳ�����ʱ������
bool verifySizes_closeImg(const RotatedRect & candidate); //�����ʱ�ĳ���������Ҫ���������     


void optimPosDetect(vector <RotatedRect> & rects_sImg , vector <RotatedRect> & rects_grayImg, //���������һ����λ
	vector <RotatedRect> & rects_closeImg,vector <RotatedRect> & rects_optimal );
float calOverlap(const Rect& box1,const Rect& box2);  //����2��������ص�����


void normalPosArea(Mat &intputImg, vector<RotatedRect> &rects_optimal, vector <Mat>& output_area );  //���Ʋü�����׼��Ϊ144*33


void svm_train(CvSVM & );  //ȡ��SVM.xml�е���������ͱ�ǩ�������ѵ��


void char_segment(const Mat & inputImg,vector <Mat>&); //�Գ��������е��ַ����зָ�
bool char_verifySizes(const RotatedRect &);   //�ַ�������Ҫ���������
//void char_sort(vector <RotatedRect > & in_char ); //���ַ������������

// ����map����
void char_sort(vector<RotatedRect > & in_char);


void features(const Mat & in , Mat & out ,int sizeData);  //���һ���ַ������Ӧ����������
Mat projectHistogram(const Mat& img ,int t);             //����ˮƽ���ۼ�ֱ��ͼ��ȡ����tΪ0����1


void ann_train(CvANN_MLP &ann ,int numCharacters, int nlayers); //ȡ��ann_xml�е����ݣ�������������ѵ��
void classify(CvANN_MLP& ann, vector<Mat> &char_feature , vector<int> & char_result); //ʹ��������ģ��Ԥ�⳵���ַ�������ӡ����Ļ

bool cmp_by_value(const std::pair<int, double> &lhs, const std::pair<int, double> &rhs);