#include "opencv2/core/core.hpp"
#include "opencv2//highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <time.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

using namespace cv;


void RgbConvToGray(const Mat& inputImage,Mat & outpuImage); //rgb转为灰度


void posDetect(Mat &, vector <RotatedRect> &); //粗步选取候选车牌区域
bool verifySizes(const RotatedRect & );  //车牌区域需要满足的条件
void posDetect_closeImg(Mat &inputImage , vector <RotatedRect> & rects  ) ;  //考虑到车牌距离非常近的时候的情况
bool verifySizes_closeImg(const RotatedRect & candidate); //距离近时的车牌区域需要满足的条件     


void optimPosDetect(vector <RotatedRect> & rects_sImg , vector <RotatedRect> & rects_grayImg, //车牌区域进一步定位
	vector <RotatedRect> & rects_closeImg,vector <RotatedRect> & rects_optimal );
float calOverlap(const Rect& box1,const Rect& box2);  //计算2个矩阵的重叠比例


void normalPosArea(Mat &intputImg, vector<RotatedRect> &rects_optimal, vector <Mat>& output_area );  //车牌裁剪，标准化为144*33


void svm_train(CvSVM & );  //取出SVM.xml中的特征矩阵和标签矩阵进行训练


void char_segment(const Mat & inputImg,vector <Mat>&); //对车牌区域中的字符进行分割
bool char_verifySizes(const RotatedRect &);   //字符区域需要满足的条件
//void char_sort(vector <RotatedRect > & in_char ); //对字符区域进行排序

// 利用map排序
void char_sort(vector<RotatedRect > & in_char);


void features(const Mat & in , Mat & out ,int sizeData);  //获得一个字符矩阵对应的特征向量
Mat projectHistogram(const Mat& img ,int t);             //计算水平或累计直方图，取决于t为0还是1


void ann_train(CvANN_MLP &ann ,int numCharacters, int nlayers); //取出ann_xml中的数据，并进行神经网络训练
void classify(CvANN_MLP& ann, vector<Mat> &char_feature , vector<int> & char_result); //使用神经网络模型预测车牌字符，并打印至屏幕

bool cmp_by_value(const std::pair<int, double> &lhs, const std::pair<int, double> &rhs);