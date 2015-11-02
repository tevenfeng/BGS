#pragma once
#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//定义gmm模型用到的变量
#define GMM_MAX_COMPONT 6			//每个GMM最多的高斯模型个数
#define GMM_LEARN_ALPHA 0.005  
#define GMM_THRESHOD_SUMW 0.7
#define TRAIN_FRAMES 60	// 对前 TRAIN_FRAMES 帧建模

class MOG_BGS
{
public:
	MOG_BGS(void);
	~MOG_BGS(void);

	void init(const Mat _image);
	void processFirstFrame(const Mat _image);
	void trainGMM(const Mat _image);
	void getFitNum(const Mat _image);
	void testGMM(const Mat _image);
	Mat getMask(void) { return m_mask; };

private:
	Mat m_weight[GMM_MAX_COMPONT];  //权值
	Mat m_mean[GMM_MAX_COMPONT];    //均值
	Mat m_sigma[GMM_MAX_COMPONT];   //方差

	Mat m_mask;
	Mat m_fit_num;
};