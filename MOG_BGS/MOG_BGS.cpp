// MOG_BGS.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"
// This is based on the "An Improved Adaptive Background Mixture Model for
// Real-time Tracking with Shadow Detection" by P. KaewTraKulPong and R. Bowden
// Author : zouxy
// Date   : 2013-4-13
// HomePage : http://blog.csdn.net/zouxy09
// Email  : zouxy09@qq.com

#include <opencv2\core\core.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\features2d\features2d.hpp>
#include "MOG_BGS.h"
#include <iostream>
#include <cstdio>

using namespace cv;
using namespace std;

MOG_BGS::MOG_BGS(void)
{

}

MOG_BGS::~MOG_BGS(void)
{

}

// ȫ����ʼ��Ϊ0
void MOG_BGS::init(const Mat _image)
{
	/****initialization the three parameters ****/
	for (int i = 0; i < GMM_MAX_COMPONT; i++)
	{
		m_weight[i] = Mat::zeros(_image.size(), CV_32FC1);
		m_mean[i] = Mat::zeros(_image.size(), CV_8UC1);
		m_sigma[i] = Mat::zeros(_image.size(), CV_32FC1);
	}
	m_mask = Mat::zeros(_image.size(), CV_8UC1);
	m_fit_num = Mat::ones(_image.size(), CV_8UC1);
}

//gmm��һ֡��ʼ������ʵ��
//���񵽵�һ֡ʱ�Ը�˹�ֲ����г�ʼ������Ҫ������ÿ����˹�ֲ���Ȩֵ�������ͷ����ֵ��
//���е�һ����˹�ֲ���ȨֵΪ1������Ϊ��һ���������ݣ������˹�ֲ�ȨֵΪ0������Ϊ0��
//ÿ����˹�ֲ����������ʵ�����ȵĳ�ʼ���� 15
void MOG_BGS::processFirstFrame(const Mat _image)
{
	for (int i = 0; i < GMM_MAX_COMPONT; i++)
	{
		if (i == 0)
		{
			m_weight[i].setTo(1.0);
			_image.copyTo(m_mean[i]);
			m_sigma[i].setTo(15.0);
		}
		else
		{
			m_weight[i].setTo(0.0);
			m_mean[i].setTo(0);
			m_sigma[i].setTo(15.0);
		}
	}
}

// ͨ���µ�֡��ѵ��GMM
void MOG_BGS::trainGMM(const Mat _image)
{
	for (int i = 0; i < _image.rows; i++)
	{
		for (int j = 0; j < _image.cols; j++)
		{
			int num_fit = 0;

			/**************************** Update parameters Start ******************************************/
			for (int k = 0; k < GMM_MAX_COMPONT; k++)
			{
				int delm = abs(_image.at<uchar>(i, j) - m_mean[k].at<uchar>(i, j));
				long dist = delm * delm;
				// �ж��Ƿ�ƥ�䣺����ֵ���˹�ֲ��ľ�ֵ�ľ���С��3�������ʾƥ�䣩
				if (dist < 3.0 * m_sigma[k].at<float>(i, j))
				{
					// ���ƥ��
					/****update the weight****/
					m_weight[k].at<float>(i, j) += GMM_LEARN_ALPHA * (1 - m_weight[k].at<float>(i, j));

					/****update the average****/
					m_mean[k].at<uchar>(i, j) += (GMM_LEARN_ALPHA / m_weight[k].at<float>(i, j)) * delm;

					/****update the variance****/
					m_sigma[k].at<float>(i, j) += (GMM_LEARN_ALPHA / m_weight[k].at<float>(i, j)) * (dist - m_sigma[k].at<float>(i, j));
				}
				else
				{
					// �����ƥ�䡣��øø�˹ģ�͵�Ȩֵ��С
					m_weight[k].at<float>(i, j) += GMM_LEARN_ALPHA * (0 - m_weight[k].at<float>(i, j));
					num_fit++; // ��ƥ���ģ�͸���
				}
			}
			/**************************** Update parameters End ******************************************/


			/*********************** Sort Gaussian component by 'weight / sigma' Start ****************************/
			//��gmm������˹��������,�Ӵ�С����,��������Ϊ weight / sigma
			for (int kk = 0; kk < GMM_MAX_COMPONT; kk++)
			{
				for (int rr = kk; rr< GMM_MAX_COMPONT; rr++)
				{
					if (m_weight[rr].at<float>(i, j) / m_sigma[rr].at<float>(i, j) > m_weight[kk].at<float>(i, j) / m_sigma[kk].at<float>(i, j))
					{
						//Ȩֵ����
						float temp_weight = m_weight[rr].at<float>(i, j);
						m_weight[rr].at<float>(i, j) = m_weight[kk].at<float>(i, j);
						m_weight[kk].at<float>(i, j) = temp_weight;

						//��ֵ����
						uchar temp_mean = m_mean[rr].at<uchar>(i, j);
						m_mean[rr].at<uchar>(i, j) = m_mean[kk].at<uchar>(i, j);
						m_mean[kk].at<uchar>(i, j) = temp_mean;

						//�����
						float temp_sigma = m_sigma[rr].at<float>(i, j);
						m_sigma[rr].at<float>(i, j) = m_sigma[kk].at<float>(i, j);
						m_sigma[kk].at<float>(i, j) = temp_sigma;
					}
				}
			}
			/*********************** Sort Gaussian model by 'weight / sigma' End ****************************/


			/*********************** Create new Gaussian component Start ****************************/
			if (num_fit == GMM_MAX_COMPONT && 0 == m_weight[GMM_MAX_COMPONT - 1].at<float>(i, j))
			{
				//if there is no exit component fit,then start a new component
				//������ֵ���ֵ�ʱ����Ŀǰ�ֲ�����С��M������һ���ֲ������²���ֵ��Ϊ��ֵ��������ϴ󷽲�ͽ�СȨֵ
				for (int k = 0; k < GMM_MAX_COMPONT; k++)
				{
					if (0 == m_weight[k].at<float>(i, j))
					{
						m_weight[k].at<float>(i, j) = GMM_LEARN_ALPHA;
						m_mean[k].at<uchar>(i, j) = _image.at<uchar>(i, j);
						m_sigma[k].at<float>(i, j) = 15.0;

						//normalization the weight,let they sum to 1
						for (int q = 0; q < GMM_MAX_COMPONT && q != k; q++)
						{
							//�������ĸ�˹ģ�͵�Ȩֵ���и��£�����Ȩֵ��Ϊ1
							/****update the other unfit's weight,u and sigma remain unchanged****/
							m_weight[q].at<float>(i, j) *= (1 - GMM_LEARN_ALPHA);
						}
						break; //�ҵ���һ��Ȩֵ��Ϊ0�ļ���
					}
				}
			}
			else if (num_fit == GMM_MAX_COMPONT && m_weight[GMM_MAX_COMPONT - 1].at<float>(i, j) != 0)
			{
				//���GMM_MAX_COMPONT��������ֵ�������������ĸ�˹����Ȩֵ�����ĸ�˹��Ȩֵ���䣬ֻ���¾�ֵ�ͷ���
				m_mean[GMM_MAX_COMPONT - 1].at<uchar>(i, j) = _image.at<uchar>(i, j);
				m_sigma[GMM_MAX_COMPONT - 1].at<float>(i, j) = 15.0;
			}
			/*********************** Create new Gaussian component End ****************************/
		}
	}
}

//������ͼ��ÿ������gmmѡ����ʵĸ�˹��������
//��������п����Ǳ����ֲ���������ǰ�棬��С���ܵĶ��ݵķֲ�������ĩ�ˣ����ǽ�������ǰfit_num���ֲ�ѡΪ����ģ��;
//���Ź���ķֲ��У��ۻ����ʳ���GMM_THRESHOD_SUMW��ǰfit_num���ֲ�����������ģ�ͣ�ʣ��������ֲ�������ǰ��ģ�ͣ�
void MOG_BGS::getFitNum(const Mat _image)
{
	for (int i = 0; i < _image.rows; i++)
	{
		for (int j = 0; j < _image.cols; j++)
		{
			float sum_w = 0.0;	//���¸�ֵΪ0������һ���������ۻ�
			for (uchar k = 0; k < GMM_MAX_COMPONT; k++)
			{
				sum_w += m_weight[k].at<float>(i, j);
				if (sum_w >= GMM_THRESHOD_SUMW)	//�������THRESHOD_SUMW=0.6�Ļ�����ô�õ��ĸ�˹��Ŀ��Ϊ1����Ϊÿ�����ض���һ��Ȩֵ�ӽ�1
				{
					m_fit_num.at<uchar>(i, j) = k + 1;
					break;
				}
			}
		}
	}
}

//gmm���Ժ�����ʵ��
void MOG_BGS::testGMM(const Mat _image)
{
	for (int i = 0; i < _image.rows; i++)
	{
		for (int j = 0; j < _image.cols; j++)
		{
			int k = 0;
			for (; k < m_fit_num.at<uchar>(i, j); k++)
			{
				if (abs(_image.at<uchar>(i, j) - m_mean[k].at<uchar>(i, j)) < (uchar)(2.5 * m_sigma[k].at<float>(i, j)))
				{
					m_mask.at<uchar>(i, j) = 0;
					break;
				}
			}
			if (k == m_fit_num.at<uchar>(i, j))
			{
				m_mask.at<uchar>(i, j) = 255;
			}
		}
	}
}


int main(int argc, char* argv[])
{
	Mat frame, gray, mask;
	VideoCapture capture;
	capture.open("E:\\Coding\\C#\\sample.avi");

	if (!capture.isOpened())
	{
		cout << "No camera or video input!\n" << endl;
		return -1;
	}

	MOG_BGS Mog_Bgs;
	int count = 0;

	while (1)
	{
		count++;
		capture >> frame;
		if (frame.empty())
			break;
		cvtColor(frame, gray, CV_RGB2GRAY);

		if (count == 1)
		{
			Mog_Bgs.init(gray);
			Mog_Bgs.processFirstFrame(gray);
			cout << " Using " << TRAIN_FRAMES << " frames to training GMM..." << endl;
		}
		else if (count < TRAIN_FRAMES)
		{
			Mog_Bgs.trainGMM(gray);
		}
		else if (count == TRAIN_FRAMES)
		{
			Mog_Bgs.getFitNum(gray);
			cout << " Training GMM complete!" << endl;
		}
		else
		{
			Mog_Bgs.testGMM(gray);
			mask = Mog_Bgs.getMask();
			morphologyEx(mask, mask, MORPH_OPEN, Mat());
			erode(mask, mask, Mat(7, 7, CV_8UC1), Point(-1, -1));  // You can use Mat(5, 5, CV_8UC1) here for less distortion
			dilate(mask, mask, Mat(7, 7, CV_8UC1), Point(-1, -1));
			imshow("mask", mask);
		}

		imshow("input", frame);

		if (cvWaitKey(10) == 'q')
			break;
	}

	return 0;
}
