#pragma once
#include <opencv2/photo.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>

using namespace cv;
using namespace std;

struct baseDetail {
	vector<Mat>detail;
	Mat base;
};


class DeghostingPeceKautz {

private:
	static float MaxQuart(Mat matrix, float percentile);
	static cv::Mat  ClampImg(Mat img, double a, double b);
	static cv::Mat pyrVal(baseDetail pyramid);
	static baseDetail pyrAdd(baseDetail pA, baseDetail pB);
	static vector<baseDetail>  pyrLst2OP(vector<baseDetail> lstIn1, vector<baseDetail>  lstIn2);
	static baseDetail pyrMul(baseDetail pA, baseDetail pB);
	static vector<baseDetail>  pyrLstS2OP(vector<baseDetail> lstIn, baseDetail pyrImg);
	static cv::Mat pyrGaussGenAux(cv::Mat img);
	static baseDetail pyrGaussGen(cv::Mat img);
	static cv::Mat RemoveSpecials(cv::Mat img);
	static void pyrLapGenAux(Mat img, Mat &tL0, Mat &tB0);
	static baseDetail pyrLapGen(Mat img);
	static vector<baseDetail> pyrImg3(Mat img);
	static cv::Mat WardComputeThreshold(Mat img, float wardPercentile, float wardTolerance = 4.0 / 256.0);
	static cv::Mat PeceKautzMoveMask(int& num, vector<Mat>& imageStack, int iterations = 1, int ke_size = 3, int kd_size = 17, float ward_percentile = 0.5);
	static cv::Mat MertensSaturation(cv::Mat img);
	static cv::Mat MertensContrast(cv::Mat img);
	static cv::Mat Mean(cv::Mat img);
	static cv::Mat MertensWellExposedness(cv::Mat img, float we_mean = 0.5, float we_sigma = 0.2);
public:
	static Mat createPeceKautz(vector<Mat> &imageStack, int iterations = 4, int ke_size = 3, int kd_size = 17, float ward_percentile = 0.5);

};