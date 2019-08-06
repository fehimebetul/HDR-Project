#include "deghostingPeceKautz.h"

float DeghostingPeceKautz::MaxQuart(Mat matrix, float percentile) {

	if (percentile > 1.0)
		percentile = 1.0;
	if (percentile < 0.0)
		percentile = 0.0;

	int n = matrix.rows;
	int m = matrix.cols;

	matrix = matrix.reshape(1, n*m);
	cv::sort(matrix, matrix, SORT_EVERY_COLUMN + SORT_ASCENDING);
	int index = round(n * m * percentile);
	index = max(index, 1);
	float ret = matrix.ptr<float>(index, 0)[0];
	return ret;
}

cv::Mat  DeghostingPeceKautz:: ClampImg(Mat img, double a, double b)
{
	cv::Mat out = img.clone();

	out.setTo(a, img<a);
	out.setTo(b, img>b);
	return out;

}

cv::Mat DeghostingPeceKautz::pyrVal(baseDetail pyramid) {

	vector<Mat> list = pyramid.detail;
	Mat base = pyramid.base.clone();
	int n = list.size();

	cv::Mat img;

	for (int i = 0; i < n; i++) {
		int ind = n - i - 1;
		int r = list[ind].rows;
		int c = list[ind].cols;

		if (i == 0)
		{
			resize(base, base, Size(c, r), 0.5, 0.5, INTER_LINEAR);
			img = base + list[ind];
		}
		else
		{
			resize(img, img, Size(c, r), 0.5, 0.5, INTER_LINEAR);
			img = img + list[ind];
		}
	}

	return img;
}

baseDetail DeghostingPeceKautz::pyrAdd(baseDetail pA, baseDetail pB) {

	baseDetail pOut;
	//checking lenght of the pyramids
	int nA = pA.detail.size();
	int nB = pB.detail.size();

	if (nA != nB)
		cout << "Error" << endl;

	//adding base layers
	pOut.base = pA.base + pB.base;
	pOut.detail = vector<Mat>(nA);;

	for (int i = 0; i < nA; i++) {
		pOut.detail[i] = pA.detail[i] + pB.detail[i];
	}
	return pOut;

}

vector<baseDetail> DeghostingPeceKautz::pyrLst2OP(vector<baseDetail> lstIn1, vector<baseDetail>  lstIn2)
{
	int n = lstIn1.size();
	vector<baseDetail> lstOut;

	for (int i = 0; i < n; i++) {
		baseDetail p = pyrAdd(lstIn1[i], lstIn2[i]);
		lstOut.push_back(p);
	}
	return lstOut;
}

baseDetail DeghostingPeceKautz::pyrMul(baseDetail pA, baseDetail pB)
{
	baseDetail pOut;
	//checking lenght of the pyramids
	int nA = pA.detail.size();
	int nB = pB.detail.size();

	if (nA != nB)
		cout << "Error" << endl;

	//Multiplying base levels
	multiply(pA.base, pB.base, pOut.base);
	pOut.detail = vector<Mat>(nA);

	//Multiplying details of each level
	for (int i = 0; i < nA; i++) {
		multiply(pA.detail[i], pB.detail[i], pOut.detail[i]);
	}
	return pOut;
}

vector<baseDetail>  DeghostingPeceKautz::pyrLstS2OP(vector<baseDetail> lstIn, baseDetail pyrImg)
{
	int n = lstIn.size();
	vector<baseDetail> lstOut;

	for (int i = 0; i < n; i++) {
		baseDetail p = pyrMul(lstIn[i], pyrImg);
		lstOut.push_back(p);
	}
	return lstOut;
}

cv::Mat DeghostingPeceKautz::pyrGaussGenAux(cv::Mat img) {
	Mat kernel = (Mat_<float>(1, 5) << 1, 4, 6, 4, 1);
	cv::Mat mtx = kernel.t() * kernel;
	mtx = mtx / sum(mtx)[0];

	//Convolution
	Mat imgB = img.clone();
	filter2D(img, imgB, -1, mtx, cv::Point(-1, -1), (0.0), BORDER_REPLICATE);

	//Downsampling
	cv::Mat imgOut;
	resize(imgB, imgOut, Size(ceil(img.cols / 2.0), ceil(img.rows / 2.0)), 0.5, 0.5, INTER_LINEAR);
	return imgOut;
}

baseDetail DeghostingPeceKautz::pyrGaussGen(cv::Mat img) {

	int r = img.rows;
	int c = img.cols;
	int levels = floor(log2(min(r, c)));
	vector<Mat> list;

	for (int i = 0; i < levels; i++) {
		//Detail Layer
		list.push_back(img.clone());
		//Next Layer

		img = pyrGaussGenAux(img.clone()).clone();
	}
	//base layer
	baseDetail p;
	p.detail = list;
	p.base = img.clone();
	return p;
}

cv::Mat DeghostingPeceKautz::RemoveSpecials(cv::Mat img) {
	int  clamping_value = 0;

	Point* point = NULL;
	patchNaNs(img, clamping_value);

	return img;
}

void DeghostingPeceKautz::pyrLapGenAux(Mat img, Mat &tL0, Mat &tB0)
{
	Mat kernel = (Mat_<float>(1, 5) << 1, 4, 6, 4, 1);
	cv::Mat mtx = kernel.t() * kernel;
	mtx = mtx / sum(mtx)[0];

	//Convolution
	Mat imgB = img.clone();
	filter2D(img, imgB, -1, mtx, cv::Point(-1, -1), (0.0), BORDER_REPLICATE);

	//Downsampling
	resize(imgB, tL0, Size(ceil(img.cols / 2.0), ceil(img.rows / 2.0)), 0.5, 0.5, INTER_LINEAR);
	//Upsampling
	Mat imgE; resize(tL0, imgE, Size(img.cols, img.rows), 0.5, 0.5, INTER_LINEAR);

	//Diff
	tB0 = img - imgE;
}

baseDetail DeghostingPeceKautz::pyrLapGen(Mat img) {
	int r = img.rows;
	int c = img.cols;
	int levels = floor(log2(min(r, c)));

	vector<Mat> list;
	cv::Mat tL0, tB0;
	for (int i = 0; i < levels; i++) {
		//calculating detail and base layers

		pyrLapGenAux(img, tL0, tB0);
		img = tL0.clone();

		//Detail layer
		list.push_back(tB0.clone());
	}
	//base layer
	baseDetail p;
	p.detail = list;
	p.base = tL0.clone();
	return p;
}

vector<baseDetail> DeghostingPeceKautz::pyrImg3(Mat img) {

	vector<baseDetail> lstOut;
	int col = img.channels();

	vector<Mat>bgr(col);
	split(img, bgr);


	for (int i = 0; i < col; i++) {
		lstOut.push_back(pyrLapGen(bgr[i].clone()));
	}

	return lstOut;
}

cv::Mat DeghostingPeceKautz::WardComputeThreshold(Mat img, float wardPercentile, float wardTolerance) {

	cv::Mat grey;
	if (img.channels() == 1) {
		grey = img.clone();
	}
	else
	{
		vector<Mat>bgr(3);
		split(img, bgr);

		grey = (54 * bgr[0]) + (183 * bgr[1]) + (19 * bgr[2]);
		grey = grey / 256;
	}

	float medVal = MaxQuart(grey.clone(), wardPercentile);
	cv::Mat imgThr = Mat::zeros(Size(grey.cols, grey.rows), CV_32F);
	//cv::threshold(grey, imgThr, medVal, 1, THRESH_BINARY);
	imgThr.setTo(1.0, grey>medVal);

	float A = medVal - wardTolerance;
	float B = medVal + wardTolerance;


	cv::Mat imgEb = Mat::ones(Size(grey.cols, grey.rows), CV_32F);
	imgEb.setTo(0.0, (grey >= A & grey <= B));

	return imgThr;
}

cv::Mat DeghostingPeceKautz::PeceKautzMoveMask(int& num, vector<Mat>& imageStack, int iterations , int ke_size, int kd_size, float ward_percentile ) {

/*
        For each image in exposure stack, apply the MTB algorithm, yielding a stack of bitmap
		Input:
	      -imageStack : an exposure stack of LDR images
           -iterations : number of iterations for improving the movements'
           -ke_size: size of the erosion kernel
           -kd_size : size of the dilation kernel
           -ward_percentile :perceptage of the pixel change
			-num : number of different connected components in moveMask
        Output :
           -moveMaskOut : movements' mask
           
*/

	int n = imageStack.size();

	cv::Mat moveMask = WardComputeThreshold(imageStack[0], ward_percentile);

	/*
	If the scene is static, expected that pixel preserve its bit value across all bitmaps
	If the value in a changes, know that there is movement. So inder to detect movement pixels
	They sum up all bitmaps. 
	In the paper, this is M*
	*/
	cv::Mat mask;
	for (int i = 1; i < n; i++) {
		mask = WardComputeThreshold(imageStack[i], ward_percentile);
		moveMask = moveMask + mask;
	}

	/*
	Any pixel in M* that neither 0 or N (assumes N exposure) is classified as a movement
	*/
	moveMask.setTo(0, moveMask == n);
	moveMask.setTo(1, moveMask > 0);

	/*
	M* may contain a certain amount of noise that could lead to incorrent movement detection
	Refine M* using a sequence of morphological dilation and erosion in order to generate the final motion map M
	*/
	cv::Mat kernel_d = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kd_size, kd_size));
	cv::Mat kernel_e = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(ke_size, ke_size));

	for (int i = 1; i < n; i++) {
		cv::dilate(moveMask, moveMask, kernel_d, cv::Point(-1, -1), 1);
		cv::erode(moveMask, moveMask, kernel_e, cv::Point(-1, -1), 1);
	}

	/*
	Final motion map in here we named as moveMask
	*/
	/*
	After the motionis detected done M is converted into a cluster Map where each identified cluster
	has a different label which computed using Connected Component labelling. This yield the labeled motion map
	L_m with labelled cluster area \phi_i that contains the moving pixels which cause ghosting artifacts
	*/
	cv::Mat moveMask_tmp, moveMaskBinary;
	moveMask.convertTo(moveMaskBinary, CV_8U);
	/*
	Connected component method work with only 8 bit images so I convert the image for 8 bit from 32F 
	For only this method after this method convert back to the floating point
	*/
	num = connectedComponents(moveMaskBinary, moveMask_tmp, 4, CV_32S);

	moveMask_tmp.convertTo(moveMask_tmp, CV_32F);

	cv::Mat moveMaskOut = moveMask_tmp.clone();
	cv::multiply(moveMask_tmp, moveMask, moveMaskOut);
	moveMaskOut.setTo(-1, moveMaskOut == 0);
	return moveMaskOut;
}

cv::Mat DeghostingPeceKautz::MertensSaturation(cv::Mat img)
{
	int r = img.rows;
	int c = img.cols;
	int col = img.channels();
	cv::Mat Ws;
	if (col == 1)
	{
		Ws = Mat::ones(cv::Size(c, r), CV_32F);
	}
	else
	{
		cv::Mat mu = Mat::zeros(cv::Size(c, r), CV_32F);
		cv::Mat SumC = Mat::zeros(cv::Size(c, r), CV_32F);

		vector<Mat>bgr(3);
		split(img, bgr);
		for (int i = 0; i < col; i++) {
			mu += bgr[i];
		}
		cv::divide(mu, col, mu);

		for (int i = 0; i < col; i++) {
			cv::Mat temp = bgr[i].clone();
			cv::subtract(temp, mu, temp);
			multiply(temp, temp, temp);
			SumC = SumC + temp;
		}
		cv::divide(SumC, col, SumC);
		sqrt(SumC, Ws);
	}
	return Ws;
}

cv::Mat DeghostingPeceKautz::MertensContrast(cv::Mat img)
{
	Mat h = (Mat_<float>(3, 3) << 0, 1, 0, 1, -4, 1, 0, 1, 0);
	Mat dst = img.clone();

	filter2D(img, dst, -1, h, cv::Point(-1, -1), (0.0), BORDER_REPLICATE);

	return abs(dst);
}

cv::Mat DeghostingPeceKautz::Mean(cv::Mat img) {


	vector<Mat>bgr(3);
	split(img, bgr);
	cv::Mat mean = (bgr[0] + bgr[1] + bgr[2]) / 3;
	return mean;
}

cv::Mat DeghostingPeceKautz::MertensWellExposedness(cv::Mat img, float we_mean , float we_sigma ) {
	float sigma2 = 2.0 * pow(we_sigma, 2);

	int r = img.rows;
	int c = img.cols;
	int col = img.channels();

	cv::Mat output, We = Mat::ones(cv::Size(c, r), CV_32F);

	cv::Mat imgTemp = img.clone();
	imgTemp = img - Vec3f(we_mean, we_mean, we_mean);
	multiply(imgTemp, imgTemp, imgTemp);
	imgTemp = -1 * imgTemp / sigma2;
	exp(imgTemp, imgTemp);

	vector<Mat>bgr(3);
	split(imgTemp, bgr);
	for (int i = 0; i < col; i++) {
		cv::multiply(We, bgr[i], We);
	}
	return We;
}

Mat DeghostingPeceKautz::createPeceKautz(vector<Mat> &imageStack, int iterations , int ke_size, int kd_size, float ward_percentile) {
	/*
	Input:
	-imageStack: an exposure stack of LDR images
	-iterations: number of iterations for improving the movements'
	mask
	-ke_size: size of the erosion kernel
	-kd_size: size of the dilation kernel
	-ward_percentile:

	Output:
	-imgOut: tone mapped image
	*/
	int weights[] = { 1, 1, 1 };

	int wE = weights[0];
	int wS = weights[1];
	int wC = weights[2];

	int r = imageStack[0].rows;
	int c = imageStack[0].cols;
	int col = imageStack[0].channels();
	int n = imageStack.size();

	cv::Mat total = Mat::zeros(cv::Size(c, r), CV_32F);
	vector<Mat> weight = vector<Mat>(n);

	for (int i = 0; i < n; i++)
		weight[i] = Mat::ones(cv::Size(c, r), CV_32F);
	/*
	Exposure fusion does not require the camera's response curve and instead relies on three simple per-pixel quality measure
	Contrast
	Saturation
	Well-exposedness
	A weighted average of these three measures is computed for each pixeli yielding a per pixel weight map W for each exposure in the sequence.
	*/
	for (int i = 0; i < n; i++) {

		if (wE > 0.0)
		{
			cv::Mat weightE = MertensWellExposedness(imageStack[i]);
			Mat temp;
			cv::log(weightE, temp);
			cv::multiply(wE, temp, temp);
			cv::exp(temp, temp);
			cv::multiply(weight[i], temp, weight[i]);
		}
		if (wC > 0.0)
		{
			cv::Mat L = Mean(imageStack[i]);
			cv::Mat weightC = MertensContrast(L);

			Mat temp;
			cv::log(weightC, temp);
			cv::multiply(wC, temp, temp);
			cv::exp(temp, temp);
			cv::multiply(weight[i], temp, weight[i]);
		}
		if (wS > 0.0)
		{
			cv::Mat weightS = MertensSaturation(imageStack[i]);
			Mat temp;
			cv::log(weightS, temp);
			cv::multiply(wS, temp, temp);
			cv::exp(temp, temp);
			cv::multiply(weight[i], temp, weight[i]);
		}
		weight[i] = weight[i] + 1e-12;
	}

	vector<Mat>weight_move = weight;
	int num = 0;
	/*Motion Detection moveMAsk = L_m in the paper*/
	cv::Mat moveMask = PeceKautzMoveMask(num, imageStack, iterations, ke_size, kd_size, ward_percentile);

	vector<cv::Point> index;
	for (int i = 0; i < num; i++)
	{
		
		cv::Mat indexMat = Mat::zeros(cv::Size(moveMask.cols, moveMask.rows), CV_8U);
		indexMat.setTo(1, moveMask == i);
		findNonZero(indexMat, index);

		vector<double>Wvec(n);
		for (int l = 0; l < n; l++) {
			cv::Mat W = weight[l];
			double sum = 0;
			for (int m = 0; m < index.size(); m++) {
				sum += W.ptr<float>(index[m].y, index[m].x)[0];
			}
			Wvec[l] = sum;
		}

		int j = distance(Wvec.begin(), max_element(Wvec.begin(), Wvec.end()));
		cv::Mat W = Mat::zeros(cv::Size(c, r), CV_32F);

		for (int m = 0; m < index.size(); m++) {
			W.ptr<float>(index[m].y, index[m].x)[0] = 1;
		}
		cv::Mat W_inv = 1.0 - W;

		for (int m = 0; m < n; m++)
		{
			if (j != m)
				multiply(weight_move[m], W_inv, weight_move[m]);
		}
	}

	/*
	Normalization of weights to sum to sum at each pixel
	*/
	for (int i = 0; i < n; i++) {
		total = total + weight_move[i];
	}

	//empty pyramid

	vector<baseDetail> tf;
	for (int i = 0; i < n; i++) {
		// Laplacian pyramid : image
		vector<baseDetail>  pyrImg = pyrImg3(imageStack[i]);

		//Gaussian pyramid: weight_i
		cv::Mat results;
		divide(weight_move[i], total, results);
		cv::Mat weight_i = RemoveSpecials(results);
		baseDetail  pyrW = pyrGaussGen(weight_i.clone());

		//Multiplication image times weights
		vector<baseDetail> tmpVal = pyrLstS2OP(pyrImg, pyrW);

		if (i == 0) {
			tf = tmpVal;
		}
		else {
			//accumulation
			tf = pyrLst2OP(tf, tmpVal);
		}
	}

	//Evaluation of Laplacian/Gausian Pyramids
	cv::Mat imgOut = Mat::zeros(cv::Size(c, r), CV_32FC3);
	vector<Mat>bgr(3);
	split(imgOut, bgr);
	for (int i = 0; i < col; i++)
	{
		bgr[i] = pyrVal(tf[i]);
	}
	cv::merge(bgr, imgOut);
	double minVal;
	double maxVal;
	Point minLoc;
	Point maxLoc;
	cv::Mat imageOut_temp = imgOut.clone();
	minMaxLoc(imageOut_temp.reshape(1, imgOut.rows*imgOut.cols*imgOut.channels()), &minVal, &maxVal, &minLoc, &maxLoc);
	imgOut = ClampImg((imgOut / maxVal), 0.0, 1.0);


	return imgOut;
}