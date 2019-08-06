#include "deghostingPeceKautz.h"
#include <time.h>


void readImagesAndTimes(vector<Mat> &images)
{

	int numImages = 7;
	
	/*With different input images*/
	//static const char* filenames[] = { "./images/stack_ghost/IMG_0959.JPG", "./images/stack_ghost/IMG_0960.JPG", "./images/stack_ghost/IMG_0961.JPG" };
	static const char* filenames[] = { "./images/SantasLittleHelper/Img1.tif",
		"./images/SantasLittleHelper/Img2.tif", 
		"./images/SantasLittleHelper/Img3.tif", 
		"./images/SantasLittleHelper/Img4.tif", 
		"./images/SantasLittleHelper/Img5.tif", 
		"./images/SantasLittleHelper/Img6.tif", 
		"./images/SantasLittleHelper/Img7.tif",
	};
	//static const char* filenames[] = { "./images/16/1.tiff",
	//	"./images/16/2.tiff", 
	//	"./images/16/3.tiff", 
	//	"./images/16/4.tiff", 
	//	"./images/16/5.tiff", 
	//	"./images/16/6.tiff", 
	//	"./images/16/7.tiff",
	//	"./images/16/8.tiff",
	//	"./images/16/9.tiff",
	//};
	for (int i = 0; i < numImages; i++)
	{
		Mat imI = imread(filenames[i]);
		Mat im; imI.convertTo(im, CV_32FC3);
		im = im / 255.0;
		images.push_back(im);
	}

}


int main(int, char**argv)
{
	// Read images and exposure times
	cout << "Reading images ... " << endl;
	vector<Mat> images;
	readImagesAndTimes(images);

	/*
	Time for performance
	*/
	clock_t start, end;
	double elapsed;
	start = clock();

	cv::Mat img_merged = DeghostingPeceKautz::createPeceKautz(images);

	end = clock();
	elapsed = ((double)(end - start)) / CLOCKS_PER_SEC;
	std::cout << elapsed<<endl;

	// Save HDR image.
	imwrite("./outputs/hdrDebevec.hdr", img_merged.clone());
	cout << "saved hdrDebevec.hdr " << endl;

	// Tonemap using Drago's method to obtain 24-bit color image
	cout << "Tonemaping using Drago's method ... ";
	Mat ldrDrago;
	Ptr<TonemapDrago> tonemapDrago = createTonemapDrago(1.0, 0.7);
	tonemapDrago->process(img_merged.clone(), ldrDrago);
	ldrDrago = 3 * ldrDrago;
	imwrite("./outputs/ldr-Drago.jpg", ldrDrago * 255);
	cout << "saved ldr-Drago.jpg" << endl;

	// Tonemap using Durand's method obtain 24-bit color image
	cout << "Tonemaping using Durand's method ... ";
	Mat ldrDurand;
	Ptr<TonemapDurand> tonemapDurand = createTonemapDurand(1.5, 4, 1.0, 1, 1);
	tonemapDurand->process(img_merged.clone(), ldrDurand);
	ldrDurand = 3 * ldrDurand;
	imwrite("./outputs/ldr-Durand.jpg", ldrDurand * 255);
	cout << "saved ldr-Durand.jpg" << endl;

	// Tonemap using Reinhard's method to obtain 24-bit color image
	cout << "Tonemaping using Reinhard's method ... ";
	Mat ldrReinhard;
	Ptr<TonemapReinhard> tonemapReinhard = createTonemapReinhard(1.5, 0, 0, 0);
	tonemapReinhard->process(img_merged.clone(), ldrReinhard);
	imwrite("./outputs/ldr-Reinhard.jpg", ldrReinhard * 255);
	cout << "saved ldr-Reinhard.jpg" << endl;

	// Tonemap using Mantiuk's method to obtain 24-bit color image
	cout << "Tonemaping using Mantiuk's method ... ";
	Mat ldrMantiuk;
	Ptr<TonemapMantiuk> tonemapMantiuk = createTonemapMantiuk(2.2, 0.85, 1.2);
	tonemapMantiuk->process(img_merged.clone(), ldrMantiuk);
	ldrMantiuk = 3 * ldrMantiuk;
	imwrite("./outputs/ldr-Mantiuk.jpg", ldrMantiuk * 255);
	cout << "saved ldr-Mantiuk.jpg" << endl;


	// Tonemap using Mantiuk's method to obtain 24-bit color image
	cout << "Tonemaping using gamma ... ";
	Mat ldr;
	Ptr<Tonemap> tonemap = createTonemap(2.2);
	tonemap->process(img_merged.clone(), ldr);
	ldr = 3 * ldr;
	imwrite("./outputs/ldr.jpg", ldr * 255);
	cout << "saved ldr.jpg" << endl;

	cout << "Finish to click";
	cin.ignore();
}