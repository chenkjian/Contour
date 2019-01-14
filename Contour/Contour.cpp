#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

/*

Point2d rotationPoint(Point2d srcPoint, const double cosAngle, const double sinAngle)
{
	Point2d dstPoint;
	dstPoint.x = srcPoint.x * cosAngle + srcPoint.y * sinAngle;
	dstPoint.y = -srcPoint.x * sinAngle + srcPoint.y * cosAngle;
	return dstPoint;
}



void imageRotation(Mat& srcImage, Mat& dstImage, Mat_<double>& shape, double& angle)
{
	const double cosAngle = cos(angle);
	const double sinAngle = sin(angle);
	// 计算标注中心
	double center_x = 0;
	double center_y = 0;
	for (int i = 0; i < shape.rows; i++) {
		center_x += shape(i, 0);
		center_y += shape(i, 1);
	}
	center_x /= shape.rows;
	center_y /= shape.rows;

	//原图像四个角的坐标变为以旋转中心的坐标系
	Point2d leftTop(-center_x, center_y); //(0,0)
	Point2d rightTop(srcImage.cols - center_x, center_y); // (width,0)
	Point2d leftBottom(-center_x, -srcImage.rows + center_y); //(0,height)
	Point2d rightBottom(srcImage.cols - center_x, -srcImage.rows + center_y); //(width,height)

	//以center为中心旋转后四个角的坐标
	Point2d transLeftTop, transRightTop, transLeftBottom, transRightBottom;
	transLeftTop = rotationPoint(leftTop, cosAngle, sinAngle);
	transRightTop = rotationPoint(rightTop, cosAngle, sinAngle);
	transLeftBottom = rotationPoint(leftBottom, cosAngle, sinAngle);
	transRightBottom = rotationPoint(rightBottom, cosAngle, sinAngle);

	//计算旋转后图像的width，height
	double left = min({ transLeftTop.x, transRightTop.x, transLeftBottom.x, transRightBottom.x });
	double right = max({ transLeftTop.x, transRightTop.x, transLeftBottom.x, transRightBottom.x });
	double top = min({ transLeftTop.y, transRightTop.y, transLeftBottom.y, transRightBottom.y });
	double down = max({ transLeftTop.y, transRightTop.y, transLeftBottom.y, transRightBottom.y });

	int width = static_cast<int>(abs(left - right) + 0.5);
	int height = static_cast<int>(abs(top - down) + 0.5);

	// 分配内存空间
	dstImage.create(height, width, srcImage.type());

	const double dx = -abs(left) * cosAngle - abs(down) * sinAngle + center_x;
	const double dy = abs(left) * sinAngle - abs(down) * cosAngle + center_y;

	int x, y;
	for (int i = 0; i < height; i++) // y
	{
		for (int j = 0; j < width; j++) // x
		{
			//坐标变换
			x = float(j)*cosAngle + float(i)*sinAngle + dx;
			y = float(-j)*sinAngle + float(i)*cosAngle + dy;


			if ((x<0) || (x >= srcImage.cols) || (y<0) || (y >= srcImage.rows))
			{
				if (srcImage.channels() == 3)
				{
					dstImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
				}
				else if (srcImage.channels() == 1)
				{
					dstImage.at<uchar>(i, j) = 0;
				}
			}
			else
			{
				if (srcImage.channels() == 3)
				{
					dstImage.at<cv::Vec3b>(i, j) = srcImage.at<cv::Vec3b>(y, x);
				}
				else if (srcImage.channels() == 1)
				{
					dstImage.at<uchar>(i, j) = srcImage.at<uchar>(y, x);
				}
			}
		}
	}
}
*/
/*
void RotateImg2(cv::Mat& src, cv::Mat& dst, cv::Point2f center, double angle)
{
	CV_Assert(src.type() == CV_8UC1);

	angle *= CV_PI / 180.0; // 角度转换成弧度

	// 将坐标系移到旋转中心去
	float cx = center.x;
	float cy = center.y;
	float offset[3][3] =
	{
		{ 1,    0,   0 },
		{ 0,    -1,   0 },
		{ -cx,  cy,  1 }
	};
	cv::Mat offsetMat(3, 3, CV_32FC1, offset);

	// 构建旋转矩阵
	float rot[3][3] =
	{
		{ std::cos(angle),   -1 * std::sin(angle),     0 },
		{ std::sin(angle),   std::cos(angle),         0 },
		{ 0,                0,                       1 }
	};
	cv::Mat rotMat(3, 3, CV_32FC1, rot);

	// 偏移和旋转矩阵
	cv::Mat tmp = offsetMat * rotMat;

	cv::Mat topLeft = cv::Mat::ones(1, 3, CV_32FC1);
	cv::Mat topRight = cv::Mat::ones(1, 3, CV_32FC1);
	cv::Mat btmLeft = cv::Mat::ones(1, 3, CV_32FC1);
	cv::Mat btmRight = cv::Mat::ones(1, 3, CV_32FC1);
	topLeft.at<float>(0, 0) = 0.0f;
	topLeft.at<float>(0, 1) = 0.0f;
	topRight.at<float>(0, 0) = src.cols - 1;
	topRight.at<float>(0, 1) = 0.0f;
	btmLeft.at<float>(0, 0) = 0.0f;
	btmLeft.at<float>(0, 1) = src.rows - 1;
	btmRight.at<float>(0, 0) = src.cols - 1;
	btmRight.at<float>(0, 1) = src.rows - 1;

	cv::Mat topLeftDst = topLeft * tmp;
	cv::Mat topRightDst = topRight * tmp;
	cv::Mat btmLeftDst = btmLeft * tmp;
	cv::Mat btmRightDst = btmRight * tmp;

	// 计算旋转后的点图像最左边点横坐标lxMin， 最右边点横坐标lxMax
	// 最高点纵坐标lyMin. 最低点横坐标lyMax
	float lxMin = std::min(topLeftDst.at<float>(0, 0),
		std::min(topRightDst.at<float>(0, 0),
			std::min(btmLeftDst.at<float>(0, 0), btmRightDst.at<float>(0, 0))));
	float lxMax = std::max(topLeftDst.at<float>(0, 0),
		std::max(topRightDst.at<float>(0, 0),
			std::max(btmLeftDst.at<float>(0, 0), btmRightDst.at<float>(0, 0))));
	float lyMin = std::min(topLeftDst.at<float>(0, 1),
		std::min(topRightDst.at<float>(0, 1),
			std::min(btmLeftDst.at<float>(0, 1), btmRightDst.at<float>(0, 1))));
	float lyMax = std::max(topLeftDst.at<float>(0, 1),
		std::max(topRightDst.at<float>(0, 1),
			std::max(btmLeftDst.at<float>(0, 1), btmRightDst.at<float>(0, 1))));

	float offsetInv[3][3] =
	{
		{ 1.0,       0.0,        0.0 },
		{ 0.0,      -1.0,        0.0 },
		{ -lxMin,    lyMax,      1.0 }
	};

	cv::Mat offsetInvMat(3, 3, CV_32FC1, offsetInv);

	// 最终处理的矩阵
	cv::Mat rotateMat = tmp * offsetInvMat;

	int WIDTH = std::abs(int(lxMax - lxMin)) ;
	int HEIGHT = std::abs(int(lyMax - lyMin));
	dst = cv::Mat::zeros(HEIGHT, WIDTH, src.type());

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			float input[3] = { x, y, 1.0f };
			cv::Mat inputMat(1, 3, CV_32FC1, input);

			cv::Mat outputMat = inputMat * rotateMat;
			cv::Point dstPt = cv::Point(outputMat.at<float>(0, 0), outputMat.at<float>(0, 1));

			cv::Rect dstImgRect = cv::Rect(0, 0, WIDTH, HEIGHT);
			if (dstImgRect.contains(dstPt))
			{
				// 彩色图请在这里进行相应修改
				dst.at<uchar>(dstPt.y, dstPt.x) = src.at<uchar>(y, x);
			}
		}
	}
}

*/

int main()
{
	Mat srcImage, dstImage, src;
	src = imread("1.jpg");
    cvtColor(src, srcImage, CV_RGB2GRAY);

	namedWindow("原图像", CV_WINDOW_NORMAL);
	imshow("原图像", srcImage);

	//dstImage.create(srcImage.size(), srcImage.type());

	double degree = 30;
	double a = sin(degree * CV_PI / 180);
	double b = cos(degree * CV_PI / 180);

	int width = srcImage.cols;
	int height = srcImage.rows;

	int rotate_width = int(height * fabs(a) + width * fabs(b));
	int rotate_height = int(width * fabs(a) + height * fabs(b));

	dstImage.create(Size(rotate_width, rotate_height), srcImage.type());

	Point center = Point(srcImage.cols / 2, srcImage.rows / 2);

	Mat map_matrix = getRotationMatrix2D(center, degree, 1.0);
	map_matrix.at<double>(0, 2) += (rotate_width - width) / 2;     // 修改坐标偏移
	map_matrix.at<double>(1, 2) += (rotate_height - height) / 2;   // 修改坐标偏移
	//cout << map_matrix << endl << endl;
	//cout << srcImage.channels() << endl << endl;
	
	Mat rotateMat = Mat::eye(3, 3, CV_64FC1);

	rotateMat.at<double>(0, 0) = map_matrix.at<double>(0, 0);
	rotateMat.at<double>(0, 1) = map_matrix.at<double>(1, 0);
	rotateMat.at<double>(1, 0) = map_matrix.at<double>(0, 1);
	rotateMat.at<double>(1, 1) = map_matrix.at<double>(1, 1);
	rotateMat.at<double>(2, 0) = map_matrix.at<double>(0, 2);
	rotateMat.at<double>(2, 1) = map_matrix.at<double>(1, 2);

	for (int y = 0; y < srcImage.rows; y++)
	{
		for (int x = 0; x < srcImage.cols; x++)
		{
			cv::Mat inputMat = (Mat_<double>(1, 3) << x, y, 1);
			cv::Mat outputMat = inputMat * rotateMat;

			cv::Point dstPt = cv::Point(outputMat.at<double>(0, 0), outputMat.at<double>(0, 1));

			cv::Rect dstImgRect = cv::Rect(0, 0, rotate_width, rotate_height);
			if (dstImgRect.contains(dstPt))
			{
				//输入图像点的像素转到旋转后的图像上
				dstImage.at<uchar>(dstPt.y, dstPt.x) = srcImage.at<uchar>(y, x);
			}
		}
	}

	//warpAffine(srcImage, dstImage, map_matrix, { rotate_width, rotate_height }, CV_INTER_CUBIC);
	//RotateImg2(srcImage, dstImage, center, degree);
	//imageRotation(srcImage, dstImage, Mat_<double>(3, 3), degree);

	namedWindow("旋转后的图像", CV_WINDOW_NORMAL);
	imshow("旋转后的图像", dstImage);

	waitKey(0);
	return 0;
}

/*
int main()
{
	cv::Mat srcImage = cv::imread("2.jpg");
	cv::Mat dstImage;
	dstImage.create(srcImage.size(), srcImage.type());

	//旋转角度
	double angle = 70;
	double a = sin(angle * CV_PI / 180);
	double b = cos(angle * CV_PI / 180);

	cv::Size src_sz = srcImage.size();
	cv::Size dst_sz(src_sz.height, src_sz.width);

	int width = srcImage.cols;
	int height = srcImage.rows;
	int rotate_width = int(height * fabs(a) + width * fabs(b));
	int rotate_height = int(width * fabs(a) + height * fabs(b));

	//指定旋转中心
	Point center = Point(srcImage.cols / 2, srcImage.rows / 2);

	//获取旋转矩阵（2x3矩阵）
	cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);

	//根据旋转矩阵进行仿射变换
	cv::warpAffine(srcImage, dstImage, rot_mat, { rotate_width, rotate_height }, CV_INTER_CUBIC);

	//显示旋转效果
	cv::imshow("image", srcImage);
	cv::imshow("result", dstImage);

	cv::waitKey(0);

	return 0;

}
*/
/*
int main()
{
	cv::Mat matSrc = cv::imread("1.jpg", 2 | 4);

	//if (matSrc.empty()) return 1;

	const double degree = 45;
	double angle = degree * CV_PI / 180.;
	double alpha = cos(angle);
	double beta = sin(angle);
	int iWidth = matSrc.cols;
	int iHeight = matSrc.rows;
	int iNewWidth = cvRound(iWidth * fabs(alpha) + iHeight * fabs(beta));
	int iNewHeight = cvRound(iHeight * fabs(alpha) + iWidth * fabs(beta));

	//构造矩阵
	double m[6];
	m[0] = alpha;
	m[1] = beta;
	m[2] = (1 - alpha) * iWidth / 2. - beta * iHeight / 2.;
	m[3] = -m[1];
	m[4] = m[0];
	m[5] = beta * iWidth / 2. + (1 - alpha) * iHeight / 2.;

	cv::Mat M = cv::Mat(2, 3, CV_64F, m);
	//cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(iWidth / 2., iHeight / 2.), degree, 1);

	cv::Mat matDst = cv::Mat(cv::Size(iNewWidth, iNewHeight), matSrc.type(), cv::Scalar::all(0));

	double D = m[0] * m[4] - m[1] * m[3];
	D = D != 0 ? 1. / D : 0;
	double A11 = m[4] * D, A22 = m[0] * D;
	m[0] = A11; m[1] *= -D;
	m[3] *= -D; m[4] = A22;
	double b1 = -m[0] * m[2] - m[1] * m[5];
	double b2 = -m[3] * m[2] - m[4] * m[5];
	m[2] = b1; m[5] = b2;

	for (int y = 0; y < iNewHeight; ++y)
	{
		for (int x = 0; x < iNewWidth; ++x)
		{

			float fx = m[0] * x + m[1] * y + m[2];
			float fy = m[3] * x + m[4] * y + m[5];

			int sy = cvFloor(fy);
			fy -= sy;

			if (sy < 0 || sy >= iHeight) continue;

			short cbufy[2];
			cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);//防止出现浮点数
			cbufy[1] = 2048 - cbufy[0];


			int sx = cvFloor(fx);
			fx -= sx;

			if (sx < 0 || sx >= iWidth) continue;

			short cbufx[2];
			cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048); //防止溢出
			cbufx[1] = 2048 - cbufx[0];


			for (int k = 0; k < matSrc.channels(); ++k)
			{
				if (sy == iHeight - 1 || sx == iWidth - 1) {
					continue;
				}
				else {
					    matDst.at<cv::Vec3b>(y, x)[k] = (matSrc.at<cv::Vec3b>(sy, sx)[k] * cbufx[0] * cbufy[0] +
						matSrc.at<cv::Vec3b>(sy + 1, sx)[k] * cbufx[0] * cbufy[1] +
						matSrc.at<cv::Vec3b>(sy, sx + 1)[k] * cbufx[1] * cbufy[0] +
						matSrc.at<cv::Vec3b>(sy + 1, sx + 1)[k] * cbufx[1] * cbufy[1]) >> 22;
				}
			}
		}
	}
	cv::imshow("rotate_bilinear_1.jpg", matDst);

	//直接调用函数
	cv::Mat M2 = cv::getRotationMatrix2D(cv::Point2f(iWidth / 2., iHeight / 2.), degree, 1);
	cv::Mat matDst2;
	cv::warpAffine(matSrc, matDst2, M2, cv::Size(iNewWidth, iNewHeight), 1, 0, 0);
	cv::imshow("rotate_bilinear_2.jpg", matDst2);


	waitKey(0);
	return 0;
}
*/