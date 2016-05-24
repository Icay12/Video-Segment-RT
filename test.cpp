#include <cmath>
#include <cstdlib>
#include <iostream>
#include <time.h>
#include <string>
#include <unistd.h>
#include <pthread.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#define e 10
#define FRAME_NUM 100

#define NN 999 //3 bit decimal

using namespace std;
using namespace cv;

extern "C"
{
extern int dgeqrf_(int *m, int *n, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
extern int dorgqr_(int *m, int *n, int *k, double *a, int *lda, double *tau, double *work, int *lwork, int *info);
}


class MatQueue{
public:
	int h,t;
	bool first;
	Mat q[FRAME_NUM];
	MatQueue():h(0),t(-1),first(1){}
	void add(Mat& x){
		t=(t+1)%FRAME_NUM;
		q[t]=x.clone();
		if (first) 
			first=0;
		else if (t==h) 
			h=(h+1)%FRAME_NUM; 
	}
	void reset(){
		h=0; t=-1; first=1;
	}
	void getVideoMatrix(Mat& videoMatrix){
	    int ii = 0, n = 0; // Current column in img
	    for (int k=h;k!=t;k=(k+1)%FRAME_NUM){
	    	ii=0;
		    for (int i = 0; i<q[k].rows; i++)
		    {
		        for (int j = 0; j < q[k].cols; j++)
		        {
		            videoMatrix.at<double>(ii++,n) = q[k].at<uchar>(i,j);
		        }
		    }
		    n++;
		    //cerr<<"x "<<k<<" "<<n<<endl;
		}
        reset();
	}
} frameQueue;

int height=120,width=160;
volatile bool calc_start=0;
volatile bool save_frame=1;
volatile bool start_sub=0;
Mat bg(height,width,CV_8UC1),bg_f(height,width,CV_64FC1);
int isCamera;

void MatToArray(Mat& X, double* array)
{
    int m = X.rows;
    int n = X.cols;
    double* X_line;
    for(int i = 0; i < m; ++i)
    {
        X_line = (double*)(X.data+i*X.step);
        for(int j = 0; j < n; ++j)
        {
            array[i+j*m] = X_line[j];
        }
    }
}

void ArrayToMat(Mat& X, double* array)
{
    int m = X.rows;
    int n = X.cols;
    double* X_line;
    for(int i = 0; i < m; ++i)
    {
        X_line = (double*)(X.data+i*X.step);
        for(int j = 0; j < n; ++j)
        {
            X_line[j] = array[i+j*m];
        }
    }   
}



void qr( double* Q, double* R, double* A, int m, int n)
{
    
    int row;
    // Maximal rank is used by Lapacke
    int rank = m > n ? n : m;
    
    // Tmp Array for Lapacke
    double* tau = (double*)malloc(sizeof(double)*rank );
    
    // Calculate QR factorisations
    int info = 0;
    int lwork = -1;
    int lda=m;
    double iwork;

    dgeqrf_(&m, &n, A, &lda, tau, &iwork, &lwork, &info);
    lwork = (int)iwork;
    double* work = (double*)malloc(sizeof(double)*lwork);
    dgeqrf_(&m, &n, A, &lda, tau, work, &lwork, &info);
    free(work);
    
    // Copy the upper triangular Matrix R (rank x _n) into position
    for(row =0; row < rank; ++row)
    {
        memset(R+row*n, 0, row*sizeof(double)); // Set starting zeros
        memcpy(R+row*n+row, A+row*n+row, (n-row)*sizeof(double)); // Copy upper triangular part from Lapack result.
    }
    
    // Create orthogonal matrix Q (in tmpA)
    info = 0;
    lwork = -1;
    //double iwork;
    dorgqr_(&m, &rank, &rank, A, &m, tau, &iwork, &lwork, &info);
    lwork = (int)iwork;
    //double*
    work = (double*)malloc(sizeof(double)*lwork);
    dorgqr_(&m, &rank, &rank, A, &m, tau, work, &lwork, &info);
    free(work);
    
    //Copy Q (_m x rank) into position

    memcpy(Q, A, sizeof(double)*(m*n));

}


void sign(Mat& x,Mat& y)
{
    int m = x.rows;
    int n = x.cols;
    int i,j;
    //Mat y(m,n,CV_64FC1);
    double* x_line;
    double* y_line;
    for(i = 0; i < m; ++i)
    {
        x_line = (double*)(x.data+i*x.step);
        y_line = (double*)(y.data+i*y.step);
        for(j = 0; j < n; ++j)
        {
            double s = x_line[j];
            y_line[j] =  s > 0 ? 1 : (s < 0? -1:0);
        }
    }
    //return y;
}

void dotMul(Mat& x, Mat& y)
{
    int m = x.rows;
    int n = x.cols;
    int i,j;
    double* x_line;
    double* y_line;
    for(i = 0; i < m; ++i)
    {
        x_line = (double*)(x.data+i*x.step);
        y_line = (double*)(y.data+i*y.step);
        for(j = 0; j < n; ++j)
        {
            y_line[j] = x_line[j]*y_line[j];
        }
    }
}
void wthresh(Mat& x, Mat& y, int t)
{
    Mat tmp = abs(x) - t;
    tmp = (tmp + abs(tmp))/ 2;
	sign(x,y);
	dotMul(tmp,y);
}

double SSGoDec(const Mat& X, Mat& L, Mat& S, int m, int n, int rank, int tau, int power)
{
    double iter_max = 100;
    double error_bound = 0.001;
    int iter = 1;
    ///RMSE
    double RMSE;
    
    L = X.clone();
	Mat tmp(Mat::zeros(m,n, CV_64FC1));
	S = tmp.clone();
	
    //S
    
    while(1)
    {
        ///Y2=randn(n,rank);

        Mat Y2(n,rank,CV_64FC1);
        Mat mean = cv::Mat::zeros(1,1,CV_64FC1);
        Mat sigma= cv::Mat::ones(1,1,CV_64FC1);
		randn(Y2, mean, sigma);


        Mat Y1(m,rank,CV_64FC1);
        
        for(int i=0; i <= power; ++i)
        {
            Y1 = L * Y2;
            Y2 = L.t() * Y1;
        }
		

        double* Qarr = (double*)malloc(sizeof(double)*n*rank);
        double* Rarr = (double*)malloc(sizeof(double)*rank*rank);
        double Y2arr[n*rank];// = (double*)(Y2.data);
        
        MatToArray(Y2, Y2arr);
        
        qr(Qarr, Rarr, Y2arr, n, rank);
        Mat Q(n,rank,CV_64FC1);
        
        ArrayToMat(Q, Qarr);
        
        Mat L_new = (L*Q) * Q.t();

        //Update of S
        Mat T = L - L_new + S;
        L = L_new.clone();

		/*
  		for(int i=0; i < Q.rows; ++i)
		{
			for(int j=0; j < Q.cols;++j)
			{
				double t = Q.at<double>(i,j);
				printf("%4.4lf ",t);
			}
			cout<<endl;
		} 
        */

        //Soft thresholding
       	wthresh(T,S,tau);
        
        
        T = T - S;
        ///RMSE=[RMSE norm(T(:))];
        Mat tmpT = T.reshape(0,T.rows*T.cols);
        RMSE = norm(tmpT,NORM_L2,noArray());
        
        if(RMSE < error_bound || iter > iter_max)
		//if(iter > iter_max)
            break;
        else
            L = L+T;
        
        ++iter;
        

        Mat tmpS = S.reshape(0,S.rows*S.cols);
        double Snorm = norm(tmpS,NORM_L2,noArray());    

        cerr<<"Iter : "<<iter<<endl;
        cerr<<"RMSE : "<<RMSE<<endl;
		cerr<<"Snorm : "<<Snorm<<endl;

        /*
        if (iter%10==2){
        	Mat L_out, S_out;
    		L.convertTo(L_out,CV_8UC1);
    		S.convertTo(S_out,CV_8UC1);
        	getOriginalMatrix(L_out,"bg");
    		getOriginalMatrix(S_out,"fg");
        }
        */
    }
    
    Mat LS = L + S;
    Mat tmpLS = LS.reshape(0, LS.rows*LS.cols);
    Mat tmpX = X.reshape(0, X.rows*X.cols);
    double error = norm(tmpLS-tmpX,NORM_L2,noArray()) / norm(tmpX,NORM_L2,noArray());
    cout << "GoDec end" <<endl;
    return error;
}

void filter(Mat& x){
    int m = x.rows;
    int n = x.cols;
    int i,j;
    double* x_line;
    for(i = 0; i < m; ++i)
    {
        x_line = (double*)(x.data+i*x.step);
        for(j = 0; j < n; ++j)
        {
            if (fabs(x_line[j])<10)
            	x_line[j]=0;
        }
    }
}


void videoHandler(){
	VideoCapture cap;
    int frameNum = FRAME_NUM;

    if(isCamera)
	{
        cap.open(0);
	    cap.set(CV_CAP_PROP_FRAME_WIDTH, width);
		cap.set(CV_CAP_PROP_FRAME_HEIGHT, height);
	}
    else
    {
        cap.open("hall.avi");
        //int num = cap.get(CV_CAP_PROP_FRAME_COUNT);   
        //frameNum = frameNum > num ? num : frameNum;
    }

    if(!cap.isOpened())
    {
        printf("\nCan not open camera or video file\n");
        system("PAUSE");
        exit(0);
    }

    Mat img, img2, img_f, G, G_out;
	
	namedWindow("X",1);
	namedWindow("L",1);
	namedWindow("G",1);
	int cnt=0;
	char s[10];
   	while (1)
    {
        cap >> img;
        if(img.empty())
            break;
        cnt++;    
        cvtColor(img,img,CV_BGR2GRAY); // RGB to GRAY
        
        resize(img,img2,Size(160,120));
        
        imshow("X",img2);
        if (start_sub){
        	imshow("L",bg);
        	img2.convertTo(img_f,CV_64FC1);
        	G=img_f-bg_f;
        	filter(G);
        	normalize(G, G_out, 0, 255, NORM_MINMAX, CV_8UC1, Mat());
        	//equalizeHist(G_norm, G_out);
        	imshow("G",G_out);  
        	//return;
        }
        char c = (char)waitKey(50);
        if(c == 27)   // quit after ESC
            break;
       	cerr << "cnt : "<<cnt <<endl;
        if (save_frame) frameQueue.add(img2);
        if (cnt==FRAME_NUM) calc_start=1;// return;}
        
        if (start_sub && cnt<2000){
        	sprintf(s,"res/%d_X.jpg",cnt);
        	imwrite(s,img2);
        	sprintf(s,"res/%d_L.jpg",cnt);
        	imwrite(s,bg);
        	sprintf(s,"res/%d_G.jpg",cnt);
        	imwrite(s,G_out);
        }
    }

}


int main(void)
{
	pthread_t vhid;
	isCamera=1;
	pthread_create(&vhid,NULL,(void* (*)(void*))videoHandler,NULL);

	//cerr << "here 1"<<endl;
	while (!calc_start){}
	//cerr << "here 2"<<endl;

	Mat X(width*height,FRAME_NUM,CV_64FC1);
	Mat L(width*height,FRAME_NUM,CV_64FC1);
	Mat S(width*height,FRAME_NUM,CV_64FC1);
	while (1){
		//cerr << "here 3"<<endl;
		save_frame = 0;
		frameQueue.getVideoMatrix(X);
		save_frame = 1;
	
		double tau = 8;
		//cerr << "here 4"<<endl;
		double err = SSGoDec(X,L,S,height*width,FRAME_NUM,1,8,0);
		cout << "Err : " << err << endl;
		
		start_sub = 0;
        for (int i = 0; i<L.rows; i++)
            bg_f.at<double>(i/width,i%width) = L.at<double>(i,FRAME_NUM/2);
        bg_f.convertTo(bg,CV_8UC1);
        
        //imwrite("bg.jpg", bg );
        //cerr << "here 5"<<endl;
        //break;
        start_sub = 1;
    }

    

 	return 0;   
}
