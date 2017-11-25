#ifndef DETECTOR_HPP
#define DETECTOR_HPP

#include <caffe/caffe.hpp>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <memory>
#include <vector>
#include <iosfwd>
#include <string>
#include <algorithm>
#include <utility>

using namespace caffe;
using namespace std;
using namespace cv;

struct sortScore{
    float fScore;
    int nNo;
};


class MTCNN_DETECTOR{
public:
    MTCNN_DETECTOR(const string& ,
                   const string& ,
                   const string& ,
                   const string& ,
                   const string& ,
                   const string& ,
                   const string& ,
                   const string& ,
                   const Mat& ,
                   bool ,
                   float ,
                   int ,
                   vector<float>& );
    
    vector<vector<Point2f> > GetFacePoints();
    vector<Rect> GetFaceBoundingBox();
    void FaceDetect();
 
private:    

    void GenerateScales();
    void ImageNormalization(const Mat& , Mat& );
    void NetPredict(const Mat& ,
                    shared_ptr<Net<float> >& ,
                    int ,
                    Size ,
                    vector<Blob<float>*>& 
                    /*Blob<float>**/ );
    
    void NetPredict(const vector<Mat>& ,
                    shared_ptr<Net<float> >& ,
                    int ,
                    Size ,
                    vector<Blob<float>*>&
                    /*Blob<float>**/ );
    void WrapInputLayer(shared_ptr<Net<float> >& , vector<Mat>& );
    void WrapInputLayer(shared_ptr<Net<float> >& , vector<vector<Mat> >& );
    void Preprocess(const cv::Mat& ,
                    std::vector<cv::Mat>& ,
                    int ,
                    Size );
    void Preprocess(const vector<cv::Mat>& ,
                    std::vector<vector<cv::Mat> >& ,
                    int ,
                    Size ); 
    
    void Preprocess(const vector<cv::Mat>& ,
                          vector<cv::Mat>& ,
                          int ,
                          Size );
    
    void GenerateBoundingBox(Blob<float>* , Blob<float>* , vector<Rect>& , vector<float>&, vector<Vec4f>&, float );
    void GenerateBoundingBox(Blob<float>* , Blob<float>* , vector<Rect>& , vector<float>&, vector<Vec4f>&);
    void GenerateBoundingBox(Blob<float>* , Blob<float>* , Blob<float>* , vector<Rect>& , vector<float>&, vector<Vec4f>&, vector<vector<Point2f> >&);
    void GenerateOutputPoints(vector<Blob<float>*>&,vector<Point2f>&);
    void RectifyPoints(Point2f& );
    
    void LocatePoints(const Mat& , vector<Point>& );
    void CopyMat(const Mat& , Mat& , Mat& );
    void Nms(const vector<Rect>& , const vector<float>& , vector<int>&, float , string );
    void InitializeScore(const vector<float>& , vector<sortScore>& );
    void PickResult(vector<Rect>&, vector<float>&, vector<Vec4f>&, const vector<int>&);
    void PickResult(vector<Rect>&, vector<float>&, vector<Vec4f>&, vector<vector<Point2f> >&, const vector<int>&);
    void JoinResult(vector<Rect>&, vector<float>&, vector<Vec4f>&, 
                    const vector<Rect>&, const vector<float>&, const vector<Vec4f>&);
    void ReRectangle(const Point& , const Point& , Rect&);
    void RectifyRectangle(const Rect&, Rect& , Rect& , Vec2f&);
    /*void GenerateScore(Blob<float>*, vector<float>&);
    void GenerateConv5_2(Blob<float>*, vector<Vec4f>&);
    void RectifyBBoxandScore(vector<Rect>&, vector<float>&);*/
    
private:
    Mat m_matProcessMat;
    Mat m_matNormalizeMat;
    
    shared_ptr<Net<float> > m_sptrnetfPNet;
    shared_ptr<Net<float> > m_sptrnetfRNet;
    shared_ptr<Net<float> > m_sptrnetfONet;
    shared_ptr<Net<float> > m_sptrnetfLNet;
    
    string m_strPNetModelPath;
    string m_strRNetModelPath;
    string m_strONetModelPath;
    
    float m_fFactor;
    int m_nminSize;
    int m_nImageWidth;
    int m_nImageHeight;
    
    float m_fProportion;
    vector<float> m_vfThreshold;
    vector<float> m_vfScales;
    bool m_bFastReSize;
    
    int m_nPNetChannelsNum, m_nRNetChannelsNum, m_nONetChannelsNum, m_nLNetChannelsNum;
    
    Size m_szPNetInputGeom, m_szRNetInputGeom, m_szONetInputGeom, m_szLNetInputGeom;
    
    vector<vector<Point2f> >vvpt2fFacePoints;
    vector<Rect>vrectFaceBoundingBox;
    
};

#endif

