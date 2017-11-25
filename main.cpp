#include "prefix.hpp"

int main(int argc, char **argv) {
    Mat matProcessMat = imread(".//image//test1.jpg", -1);
    float fFactor = 0.709f;
    int nminSize = 20;
    vector<float> vfThreshold(3);
    vfThreshold[0] = 0.6;
    vfThreshold[1] = 0.7;
    vfThreshold[1] = 0.7;
    
    string strPNetModelPath = ".//model//det1.prototxt";
    string strRNetModelPath = ".//model//det2.prototxt";
    string strONetModelPath = ".//model//det3.prototxt";
    string strLNetModelPath = ".//model//det4.prototxt";
    string strPNetTrainedPath = ".//model//det1.caffemodel";
    string strRNetTrainedPath = ".//model//det2.caffemodel";
    string strONetTrainedPath = ".//model//det3.caffemodel";
    string strLNetTrainedPath = ".//model//det4.caffemodel";
    
    
    MTCNN_DETECTOR  mdFaceDetector(strPNetModelPath,
                                   strRNetModelPath,
                                   strONetModelPath,
                                   strLNetModelPath,
                                   strPNetTrainedPath,
                                   strRNetTrainedPath,
                                   strONetTrainedPath,
                                   strLNetTrainedPath,
                                   matProcessMat,
                                   false,
                                   fFactor,
                                   nminSize,
                                   vfThreshold);
    mdFaceDetector.FaceDetect();
    
    vector<Rect> vrectFaceBoundingBox = mdFaceDetector.GetFaceBoundingBox();
    vector<vector<Point2f> > vvpt2fFacePoints = mdFaceDetector.GetFacePoints();
   
    cout << "The output BoundingBox's size is:" << vrectFaceBoundingBox.size()<<endl;
    cout << "The output FacePoints' size is:" << vvpt2fFacePoints.size()<<endl;
    Mat matDisplayMat = matProcessMat.clone();
    int m, n;
    for(n=0; n < vrectFaceBoundingBox.size(); n++){
        rectangle(matDisplayMat, vrectFaceBoundingBox[n], Scalar(0, 255, 255), 2, LINE_AA);
        cout << vrectFaceBoundingBox[n] << endl;
        /*for(m=0; m < vvpt2fFacePoints[n].size(); m++){
            circle(matDisplayMat, Point(vvpt2fFacePoints[n][m].x, vvpt2fFacePoints[n][m].y), 2, Scalar(0.0, 0.0, 255.0), -1);
        }*/
    }
    
    namedWindow("matDisplayMat");
    imshow("matDisplayMat", matDisplayMat);
    waitKey(-1);
    
    return 0;
}
