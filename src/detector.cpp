#include "detector.hpp"

MTCNN_DETECTOR::MTCNN_DETECTOR(const string& strPNetModelPath,
                               const string& strRNetModelPath,
                               const string& strONetModelPath,
                               const string& strLNetModelPath,
                               const string& strPNetTrainedPath,
                               const string& strRNetTrainedPath,
                               const string& strONetTrainedPath,
                               const string& strLNetTrainedPath,
                               const Mat& matProcessMat,
                               bool bFastReSize,
                               float fFactor,
                               int nminSize,
                               vector<float>& vfThreshold){
    
    m_matProcessMat = matProcessMat;
    m_strPNetModelPath = strPNetModelPath;
    m_strRNetModelPath = strRNetModelPath;
    m_strONetModelPath = strONetModelPath;
    m_fFactor = fFactor;
    m_nminSize = nminSize;
    m_vfThreshold = vfThreshold;
    m_fProportion = 12.0f/nminSize;
    m_nImageWidth = matProcessMat.cols;
    m_nImageHeight = matProcessMat.rows;
    m_bFastReSize = bFastReSize;
    
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif
    /*Load the network*/
    m_sptrnetfPNet.reset(new Net<float>(strPNetModelPath, TEST));
    m_sptrnetfRNet.reset(new Net<float>(strRNetModelPath, TEST));
    m_sptrnetfONet.reset(new Net<float>(strONetModelPath, TEST));
    m_sptrnetfLNet.reset(new Net<float>(strLNetModelPath, TEST));
    
    m_sptrnetfPNet->CopyTrainedLayersFrom(strPNetTrainedPath);
    m_sptrnetfRNet->CopyTrainedLayersFrom(strRNetTrainedPath);
    m_sptrnetfONet->CopyTrainedLayersFrom(strONetTrainedPath);
    m_sptrnetfLNet->CopyTrainedLayersFrom(strLNetTrainedPath);
    
    CHECK_EQ(m_sptrnetfPNet->num_inputs(), 1);
    CHECK_EQ(m_sptrnetfRNet->num_inputs(), 1);
    
    Blob<float>* bfPNetInput_layer = m_sptrnetfPNet->input_blobs()[0];
    Blob<float>* bfRNetInput_layer = m_sptrnetfRNet->input_blobs()[0];
    Blob<float>* bfONetInput_layer = m_sptrnetfONet->input_blobs()[0];
    CHECK_EQ(m_sptrnetfONet->num_inputs(), 1);
    Blob<float>* bfLNetInput_layer = m_sptrnetfLNet->input_blobs()[0];
    
    m_nPNetChannelsNum = bfPNetInput_layer->channels();
    m_nRNetChannelsNum = bfRNetInput_layer->channels();
    m_nONetChannelsNum = bfONetInput_layer->channels();
    m_nLNetChannelsNum = bfLNetInput_layer->channels();
    
    CHECK(m_nPNetChannelsNum == 3 || m_nPNetChannelsNum == 1)
    << "PNet Input layer should have 1 or 3 channels.";
    
    CHECK(m_nRNetChannelsNum == 3 || m_nRNetChannelsNum == 1)
    << "RNet Input layer should have 1 or 3 channels.";
    
    CHECK(m_nONetChannelsNum == 3 || m_nONetChannelsNum == 1)
    << "ONet Input layer should have 1 or 3 channels.";
    
    CHECK(m_nLNetChannelsNum == 15 || m_nLNetChannelsNum == 5)
    << "LNet Input layer should have 5 or 15 channels.";
    
    m_szPNetInputGeom = Size(bfPNetInput_layer->width(), bfPNetInput_layer->height());
    m_szRNetInputGeom = Size(bfRNetInput_layer->width(), bfRNetInput_layer->height());
    m_szONetInputGeom = Size(bfONetInput_layer->width(), bfONetInput_layer->height());
    m_szLNetInputGeom = Size(bfLNetInput_layer->width(), bfLNetInput_layer->height());
}

void MTCNN_DETECTOR::FaceDetect(){
    GenerateScales();
    
    
    vector<Rect>vrectTotalBoundingBox;
    vector<float>vfTotalScores;
    vector<Vec4f>vv4fTotalConv4_2;
    int nAlterHeight, nAlterWidth;
    Mat matFloatProcessMat, matTempProcessMat;
    cvtColor(m_matProcessMat, matTempProcessMat, CV_BGR2RGB);
    matTempProcessMat.convertTo(matTempProcessMat, CV_32FC3);
    matFloatProcessMat = matTempProcessMat.t();
    /*Mat matTestMat;
    matFloatProcessMat.convertTo(matTestMat, CV_8UC3);
        
    namedWindow("matFloatProcessMat");
    imshow("matFloatProcessMat", matFloatProcessMat);
    waitKey(-1);*/
    
    cout << matFloatProcessMat.type() << endl;
    cout << CV_32FC3 << endl;    
    
    for(int i=0; i < m_vfScales.size(); i++){
        nAlterHeight = static_cast<int>(m_nImageHeight * m_vfScales[i] + 0.5f);
        nAlterWidth = static_cast<int>(m_nImageWidth * m_vfScales[i] + 0.5f);
    
        cout << "ProcessMat's width is:" << m_matProcessMat.cols << endl;
        cout << "ProcessMat's height is:" << m_matProcessMat.rows << endl;
        cout << "ProcessMat's channels is:" << m_matProcessMat.channels() << endl;
        cout << "nAlterWidth is:" << nAlterWidth << endl;
        cout << "nAlterHeight is:" << nAlterHeight << endl;
        if(m_bFastReSize){
            ImageNormalization(matFloatProcessMat, m_matNormalizeMat);
            resize(m_matNormalizeMat, m_matNormalizeMat, Size(nAlterWidth, nAlterHeight));
        }else{
            resize(matFloatProcessMat, m_matNormalizeMat, Size(nAlterWidth, nAlterHeight));
            cout << "nAlterWidth is:" << nAlterWidth << endl;
            cout << "nAlterHeight is:" << nAlterHeight << endl;
            cout << "m_matNormalizeMat channels is:" << m_matNormalizeMat.channels() << endl;
            ImageNormalization(m_matNormalizeMat, m_matNormalizeMat);
        }

        
        //vector<string>vstrOutBlobNames;
        //vstrOutBlobNames.push_back("conv4-2");
        //vstrOutBlobNames.push_back("prob1");
        cout << "NormalizeMat's width is:" << m_matNormalizeMat.cols << endl;
        cout << "NormalizeMat's height is:" << m_matNormalizeMat.rows << endl;
        cout << "NormalizeMat's channels is:" << m_matNormalizeMat.channels() << endl;
        
        Blob<float>* blfProb1OutputLayer;
        Blob<float>* blfPConv4_2OutputLayer;
        vector<Blob<float>*>vblfPOutputLayer;
        
        cout << "PNet is predicting!" << endl;
        cout << "PNet input geom is:" << m_szPNetInputGeom <<endl;
        m_szPNetInputGeom = Size(nAlterWidth, nAlterHeight);
        NetPredict(m_matNormalizeMat, 
                   m_sptrnetfPNet,
                   m_nPNetChannelsNum,
                   m_szPNetInputGeom,
                   vblfPOutputLayer);
        blfProb1OutputLayer = vblfPOutputLayer[1];
        blfPConv4_2OutputLayer = vblfPOutputLayer[0];
        
        vector<Rect>vrectBoundingBox;
        vector<float>vfScores;
        vector<Vec4f>vv4fConv4_2;
        
        cout << "vblfPOutputLayer is:" << vblfPOutputLayer.size() <<endl;
        cout << "blfProb1OutputLayer's width is:" << blfProb1OutputLayer->width() << endl;
        cout << "blfProb1OutputLayer's height is:" << blfProb1OutputLayer->height() << endl;
    
        GenerateBoundingBox(blfProb1OutputLayer, blfPConv4_2OutputLayer, vrectBoundingBox, vfScores, vv4fConv4_2, m_vfScales[i]);
        
        /*vrectFaceBoundingBox.resize(vrectBoundingBox.size());
        for(int m = 0; m < vrectBoundingBox.size(); m++){
            vrectFaceBoundingBox[m] = vrectBoundingBox[m];
        }
                
        return ;*/
        vector<int>viPick;
        Nms(vrectBoundingBox, vfScores, viPick, 0.5f, "Union");
        cout << "viPick's size is:" <<viPick.size() << endl;
        
        if(vrectBoundingBox.size()){
            PickResult(vrectBoundingBox, vfScores, vv4fConv4_2, viPick);
            JoinResult(vrectTotalBoundingBox, vfTotalScores, vv4fTotalConv4_2, vrectBoundingBox, vfScores, vv4fConv4_2);
        }
    }
    if(vrectTotalBoundingBox.size()){
        cout << vrectTotalBoundingBox.size() <<endl;
        for(int m = 0; m < vrectTotalBoundingBox.size(); m++){
            cout << "vrectTotalBoundingBox:" << vrectTotalBoundingBox[m] << endl;
        }
        vector<int>viTotalPick;
        cout << "RNet test 1!" <<endl;
        Nms(vrectTotalBoundingBox, vfTotalScores, viTotalPick, 0.7f, "Union");
        PickResult(vrectTotalBoundingBox, vfTotalScores, vv4fTotalConv4_2, viTotalPick);
        
        Rect rectTempBoundingBox;
        Point ptTLPoint, ptBRPoint;
        vector<Rect>vrectSrcRect(vrectTotalBoundingBox.size());
        vector<Rect>vrectDstRect(vrectTotalBoundingBox.size());
        vector<Vec2f>vv2fTempSize(vrectTotalBoundingBox.size());
        cout << "RNet test 2!" <<endl;
        for(int n = 0; n < vrectTotalBoundingBox.size(); n++){
            rectTempBoundingBox = vrectTotalBoundingBox[n];
            cout << "rectTempBoundingBox:" << rectTempBoundingBox << endl;
            ptTLPoint.x = rectTempBoundingBox.x + vv4fTotalConv4_2[n][0] * rectTempBoundingBox.width;
            ptTLPoint.y = rectTempBoundingBox.y + vv4fTotalConv4_2[n][1] * rectTempBoundingBox.height;
            ptBRPoint.x = rectTempBoundingBox.br().x + vv4fTotalConv4_2[n][2] * rectTempBoundingBox.width;
            ptBRPoint.y = rectTempBoundingBox.br().y + vv4fTotalConv4_2[n][3] * rectTempBoundingBox.height;
            RectifyRectangle(rectTempBoundingBox, vrectSrcRect[n], vrectDstRect[n], vv2fTempSize[n]);
        }
        
        if(vrectTotalBoundingBox.size()){
            cout << "RNet test 3!" <<endl;
            cout << "vrectTotalBoundingBox's size is:" << vrectTotalBoundingBox.size() << endl;
            vector<Mat>vmatTempMat(vrectTotalBoundingBox.size());
            Mat matTmpMat;
            vector<Mat> vmatNormalizeTempMat(vrectTotalBoundingBox.size());
            for(int n = 0; n < vmatTempMat.size(); n++){
                vmatTempMat[n] = Mat::zeros(24, 24, 3);
                matTmpMat = Mat::zeros(vv2fTempSize[n][1], vv2fTempSize[n][0], CV_32FC3);
                cout << "vv2fTempSize[n]" << vv2fTempSize[n] << endl;
                cout << "vrectSrcRect[n]" << vrectSrcRect[n] << endl;
                cout << "vrectDstRect[n]" << vrectDstRect[n] << endl;
                cout << "m_matNormalizeMat.size:" << m_matNormalizeMat.size() << endl;
                matTmpMat(vrectSrcRect[n]) = matTempProcessMat(vrectDstRect[n]);
                matTmpMat = matTmpMat.t();
                resize(matTmpMat, vmatTempMat[n], Size(24, 24));
                ImageNormalization(vmatTempMat[n], vmatNormalizeTempMat[n]);
            }
            
            Blob<float>* blfRProb1OutputLayer;
            Blob<float>* blfRConv5_2OutputLayer;
            vector<Blob<float>*>vblfROutputLayer;
            
            cout << "RNet is predicting!" << endl;
            cout << "vmatNormalizeTempMat's size is:" << vmatNormalizeTempMat.size() << endl;
            NetPredict(vmatNormalizeTempMat, 
                       m_sptrnetfRNet,
                       m_nRNetChannelsNum,
                       m_szRNetInputGeom,
                       vblfROutputLayer);
            
            blfRProb1OutputLayer = vblfROutputLayer[1];
            blfRConv5_2OutputLayer = vblfROutputLayer[0];
            
            cout << "blfRProb1OutputLayer's num is:" << blfRProb1OutputLayer -> num() << endl;
            cout << "blfRConv5_2OutputLayer's num is:" << blfRConv5_2OutputLayer -> num() <<endl;            
            
            vector<float>vfScore;
            vector<Vec4f>vv4fTotalConv5_2;
            /*GenerateScore(blfRProb1OutputLayer, vfScore);
            GenerateConv5_2(blfRConv5_2OutputLayer, vv4fTotalConv5_2);
            RectifyBBoxandScore(vrectTotalBoundingBox, vfScore);*/
            GenerateBoundingBox(blfRProb1OutputLayer, blfRConv5_2OutputLayer, vrectTotalBoundingBox, vfScore, vv4fTotalConv5_2);
            if(vrectTotalBoundingBox.size()){
                vector<int>viRPick;
                Nms(vrectTotalBoundingBox, vfScore, viRPick, 0.7f, "Union");
                PickResult(vrectTotalBoundingBox, vfScore, vv4fTotalConv5_2, viRPick);
                cout << "RNet vrectTotalBoundingBox's size is:" << vrectTotalBoundingBox.size() << endl;
                

                Rect rectTempRBoundingBox;
                Point ptRTLPoint, ptRBRPoint;
                vector<Rect>vrectRSrcRect(vrectTotalBoundingBox.size());
                vector<Rect>vrectRDstRect(vrectTotalBoundingBox.size());
                vector<Vec2f>vv2fRTempSize(vrectTotalBoundingBox.size());
                for(int n=0; vrectTotalBoundingBox.size(); n++){
                    rectTempRBoundingBox = vrectTotalBoundingBox[n];
                    ptRTLPoint.x = rectTempBoundingBox.x + vv4fTotalConv5_2[n][0] * rectTempBoundingBox.width;
                    ptRTLPoint.y = rectTempBoundingBox.y + vv4fTotalConv5_2[n][1] * rectTempBoundingBox.height;
                    ptRBRPoint.x = rectTempBoundingBox.br().x + vv4fTotalConv5_2[n][2] * rectTempBoundingBox.width;
                    ptRBRPoint.y = rectTempBoundingBox.br().y + vv4fTotalConv5_2[n][3] * rectTempBoundingBox.height;
                    ReRectangle(ptRTLPoint, ptRBRPoint, vrectTotalBoundingBox[n]);
                    RectifyRectangle(vrectTotalBoundingBox[n], vrectRSrcRect[n], vrectRDstRect[n], vv2fRTempSize[n]);
                }
                if(vrectTotalBoundingBox.size()){
                    vector<Mat>vmatOTempMat(vrectTotalBoundingBox.size());
                    Mat matOTmpMat;
                    vector<Mat> vmatONormalizeTempMat(vrectTotalBoundingBox.size());
                    for(int n = 0; n < vmatOTempMat.size(); n++){
                        vmatOTempMat[n] = Mat::zeros(48, 48, 3);
                        matOTmpMat = Mat::zeros(vv2fRTempSize[n][1], vv2fRTempSize[n][0], 3);
                        matOTmpMat(vrectRSrcRect[n]) = m_matNormalizeMat(vrectRDstRect[n]);
                        cout << "vrectRSrcRect[n]" << vrectRSrcRect[n] << endl;
                        cout << "vrectRDstRect[n]" << vrectRDstRect[n] << endl;
                        resize(matOTmpMat, vmatOTempMat[n], Size(48, 48));
                        ImageNormalization(vmatOTempMat[n], vmatONormalizeTempMat[n]);
                    }
                    
                    Blob<float>* blfOProb1OutputLayer;
                    Blob<float>* blfOConv6_2OutputLayer; 
                    Blob<float>* blfOPointsOutputLayer; 
                    vector<Blob<float>*>vblfOOutputLayer;
                    cout << "ONet is predicting!" << endl;
                    NetPredict(vmatONormalizeTempMat, 
                               m_sptrnetfONet,
                               m_nONetChannelsNum,
                               m_szONetInputGeom,
                               vblfOOutputLayer);
                    blfOProb1OutputLayer = vblfOOutputLayer[2];
                    blfOPointsOutputLayer = vblfOOutputLayer[1];
                    blfOConv6_2OutputLayer = vblfOOutputLayer[0];
                    
                    vector<float>vfOScore;
                    vector<Vec4f>vv4fTotalOConv6_2;
                    vector<vector<Point2f> >vvpt2fOPoints;
                    
                    GenerateBoundingBox(blfOProb1OutputLayer, blfOConv6_2OutputLayer, blfOPointsOutputLayer, vrectTotalBoundingBox, vfOScore, vv4fTotalOConv6_2, vvpt2fOPoints);
                    
                    if(vrectTotalBoundingBox.size()){
                        Rect rectTempOBoundingBox;
                        Point ptOTLPoint, ptOBRPoint;
                        vector<Rect>vrectOSrcRect(vrectTotalBoundingBox.size());
                        vector<Rect>vrectODstRect(vrectTotalBoundingBox.size());
                        vector<Vec2f>vv2fOTempSize(vrectTotalBoundingBox.size());
                        for(int n=0; vrectTotalBoundingBox.size(); n++){
                            rectTempOBoundingBox = vrectTotalBoundingBox[n];
                            ptOTLPoint.x = rectTempOBoundingBox.x + vv4fTotalOConv6_2[n][0] * rectTempOBoundingBox.width;
                            ptOTLPoint.y = rectTempOBoundingBox.y + vv4fTotalOConv6_2[n][1] * rectTempOBoundingBox.height;
                            ptOBRPoint.x = rectTempOBoundingBox.br().x + vv4fTotalOConv6_2[n][2] * rectTempOBoundingBox.width;
                            ptOBRPoint.y = rectTempOBoundingBox.br().y + vv4fTotalOConv6_2[n][3] * rectTempOBoundingBox.height;
                            vrectTotalBoundingBox[n] = Rect(ptOTLPoint.x, ptOTLPoint.y, ptOBRPoint.x - ptOTLPoint.x, ptOBRPoint.y - ptOTLPoint.y);
                        }
                        vector<int>viOPick;
                        Nms(vrectTotalBoundingBox, vfOScore, viOPick, 0.7f, "Min");
                        PickResult(vrectTotalBoundingBox, vfOScore, vv4fTotalOConv6_2, vvpt2fOPoints, viOPick);
                        
                        if(vrectTotalBoundingBox.size()){
                            //vector<Mat>vmatLTempMat(vrectTotalBoundingBox.size());
                            
                            int nMaxWidth = -1, nMaxHeight = -1;
                            for(int n = 0; n < vrectTotalBoundingBox.size(); n++){
                                //vmatLTempMat[n] = Mat::zeros(24, 24, 15);
                                if(nMaxWidth <  vrectTotalBoundingBox[n].width){
                                    nMaxWidth = vrectTotalBoundingBox[n].width;
                                }
                                if(nMaxHeight <  vrectTotalBoundingBox[n].height){
                                    nMaxHeight = vrectTotalBoundingBox[n].height;
                                }
                            }
                            
                            nMaxWidth = static_cast<int>(0.25f * nMaxWidth);
                            nMaxHeight = static_cast<int>(0.25f * nMaxHeight);
                            
                            nMaxWidth = (nMaxWidth%2?nMaxWidth:nMaxWidth+1);
                            nMaxHeight = (nMaxHeight%2?nMaxHeight:nMaxHeight+1);
                            
                            vector<Rect>vrectLSrcRect(vrectTotalBoundingBox.size() * 5);
                            vector<Rect>vrectLDstRect(vrectTotalBoundingBox.size() * 5);
                            vector<Vec2f>vv2fLTempSize(vrectTotalBoundingBox.size() * 5);
                            //vector<vector<Point2f>>vpt2fPoints(vrectTotalBoundingBox.size());
                            vector<Point2f>vpt2fTempPoints;
                            Rect rectTempRect;
                            for(int n = 0; n < vrectTotalBoundingBox.size(); n++){
                                vpt2fTempPoints = vvpt2fOPoints[n];
                                for(int m = 0; m < 5; m++){
                                    rectTempRect = Rect(static_cast<int>(vpt2fTempPoints[n].x - 0.5f * nMaxWidth + 0.5f),
                                                        static_cast<int>(vpt2fTempPoints[n].y - 0.5f * nMaxHeight + 0.5f),
                                                        nMaxWidth,
                                                        nMaxHeight);
                                    
                                    RectifyRectangle(rectTempRect, vrectLSrcRect[5*n + m], vrectLDstRect[5*n + m], vv2fLTempSize[5 * n + m]);
                                }
                            }
                            
                            vector<vector<Mat> >vvmatLNormalizeTempMat(vrectTotalBoundingBox.size());
                            for(int n = 0; n < vvmatLNormalizeTempMat.size(); n++){
                                vector<Mat>vmatLNormalizeTempMat(5);
                                vvmatLNormalizeTempMat[n] = vmatLNormalizeTempMat;
                            }
                            
                            Mat matLTmpMat;
                            Mat matLTempMat;
                            //vector<Mat> vmatLNormalizeTempMat(vrectTotalBoundingBox.size());

                            vector<vector<Blob<float>*> >vvblfLOutputLayer(vrectTotalBoundingBox.size()); 
                            vvpt2fFacePoints.resize(vrectTotalBoundingBox.size());
                            vrectFaceBoundingBox.resize(vrectTotalBoundingBox.size());
                            for(int m = 0; m < vrectTotalBoundingBox.size(); m++){
                                vrectFaceBoundingBox[m] = vrectTotalBoundingBox[m];
                                vpt2fTempPoints = vvpt2fOPoints[m];
                                for(int n = 0; n < 5; n++){
                                    matLTempMat = Mat::zeros(24, 24, 3);
                                    matLTmpMat = Mat::zeros(vv2fLTempSize[n][1], vv2fLTempSize[n][0], 3);
                                    matLTmpMat(vrectLSrcRect[n]) = m_matNormalizeMat(vrectLDstRect[n]);
                                    resize(matLTmpMat, matLTempMat, Size(24, 24));
                                    ImageNormalization(matLTempMat, vvmatLNormalizeTempMat[m][n]);
                                }
                                cout << "LNet is predicting!" << endl;
                                NetPredict(vvmatLNormalizeTempMat[m], 
                                           m_sptrnetfLNet,
                                           m_nLNetChannelsNum,
                                           m_szLNetInputGeom,
                                           vvblfLOutputLayer[m]);
                                vector<Point2f>vpt2fOutputPoints;
                                Point2f pt2fTempPoints;
                                GenerateOutputPoints(vvblfLOutputLayer[m],vpt2fOutputPoints);
                                vvpt2fFacePoints[m].resize(vpt2fOutputPoints.size());
                                for(int l=0; l < vpt2fOutputPoints.size(); l++){
                                    pt2fTempPoints = vpt2fOutputPoints[l];
                                    RectifyPoints(pt2fTempPoints);
                                    vvpt2fFacePoints[m][l] = Point2f(vpt2fTempPoints[l].x+(pt2fTempPoints.x - 0.5f)*vrectTotalBoundingBox[m].width,
                                                                     vpt2fTempPoints[l].y+(pt2fTempPoints.y - 0.5f)*vrectTotalBoundingBox[m].height);             
                                }
                            }
                        }
                    }
                }                
            } 
        }
    }

}

void MTCNN_DETECTOR::RectifyPoints(Point2f& pt2fPoints){
    if(fabs(pt2fPoints.x - 0.5f) < 0.35f){
        pt2fPoints.x = 0.5f;
    }
    if(fabs(pt2fPoints.y - 0.5f) < 0.35f){
        pt2fPoints.y = 0.5f;
    }    
}


/*void MTCNN_DETECTOR::RectifyBBoxandScore(vector<Rect>&vrectTotalBoundingBox, vector<float>&vfScore){
    if(!vfScore.size()){
        cerr << "The RectifyBBoxand input Score is empty!" << endl;
        exit(-1);
    }
    
    vector<float>vfTempScore = vfScore;
    vector<Rect>vrectTempTotalBoundingBox = vrectTotalBoundingBox;
    vfScore.clear();
    vrectTotalBoundingBox.clear();
    int n;
    for(n = 0; n < vfTempScore.size(); n++){
        if(vfTempScore[n] > m_vfThreshold[1]){
            vfScore.push_back(vfTempScore[n]);
            vrectTotalBoundingBox.push_back(vrectTempTotalBoundingBox[n]);
        }
    }
}

void MTCNN_DETECTOR::GenerateConv5_2(Blob<float>*pblfRConv5_2OutputLayer, vector<Vec4f>&vv4fTotalConv5_2){
    if(vv4fTotalConv5_2.size()){
        vv4fTotalConv5_2.clear();
    }
    
    int nChannels = pblfRConv5_2OutputLayer->channels();
    int nWidth = pblfRConv5_2OutputLayer->width();
    int nHeight = pblfRConv5_2OutputLayer->height(); 
    CHECK(nChannels == 4);
    
    const float* pfConv5_2Begin = pblfRConv5_2OutputLayer -> cpu_data();
    
    vector<Mat> vmatConv5_2Mat(nChannels);
    
    int n;
    for(n=0; n < vmatConv5_2Mat.size(); n++){
        vmatConv5_2Mat[n] = Mat(nHeight, nWidth, CV_32FC1, pfConv5_2Begin + n * nHeight * nWidth);
    }
    
    vector<float *>vpfmatConv5_2Mat(nChannels);
    
    for(n=0; n < vpfmatConv5_2Mat.size(); n++){
        vpfmatConv5_2Mat[n] = (float *)(vmatConv5_2Mat[n].data);
    }
    
    int r, c;
    for(r = 0; r < nHeight; r++){
        for(c = 0; c < nWidth; c++){
            for(){
                
            }
        }
    }
}

void MTCNN_DETECTOR::GenerateScore(Blob<float>*pbfProb1OutputLayer, vector<float>&vfScore){
    if(vfScore.size()){
        vfScore.clear();
    }
    
    int nChannels = pbfProb1OutputLayer->channels();
    int nWidth = pbfProb1OutputLayer->width();
    int nHeight = pbfProb1OutputLayer->height();
    CHECK(nChannels == 2);
    
    const float* pfProb1Begin = pbfProb1OutputLayer->cpu_data();
    
    vector<Mat> vmatProb1Mat(nChannels);
    
    int n;
    for(n = 0; n < vmatProb1Mat.size(); n++){
        vmatProb1Mat[n] = Mat(nHeight, nWidth, CV_32FC1, pfProb1Begin + n * nHeight * nWidth);
    }
    
    float* pfmatProb1Mat = (float*)(vmatProb1Mat[1].data);
    
    int r, c;
    for(r = 0; r < nHeight; r++){
        for(c = 0; c < nWidth; c++){
            vfScore.push_back(pfmatProb1Mat[r * nWidth + c]);
        }
    }
}*/

void MTCNN_DETECTOR::RectifyRectangle(const Rect&rectBoundingBox, Rect& rectSrcRect, Rect& rectDstRect, Vec2f&v2fTempSize){
    v2fTempSize[0] = rectBoundingBox.width;
    v2fTempSize[1] = rectBoundingBox.height;
    
    rectSrcRect.x = 0;
    rectSrcRect.y = 0;
    rectSrcRect.width = rectBoundingBox.width;
    rectSrcRect.height = rectBoundingBox.height;
    
    rectDstRect = rectBoundingBox;
    
    cout << "RectifyRectangle" << rectBoundingBox << endl;
    
    if(rectDstRect.x < 0){
        rectSrcRect.width = rectSrcRect.width + rectDstRect.x;rectSrcRect.x = -rectDstRect.x; 
        rectDstRect.width = rectDstRect.width + rectDstRect.x;rectDstRect.x = 0;
    }
    if(rectDstRect.y < 0){
        rectSrcRect.height = rectSrcRect.height + rectDstRect.y;rectSrcRect.y = -rectDstRect.y;
        rectDstRect.height = rectDstRect.height + rectDstRect.y;rectDstRect.y = 0;
    }
    
    if(rectDstRect.x + rectDstRect.width >= m_nImageWidth){
        rectSrcRect.width = rectSrcRect.width -(rectDstRect.width - (m_nImageWidth - 1 -rectDstRect.x));
        rectDstRect.width = m_nImageWidth - 1 - rectDstRect.x;
    }
    
    if(rectDstRect.y + rectDstRect.height >= m_nImageHeight){
        rectSrcRect.height = rectSrcRect.height - (rectDstRect.height - (m_nImageHeight - 1 - rectDstRect.y));
        rectDstRect.height = m_nImageHeight - 1 - rectDstRect.y;
    }
        
}

void MTCNN_DETECTOR::GenerateOutputPoints(vector<Blob<float>*>&vblfLOutputLayer, vector<Point2f>&vpt2fOutputPoints){
    if(vpt2fOutputPoints.size()){
        vpt2fOutputPoints.clear();
    }
    
    for(int n=0; n < vblfLOutputLayer.size(); n++){
        float* pfOutputBegin = vblfLOutputLayer[n]->mutable_cpu_data();

        int nPointsOutputLayerNum = vblfLOutputLayer[n]->num();
        int nPointsOutputLayerChannels = vblfLOutputLayer[n]->channels();;
        int nPointsOutputLayerWidth = vblfLOutputLayer[n]->width();
        int nPointsOutputLayerHeight = vblfLOutputLayer[n]->height();
        
        CHECK(nPointsOutputLayerNum == 1);
        
        vector<Mat>vmatPointsMat(nPointsOutputLayerChannels);
        
        for(int n = 0; n < nPointsOutputLayerChannels; n++){
            vmatPointsMat[n] = Mat(nPointsOutputLayerHeight, nPointsOutputLayerWidth, CV_32FC1,
                                   pfOutputBegin + n * nPointsOutputLayerWidth * nPointsOutputLayerHeight);
        }
        
        Mat matTempPointsOutputLayer = vmatPointsMat[0];
        
        CHECK(matTempPointsOutputLayer.rows == 2 && matTempPointsOutputLayer.cols == 1);
        int r, c;
        Point2f ptTempPoint = Point2f(*(matTempPointsOutputLayer.ptr<float>(0)),
                                      *(matTempPointsOutputLayer.ptr<float>(1)));
                
        vpt2fOutputPoints.push_back(ptTempPoint);        
    }   
    
}

void MTCNN_DETECTOR::ReRectangle(const Point& ptTLPoint, const Point& ptBRPoint, Rect& rectTotalBox){
    int nHeight = ptBRPoint.y - ptTLPoint.y;
    int nWidth = ptBRPoint.x - ptTLPoint.x;
    
    int nLength = max(nHeight, nWidth);
    
    rectTotalBox.x = static_cast<int>(rectTotalBox.x + 0.5f * nWidth - nLength * 0.5f);
    rectTotalBox.y = static_cast<int>(rectTotalBox.y + 0.5f * nHeight - nLength * 0.5f);
    rectTotalBox.width = nLength;
    rectTotalBox.height = nLength;
    
}

void MTCNN_DETECTOR::JoinResult(vector<Rect>&vrectTotalBoundingBox, vector<float>&vfTotalScores, vector<Vec4f>&vv4fTotalConv4_2, 
                                const vector<Rect>&vrectBoundingBox, const vector<float>&vfScores, const vector<Vec4f>&vv4fConv4_2){
    if(vrectBoundingBox.size() != vfScores.size() || vrectBoundingBox.size() != vv4fConv4_2.size() || vfScores.size() != vv4fConv4_2.size()){
        cerr << "JoinResult input data error!" << endl;
    }
    
    for(int n = 0; n < vrectBoundingBox.size(); n++){
        vrectTotalBoundingBox.push_back(vrectBoundingBox[n]);
        vfTotalScores.push_back(vfScores[n]);
        vv4fTotalConv4_2.push_back(vv4fConv4_2[n]);
    }
}

void  MTCNN_DETECTOR::PickResult(vector<Rect>&vrectBoundingBox, vector<float>&vfScores, vector<Vec4f>&vv4fConv, const vector<int>&viPick){
    
    if(!vrectBoundingBox.size() || !vfScores.size() || !vv4fConv.size()){
        cerr << "PickResult input data error!" << endl;
        exit(-1);
    }
    
    vector<Rect>vvpt2fTempBoundingBox = vrectBoundingBox;
    vector<float>vfTempScores = vfScores;
    vector<Vec4f>vv4fTempConv = vv4fConv;
    
    vrectBoundingBox.clear();
    vfScores.clear();
    vv4fConv.clear();
    
    
    for(int n=0; n < viPick.size(); n++){
        vrectBoundingBox.push_back(vvpt2fTempBoundingBox[viPick[n]]);
        vfScores.push_back(vfTempScores[viPick[n]]);
        vv4fConv.push_back(vv4fTempConv[viPick[n]]);
    }
}

void  MTCNN_DETECTOR::PickResult(vector<Rect>&vrectBoundingBox, vector<float>&vfScores, vector<Vec4f>&vv4fConv, vector<vector<Point2f> >&vvpt2fPoints, const vector<int>&viPick){
    
    if(!vrectBoundingBox.size() || !vfScores.size() || !vv4fConv.size()){
        cerr << "PickResult input data error!" << endl;
        exit(-1);
    }
    
    vector<Rect>vvpt2fTempBoundingBox = vrectBoundingBox;
    vector<float>vfTempScores = vfScores;
    vector<Vec4f>vv4fTempConv = vv4fConv;
    vector<vector<Point2f> >vvpt2fTempPoints = vvpt2fPoints;
    
    vrectBoundingBox.clear();
    vfScores.clear();
    vv4fConv.clear();
    
    
    for(int n=0; n < viPick.size(); n++){
        vrectBoundingBox.push_back(vvpt2fTempBoundingBox[viPick[n]]);
        vfScores.push_back(vfTempScores[viPick[n]]);
        vv4fConv.push_back(vv4fTempConv[viPick[n]]);
        vvpt2fPoints.push_back(vvpt2fTempPoints[viPick[n]]);
    }
}


void MTCNN_DETECTOR::GenerateScales(){
    if(m_vfScales.size()){
        m_vfScales.clear();
    }
    
    float fFactorIter = 1.0f;
    
    //int nImageWidth = m_matProcessMat.cols;
    //int nImageHeight = m_matProcessMat.rows;
    
    int nminLength = min(m_nImageWidth, m_nImageHeight);
    
    nminLength = static_cast<int>(m_fProportion * nminLength);
    
    int n = 0;
    while(nminLength >= 12){
        m_vfScales.push_back(m_fProportion * fFactorIter);
        nminLength = static_cast<int>(nminLength * m_fFactor);
        fFactorIter *= m_fFactor;
        cout << "m_vfScales is:" << m_vfScales[n] <<endl;
        n++;
    }
}

void MTCNN_DETECTOR::ImageNormalization(const Mat& matSrcMat, Mat& matDstMat){
    Mat matTempMat = matSrcMat.clone();
    
    if(!matDstMat.empty()){
        matDstMat.release();
    }
    matDstMat = Mat(matTempMat.rows, matTempMat.cols, CV_32FC3, Scalar(0.0, 0.0, 0.0));
    
    int nImageWidth = matTempMat.cols;
    int nImageHeight = matTempMat.rows;
    int nImageChannels = matTempMat.channels();
    
    Vec3f* pmatTempMat = matTempMat.ptr<Vec3f>(0);
    Vec3f* pfmatDstMat = matDstMat.ptr<Vec3f>(0);
    
    int n, r, c;
    int nTempR;
    
    //cout << "ImageNormalization.data:" << endl;
    for(r = 0; r < nImageHeight; r++){
        nTempR = r * nImageWidth;
        for(c = 0; c < nImageWidth; c++){
            pfmatDstMat[nTempR + c][0] = (pmatTempMat[nTempR + c][0] - 127.5f)*(1.0f/127.5f);
            pfmatDstMat[nTempR + c][1] = (pmatTempMat[nTempR + c][1] - 127.5f)*(1.0f/127.5f);
            pfmatDstMat[nTempR + c][2] = (pmatTempMat[nTempR + c][2] - 127.5f)*(1.0f/127.5f);
            //cout << pfmatDstMat[nTempR + c] << " ";
        }
        //cout << endl;
    } 
}

void MTCNN_DETECTOR::NetPredict(const vector<Mat>& vmatSrcImage, 
                                shared_ptr<Net<float> >& Net_,
                                int nChannelNum,
                                Size szInputGeom,
                                /*vector<string>&vstrOutBlobNames,*/
                                vector<Blob<float>*>& blfpOutputLayer
                                /*Blob<float>* blfConv4_2OutputLayer*/){
    
    //if(!blfpOutputLayer.size()){
        //cerr << "The number of output layer error!" <<endl;
        //exit(-1);
    //}
    Blob<float>* pbfInputLayer = Net_->input_blobs()[0];
    
    cout << "NetPredict's size is:" << szInputGeom <<endl;
    cout << "nChannelNum's size is:" << nChannelNum <<endl;
    
    if(nChannelNum > 3){
        pbfInputLayer->Reshape(1, nChannelNum, szInputGeom.height, szInputGeom.width);
    }else{
        pbfInputLayer->Reshape(vmatSrcImage.size(), nChannelNum, szInputGeom.height, szInputGeom.width);
    }
    
    Net_->Reshape();
    if(nChannelNum > 3){
        vector<Mat>vmatInputChannels;
        
        WrapInputLayer(Net_, vmatInputChannels);
        Preprocess(vmatSrcImage, vmatInputChannels, nChannelNum, szInputGeom);        
  
        CHECK((float *)(vmatInputChannels[0].data) == Net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
            
        Net_->Forward();
        
        blfpOutputLayer = Net_->output_blobs();
    }else{
        vector<vector<Mat> >vvmatInputChannels;
        
        WrapInputLayer(Net_, vvmatInputChannels);
        Preprocess(vmatSrcImage, vvmatInputChannels, nChannelNum, szInputGeom);
    
        CHECK((float *)(vvmatInputChannels[0][0].data) == Net_->input_blobs()[0]->cpu_data())
            << "Input channels are not wrapping the input layer of the network.";
        
        Net_->Forward();
    
        blfpOutputLayer = Net_->output_blobs();
    }    
 }
 
 void MTCNN_DETECTOR::NetPredict(const Mat& matSrcImage, 
                                shared_ptr<Net<float> >& Net_,
                                int nChannelNum,
                                Size szInputGeom,
                                /*vector<string>&vstrOutBlobNames,*/
                                vector<Blob<float>*>& blfOutputLayer
                                /*Blob<float>* blfConv4_2OutputLayer*/){
    cout << "NetPredict2 process 1!" << endl;
    Blob<float>* pbfInputLayer = Net_->input_blobs()[0];
    pbfInputLayer->Reshape(1, nChannelNum, szInputGeom.height, szInputGeom.width);
    
    cout << "NetPredict2 process 2!" << endl;
    Net_->Reshape();
    
    vector<Mat>vmatInputChannels;
    
    cout << "NetPredict2 process 3!" << endl;
    WrapInputLayer(Net_, vmatInputChannels);
    cout << "NetPredict2 process 4!" << endl;
    cout << "Preprocess nChannelNum is:" << nChannelNum <<endl;
    Preprocess(matSrcImage, vmatInputChannels, nChannelNum, szInputGeom);
    
    CHECK((float *)(vmatInputChannels[0].data) == Net_->input_blobs()[0]->cpu_data())
        << "Input channels are not wrapping the input layer of the network.";
    cout << "NetPredict2 process 5!" << endl;
    Net_->Forward();
    cout << "NetPredict2 process 6!" << endl;
    blfOutputLayer = Net_->output_blobs();
    
    cout << "blfOutputLayer's size is:" << blfOutputLayer.size() <<endl;
    cout << "blfOutputLayer's width is:" << blfOutputLayer[0]->width() <<endl;
    cout << "blfOutputLayer's height is:" << blfOutputLayer[0]->height() <<endl;
    
}

void MTCNN_DETECTOR::WrapInputLayer(shared_ptr<Net<float> >& Net_, vector<Mat>& vmatInputChannels){
    Blob<float>* bfInputLayer = Net_->input_blobs()[0];
    
    int nWidth = bfInputLayer->width();
    int nHeight = bfInputLayer->height();
    
    cout << "WrapInputLayer nWidth is:"<< nWidth  <<endl;
    cout << "WrapInputLayer nHeight is:" << nHeight <<endl;
    
    float* pfInputData = bfInputLayer->mutable_cpu_data();
    
    for(int i=0; i < bfInputLayer->channels(); i++){
        Mat matChannel(nHeight, nWidth, CV_32FC1, pfInputData);
        vmatInputChannels.push_back(matChannel);
        pfInputData += nWidth * nHeight;
    }
    
}
/*RNet WrapInputLayer*/
void MTCNN_DETECTOR::WrapInputLayer(shared_ptr<Net<float> >& Net_, vector<vector<Mat> >& vvmatInputChannels){
    int nBlobNum = (Net_->input_blobs()).size();
    cout << "nBlobNum:" << nBlobNum << endl;
    
    for(int m = 0; m < nBlobNum; m++){
        Blob<float>* bfInputLayer = Net_->input_blobs()[m];
     
        int nWidth = bfInputLayer->width();
        int nHeight = bfInputLayer->height();
    
        float* pfInputData = bfInputLayer->mutable_cpu_data();
    
        int i, j;
        vector<Mat>vmatInputChannels;

        for(i=0; i < bfInputLayer->channels(); i++){
            Mat matChannel(nHeight, nWidth, CV_32FC1, pfInputData);
            vmatInputChannels.push_back(matChannel);
            pfInputData += nWidth * nHeight;
        }
        vvmatInputChannels.push_back(vmatInputChannels);
    }
    
}

void MTCNN_DETECTOR::Preprocess(const cv::Mat& matSrcImage,
                                std::vector<cv::Mat>& vmatInputChannels,
                                int nChannelNum,
                                Size szInputGeom) {
    Mat matSample;
    
    matSample = matSrcImage;
    
    Mat matSampleResized;
    cout << "SampleImage's size is:" << matSample.size() << endl;
    cout << "szInputGeom's size is:" << szInputGeom << endl;
    if(matSample.size() != szInputGeom)
        resize(matSample, matSample, szInputGeom);
    //else
        //matSampleResized = matSample;
    
    //Mat matSampleFloat;
    
    //if(nChannelNum == 3)
        //matSampleResized.convertTo(matSampleFloat, CV_32FC3);
    
    split(matSample, vmatInputChannels);
}

void MTCNN_DETECTOR::Preprocess(const vector<cv::Mat>& vmatSrcImage,
                                std::vector<vector<cv::Mat> >& vvmatInputChannels,
                                int nChannelNum,
                                Size szInputGeom) {
    vector<Mat> vmatSample(vmatSrcImage.size());
    int n;
    
    vmatSample = vmatSrcImage;

    CHECK(vmatSrcImage.size() == vvmatInputChannels.size());
    /*vector<Mat> vmatSampleResized(vmatSrcImage.size());
    if(vmatSample[0].size() != szInputGeom){
        for(n = 0; n < vmatSample.size(); n++){
            resize(vmatSample[n], vmatSampleResized[n], szInputGeom); 
        }
    }
    else
        vmatSampleResized = vmatSample;
    
    vector<Mat> vmatSampleFloat(vmatSrcImage.size());
    
    if(nChannelNum == 3){
        for(n=0; n < vmatSampleResized.size(); n++){
            vmatSampleResized[n].convertTo(vmatSampleFloat[n], CV_32FC3);
        }
    }*/

    for(n=0; n < vmatSample.size(); n++){
        split(vmatSample[n], vvmatInputChannels[n]);
    }
}


void MTCNN_DETECTOR::Preprocess(const vector<cv::Mat>& vmatSrcImage,
                                vector<cv::Mat>& vvmatInputChannels,
                                int nChannelNum,
                                Size szInputGeom) {
    vector<Mat> vmatSample(vmatSrcImage.size());
    int n;
    vmatSample = vmatSrcImage;
    
    vector<Mat> vmatSampleResized(vmatSrcImage.size());
    if(vmatSample[0].size() != szInputGeom){
        for(n = 0; n < vmatSample.size(); n++){
            resize(vmatSample[n], vmatSampleResized[n], szInputGeom); 
        }
    }
    else
        vmatSampleResized = vmatSample;
    
    vector<Mat> vmatSampleFloat(vmatSrcImage.size());
    
    if(nChannelNum >= 3){
        for(n=0; n < vmatSampleResized.size(); n++){
            vmatSampleResized[n].convertTo(vmatSampleFloat[n], CV_32FC3);
        }
    }

    int m;
    vector<Mat>vmatTempSplitImage(3);
    for(n=0; n < vmatSampleResized.size(); n++){
        split(vmatSampleResized[n], vmatTempSplitImage);
        for(m=0; m < vmatTempSplitImage.size(); m++){
            vvmatInputChannels[3*n + m];
        }
    }
    
    
}

void MTCNN_DETECTOR::GenerateBoundingBox(Blob<float>* blfProbOutputLayer, Blob<float>* blfConvOutputLayer, Blob<float>* blfPointsOutputLayer, vector<Rect>& vrectBoundingBox, vector<float>&vfScores, vector<Vec4f>&vv4fConv, vector<vector<Point2f> >&vvpt2fPoints){
    float* pfProbBegin = blfProbOutputLayer->mutable_cpu_data();
    float* pfConvBegin = blfConvOutputLayer->mutable_cpu_data();
    float* pfPointsBegin = blfPointsOutputLayer->mutable_cpu_data();
    
    int nProbOutputLayerNum = blfProbOutputLayer->num();
    int nProbOutputLayerChannels = blfProbOutputLayer->channels();;
    int nProbOutputLayerWidth = blfProbOutputLayer->width();
    int nProbOutputLayerHeight = blfProbOutputLayer->height();
    
    int nConvOutputLayerNum = blfConvOutputLayer->num();
    int nConvOutputLayerChannels = blfConvOutputLayer->channels();;
    int nConvOutputLayerWidth = blfConvOutputLayer->width();
    int nConvOutputLayerHeight = blfConvOutputLayer->height();
    
    int nPointsOutputLayerNum = blfPointsOutputLayer->num();
    int nPointsOutputLayerChannels = blfPointsOutputLayer->channels();;
    int nPointsOutputLayerWidth = blfPointsOutputLayer->width();
    int nPointsOutputLayerHeight = blfPointsOutputLayer->height();
    
    CHECK(nProbOutputLayerNum == 1 && nConvOutputLayerNum == 1 && nPointsOutputLayerNum ==1 &&
          nProbOutputLayerChannels == 2 && nConvOutputLayerChannels == 4 && nPointsOutputLayerChannels == 10);
    
    cout << nProbOutputLayerNum << endl;
    cout << nProbOutputLayerChannels << endl;
    cout << nProbOutputLayerWidth << endl;
    cout << nProbOutputLayerHeight << endl;
    cout << nConvOutputLayerNum << endl;
    cout << nConvOutputLayerChannels << endl;
    cout << nConvOutputLayerWidth << endl;
    cout << nConvOutputLayerHeight << endl;
    cout << nPointsOutputLayerNum << endl;
    cout << nPointsOutputLayerChannels << endl;
    cout << nPointsOutputLayerWidth << endl;
    cout << nPointsOutputLayerHeight << endl;
    
    vector<Mat>vmatProbMat(nProbOutputLayerChannels),
               vmatConvMat(nConvOutputLayerChannels),
               vmatPointsMat(nPointsOutputLayerChannels);
    
    int n;
    for(n = 0; n < nProbOutputLayerChannels; n++){
        vmatProbMat[n] = Mat(nProbOutputLayerHeight, nProbOutputLayerWidth, CV_32FC1, 
                             pfProbBegin + n*nProbOutputLayerHeight*nProbOutputLayerWidth);
    }
    
    for(n = 0; n < nConvOutputLayerChannels; n++){
        vmatConvMat[n] = Mat(nConvOutputLayerHeight, nConvOutputLayerWidth, CV_32FC1,
                             pfConvBegin + n*nConvOutputLayerHeight*nConvOutputLayerWidth);
    }
    
    for(n = 0; n < nPointsOutputLayerChannels; n++){
        vmatPointsMat[n] = Mat(nPointsOutputLayerHeight, nPointsOutputLayerWidth, CV_32FC1,
                             pfPointsBegin + n*nPointsOutputLayerHeight*nPointsOutputLayerWidth);
    }
    
    //vector<float>vfScores;
    //vector<Vec4f>vv4fConv;
    
    int r;
    float* pfProbMat = (float *)(vmatProbMat[1].data);
    //int nProbMatWidth = vmatProbMat[1].width;
    for(r = 0; r < vmatProbMat[1].rows; r++){
        vfScores.push_back(*(pfProbMat + r * nProbOutputLayerWidth));
    }
    
    vector<float *>pvv4fConv(nConvOutputLayerChannels);
    for(n = 0; n < nConvOutputLayerChannels; n++){
        pvv4fConv[n] = (float *)(vmatConvMat[n].data);
    }
    
    Vec4f v4fTempConv;

    for(r = 0; r < vmatConvMat[1].rows; r++){
        for(n = 0; n < nConvOutputLayerChannels; n++){
            v4fTempConv[n] = pvv4fConv[n][r * nConvOutputLayerWidth];
        }
        vv4fConv.push_back(v4fTempConv);
    }
    
    vector<float *>pvvpt2fPoints(nPointsOutputLayerChannels);
    for(n =0; n < nPointsOutputLayerChannels; n++){
        pvvpt2fPoints[n] = (float *)(vmatPointsMat[n].data);
    }
    
    //Point2f pt2fPoints;
    vector<Point2f>vpt2fPoints(5);
    
    for(r = 0; r < vmatPointsMat[1].rows; r++){    
        for(n = 0; n < nPointsOutputLayerChannels/2; n++){
            vpt2fPoints[n] = Point2f(pvvpt2fPoints[n][r * nPointsOutputLayerWidth], pvvpt2fPoints[n + 5][r * nPointsOutputLayerWidth]);
        }
        vvpt2fPoints.push_back(vpt2fPoints);
    }
    
    Rect rectTempRect;
    int m;
    for(n = 0; n < vrectBoundingBox.size(); n++){
        rectTempRect = vrectBoundingBox[n];
        for(m = 0; m < vvpt2fPoints[n].size(); m++){
            vvpt2fPoints[n][m] = Point2f(rectTempRect.x + rectTempRect.width*vvpt2fPoints[n][m].x, rectTempRect.y + rectTempRect.height*vvpt2fPoints[n][m].y);
        }
    }
    
    vector<float>vfTempScores = vfScores;
    vector<Vec4f>vv4fTempConv = vv4fConv;
    vector<vector<Point2f> >vvpt2fTempPoints = vvpt2fPoints;
    vector<Rect>& vrectTempBoundingBox = vrectBoundingBox;
    
    CHECK(vvpt2fPoints.size() == vv4fConv.size() && 
          vvpt2fPoints.size() == vfScores.size() &&
          vv4fConv.size() == vfScores.size());
    
    vfScores.clear();
    vv4fConv.clear();
    vvpt2fPoints.clear();
    
    for(n = 0; n < vfTempScores.size(); n++){
        if(vfTempScores[n] > m_vfThreshold[2]){
            vfScores.push_back(vfTempScores[n]);
            vv4fConv.push_back(vv4fTempConv[n]);
            vvpt2fPoints.push_back(vvpt2fTempPoints[n]);
            vrectBoundingBox.push_back(vrectTempBoundingBox[n]);
        }
    }
}

/*PNet generate bounding box*/
void MTCNN_DETECTOR::GenerateBoundingBox(Blob<float>* blfProb1OutputLayer, Blob<float>* blfConv4_2OutputLayer, vector<Rect>& vrectBoundingBox, vector<float>&vfScores, vector<Vec4f>&vv4fConv4_2, float fScales){
    int nStride = 2;
    int nCellSize = 12;
        
    float* pfConv4_2Begin = blfConv4_2OutputLayer->mutable_cpu_data();
    //const float* pfConv4_2End = blfConv4_2OutputLayer->cpu_data();
    
    float* pfProb1Begin = blfProb1OutputLayer->mutable_cpu_data();
    //const float* pfProb1End = blfProb1OutputLayer->cpu_data();
    
    //vector<float>vfConv4_2Vector(pfConv4_2Begin, pfConv4_2End);
    //vector<float>vfProb1Vector(pfProb1Begin, pfProb1End);
    
    int nProb1OutputLayerNum = blfProb1OutputLayer->num();
    int nProb1OutputLayerChannels = blfProb1OutputLayer->channels();;
    int nProb1OutputLayerWidth = blfProb1OutputLayer->width();
    int nProb1OutputLayerHeight = blfProb1OutputLayer->height();
    
    int nConv4_2OutputLayerNum = blfConv4_2OutputLayer->num();
    int nConv4_2OutputLayerChannels = blfConv4_2OutputLayer->channels();;
    int nConv4_2OutputLayerWidth = blfConv4_2OutputLayer->width();
    int nConv4_2OutputLayerHeight = blfConv4_2OutputLayer->height();
    
    CHECK(nProb1OutputLayerNum == 1 && nConv4_2OutputLayerNum == 1 && 
          nProb1OutputLayerChannels == 2 && nConv4_2OutputLayerChannels == 4);
 
    cout << nProb1OutputLayerNum << endl;
    cout << nProb1OutputLayerChannels << endl;
    cout << nProb1OutputLayerWidth << endl;
    cout << nProb1OutputLayerHeight << endl;
    cout << nConv4_2OutputLayerNum << endl;
    cout << nConv4_2OutputLayerChannels << endl;
    cout << nConv4_2OutputLayerWidth << endl;
    cout << nConv4_2OutputLayerHeight << endl;

    vector<Mat> vmatProb1Mat(nProb1OutputLayerChannels),
                vmatConv4_2Mat(nConv4_2OutputLayerChannels);
        
    int n, m;
    for(n = 0; n < nProb1OutputLayerChannels; n++){
        vmatProb1Mat[n] = Mat(nProb1OutputLayerHeight, nProb1OutputLayerWidth, CV_32FC1, 
                              pfProb1Begin + n*nProb1OutputLayerHeight*nProb1OutputLayerWidth);
        cout << "Prob1Mat n is:" << n << endl;
    }
    
    for(n = 0; n < nConv4_2OutputLayerChannels; n++){
        vmatConv4_2Mat[n] = Mat(nConv4_2OutputLayerHeight, nConv4_2OutputLayerWidth, CV_32FC1,
                                pfConv4_2Begin + n*nConv4_2OutputLayerHeight*nConv4_2OutputLayerWidth);
    }
    
    //cout << "vmatProb1Mat[0].at<float>(0, 0)" << vmatProb1Mat[1].at<float>(10, 10) << endl;
    cout << "vmatProb1Mat[1].data" << endl;
    int r, c;
    float * pfProb1Mat = (float *)(vmatProb1Mat[1].data);
    for(r = 0; r < vmatProb1Mat[1].rows; r++){
        for(c = 0; c < vmatProb1Mat[1].cols; c++){
            if(pfProb1Mat[r * vmatProb1Mat[1].cols + c] > 0.6f)
                cout << pfProb1Mat[r * vmatProb1Mat[1].cols + c] << " ";
        }
        cout << endl;
    }
    
    cout << "m_vfThreshold[0]" << m_vfThreshold[0] << endl;
    Mat matMask = vmatProb1Mat[1] > m_vfThreshold[0];
    if(matMask.type() == CV_8UC1)
        cout << "matMask test pass!" << endl;
    cout << "LocatePoints start!" <<endl;
    vector<Point> vptPoints;
    //vector<Rect> vrectBoundingBox;
    LocatePoints(matMask, vptPoints);
    vrectBoundingBox.resize(vptPoints.size());   
    //vptTempPoints = vptPoints;
    cout << "vptPoints' size is:" <<vptPoints.size() << endl;
    cout << "vmatProb1Mat[1]'s width is:" << vmatProb1Mat[1].cols <<endl;
    cout << "vmatProb1Mat[1]'s height is:" << vmatProb1Mat[1].rows <<endl;
    //exit(-1);
    
    Point ptTL, ptBR;
    float* pmatScore = (float*)(vmatProb1Mat[1].data);
    vector<float *>vpfmatConv4_2Mat(nConv4_2OutputLayerChannels);
    Vec4f v4fConv4_2;
    
    cout << "Conv4_2 out!" << endl;
    for(m = 0; m < vmatConv4_2Mat.size(); m++){
        vpfmatConv4_2Mat[m] = (float*)(vmatConv4_2Mat[m].data);
    }
    cout << "Point out!" << endl;
    cout << fScales << endl;
    for(n = 0; n < vptPoints.size(); n++){
        //cout << "Point test 1!" <<endl;
        ptTL = Point(static_cast<int>((nStride*(vptPoints[n].x - 1)+1)/fScales), 
                             static_cast<int>((nStride*(vptPoints[n].y - 1)+1)/fScales));
        ptBR = Point(static_cast<int>((nStride*(vptPoints[n].x - 1)+nCellSize-1+1)/fScales), 
                             static_cast<int>((nStride*(vptPoints[n].y - 1)+nCellSize-1+1)/fScales));
        //cout << "Point test 2!" <<endl;        
        //vrectBoundingBox[n].x = ptTL.x;
        //vrectBoundingBox[n].y = ptTL.y;
        //vrectBoundingBox[n].width = ptBR.x - ptTL.x;
        //vrectBoundingBox[n].height = ptBR.y - ptTL.y;
        vrectBoundingBox[n] = Rect(ptTL.x, ptTL.y, ptBR.x - ptTL.x, ptBR.y - ptTL.y);
         //cout << "Point test 4!" <<endl;       
        vfScores.push_back(pmatScore[vptPoints[n].y * nProb1OutputLayerWidth + vptPoints[n].x]);
         //cout << "Point test 5!" <<endl;       
        for(m = 0; m < vmatConv4_2Mat.size(); m++){
            v4fConv4_2[m] = vpfmatConv4_2Mat[m][vptPoints[n].y * nProb1OutputLayerWidth + vptPoints[n].x];
        }
        //cout << "Point test 6!" <<endl;
        vv4fConv4_2.push_back(v4fConv4_2);
    }
    cout << "GenerateBoundingBox end!" <<endl;
    //CopyMat(vmatProb1Mat[1], vmatProb1Mat[1], matMask);
    
    //for(n = 0; n < vmatConv4_2Mat.size(); n++){
        //CopyMat(vmatConv4_2Mat[n], vmatConv4_2Mat[n], matMask);
    //}
}

/*RNet generate bounding box*/
void MTCNN_DETECTOR::GenerateBoundingBox(Blob<float>* blfProbOutputLayer, Blob<float>* blfConvOutputLayer, vector<Rect>& vrectBoundingBox, vector<float>&vfScores, vector<Vec4f>&vv4fConv){        
    float* pfConvBegin = blfConvOutputLayer->mutable_cpu_data();
    //const float* pfConv4_2End = blfConv4_2OutputLayer->cpu_data();
    
    float* pfProbBegin = blfProbOutputLayer->mutable_cpu_data();
    //const float* pfProb1End = blfProb1OutputLayer->cpu_data();
    
    //vector<float>vfConv4_2Vector(pfConv4_2Begin, pfConv4_2End);
    //vector<float>vfProb1Vector(pfProb1Begin, pfProb1End);
    
    int nProbOutputLayerNum = blfProbOutputLayer->num();
    int nProbOutputLayerChannels = blfProbOutputLayer->channels();;
    int nProbOutputLayerWidth = blfProbOutputLayer->width();
    int nProbOutputLayerHeight = blfProbOutputLayer->height();
    
    int nConvOutputLayerNum = blfConvOutputLayer->num();
    int nConvOutputLayerChannels = blfConvOutputLayer->channels();;
    int nConvOutputLayerWidth = blfConvOutputLayer->width();
    int nConvOutputLayerHeight = blfConvOutputLayer->height();
    
    CHECK(nProbOutputLayerNum == vrectBoundingBox.size() && nConvOutputLayerNum == vrectBoundingBox.size() && 
          nProbOutputLayerChannels == 2 && nConvOutputLayerChannels == 4 &&
          nProbOutputLayerWidth == 1 && nProbOutputLayerHeight == 1 &&
          nConvOutputLayerHeight == 1 && nConvOutputLayerHeight == 1);
 
    cout << nProbOutputLayerNum << endl;
    cout << nProbOutputLayerChannels << endl;
    cout << nProbOutputLayerWidth << endl;
    cout << nProbOutputLayerHeight << endl;
    cout << nConvOutputLayerNum << endl;
    cout << nConvOutputLayerChannels << endl;
    cout << nConvOutputLayerWidth << endl;
    cout << nConvOutputLayerHeight << endl;

    vector<vector<Mat> > vvmatProbMat(nProbOutputLayerNum),
                         vvmatConvMat(nConvOutputLayerNum);
        
    int n, m;
    cout << "RNet test1!" <<endl;
    vector<Mat> vmatTempProbMat(nProbOutputLayerChannels);
    for(m = 0; m < nProbOutputLayerNum; m++){
        vmatTempProbMat.clear();
        vmatTempProbMat.resize(nProbOutputLayerChannels);
        for(n = 0; n < nProbOutputLayerChannels; n++){
            vmatTempProbMat[n] = Mat(nProbOutputLayerHeight, nProbOutputLayerWidth, CV_32FC1, 
                                     pfProbBegin + (m * nProbOutputLayerChannels + n)*nProbOutputLayerHeight*nProbOutputLayerWidth);
            cout << "pfProbBegin:" << *(pfProbBegin + (m * nProbOutputLayerChannels + n)*nProbOutputLayerHeight*nProbOutputLayerWidth) << endl;
        }
        vvmatProbMat[m] = vmatTempProbMat;
    }
    cout << "RNet test2!" <<endl;
    vector<Mat> vmatTempConvMat(nConvOutputLayerChannels);
    for(m = 0; m < nConvOutputLayerNum; m++){
        vmatTempConvMat.clear();
        vmatTempConvMat.resize(nConvOutputLayerChannels);
        for(n = 0; n < nConvOutputLayerChannels; n++){
            vmatTempConvMat[n] = Mat(nConvOutputLayerHeight, nConvOutputLayerWidth, CV_32FC1,
                                     pfConvBegin + (m * nConvOutputLayerChannels + n)*nConvOutputLayerHeight*nConvOutputLayerWidth);
        }
        vvmatConvMat[m] = vmatTempConvMat;
    }

    
    //vector<float>vfScores;
    //vector<Vec4f>vv4fConv;

    
    int r;
    //float* pfProbMat = (float *)(vmatProbMat[1].data);
    //int nProbMatWidth = vmatProbMat[1].width;
    cout << "RNet test3!" <<endl;
    for(r = 0; r < nProbOutputLayerNum; r++){
        cout << "vvmatProbMat[r]'s size is:" << vvmatProbMat[r].size() << endl;
        vfScores.push_back(*(vvmatProbMat[r][1].ptr<float>(0, 0)));
    }
    
    /*vector<float *>pvv4fConv(nConvOutputLayerChannels);
    for(n = 0; n < nConvOutputLayerChannels; n++){
        pvv4fConv[n] = (float *)(vmatConvMat[n].data);
    }*/
    

    //int nConvMatWidth = vmatConvMat[1].width;
    cout << "RNet test4!" <<endl;
    for(r = 0; r < nConvOutputLayerNum; r++){
        Vec4f v4fTempConv;
        for(n = 0; n < nConvOutputLayerChannels; n++){
            v4fTempConv[n] = *(vvmatConvMat[r][n].ptr<float>(0, 0));
        }
        vv4fConv.push_back(v4fTempConv);
    }
    
    vector<float>vfTempScores = vfScores;
    vector<Vec4f>vv4fTempConv = vv4fConv;
    vector<Rect>& vrectTempBoundingBox = vrectBoundingBox;
    

    CHECK(vfScores.size() == vv4fConv.size() && vv4fConv.size() == vrectBoundingBox.size() && vfScores.size() == vrectBoundingBox.size());
    
    vfScores.clear();
    vv4fConv.clear();
    vrectBoundingBox.clear();
    cout << "RNet test5!" <<endl;
    for(n = 0; n < vfTempScores.size(); n++){
        cout << vfTempScores[n] <<endl;
        if(vfTempScores[n] > m_vfThreshold[1]){
            vfScores.push_back(vfTempScores[n]);
            vv4fConv.push_back(vv4fTempConv[n]);
            vrectBoundingBox.push_back(vrectTempBoundingBox[n]);
        }
    }
    
    cout << "vfScores's size is:" << vfScores.size() << endl;
    cout << "vv4fConv's size is:" << vfScores.size() << endl;
    cout << "vrectBoundingBox's size is:" << vfScores.size() << endl;
    
}

void MTCNN_DETECTOR::LocatePoints(const Mat& matMaskMat, vector<Point>& vptPoints){
    if(matMaskMat.empty()){
        cerr << "LocatePoints input image is empty!" << endl;
    }
    
    int r, c;
    int nMaskWidth = matMaskMat.cols;
    int nMaskHeight = matMaskMat.rows;
    int nTempR;
    uchar* pmatMaskMat = matMaskMat.data;
    for(r = 0; r < matMaskMat.rows; r++){
        nTempR = r * nMaskWidth;
        for(c = 0; c < matMaskMat.cols; c++){
            if(pmatMaskMat[nTempR + c]){
                vptPoints.push_back(Point(c, r));
            }
        }
        
    }
}

void MTCNN_DETECTOR::CopyMat(const Mat& matSrcMat, Mat& matDstMat, Mat& matMask){
    if(matSrcMat.empty()){
        cerr << "CopyMat input image is empty!" << endl;
        exit(-1);
    }
    if(matSrcMat.size() != matMask.size()){
        cerr << "CopyMat size of input image is not same with mask image!" <<endl;
    }
    
    Mat matTempMat = matSrcMat.clone();
    matDstMat = Mat(matTempMat.rows, matTempMat.cols, matTempMat.type(), Scalar(0.0, 0.0, 0.0));
    
    int r, c;
    
    float* pmatTempMat = (float *)(matTempMat.data);
    float* pmatDstMat = (float *)(matDstMat.data);
    uchar* pmatMask = matMask.data;
    int nmatWidth = matTempMat.cols, nmatHeight = matTempMat.rows;
    
    int nTempR;
    for(r = 0; r < matTempMat.rows; r++){
        nTempR = r * nmatWidth;
        for(c = 0; c < matTempMat.cols; c++){
            if(pmatMask > 0){
                pmatDstMat[nTempR + c] = pmatTempMat[nTempR + c];
            }
        }
    }
}

bool CompareScore(const sortScore& fsScore1, const sortScore& fsScore2){
    return fsScore1.fScore > fsScore2.fScore;
}

void MTCNN_DETECTOR::Nms(const vector<Rect>& vrectBoundingBox, const vector<float>& vfScores, vector<int>& viPick, float fThreshold, string strType){
    if(!vrectBoundingBox.size()){
        viPick.clear();
        cout << "Nms return!" << endl;
        return;
    }
    cout << "Nms 1!" << endl;
    vector<sortScore>vssScore;
    vector<sortScore>vssTrimScore;
    vector<sortScore>vssTempTrimScore;
    InitializeScore(vfScores, vssScore);
    sort(vssScore.begin(), vssScore.end(), CompareScore);
    cout << "Nms 2!" << endl;    
    int nScoreNo;
    Rect rectTemp;
    vector<Rect>vrectTempBoundingBox = vrectBoundingBox;
    vector<Rect>vrectTrimBoundingBox;
    int n, m;
    Rect rectTempBoundingBox;
    int nTempArea;
    Point ptTlPoint, ptBrPoint;
    float fRatio;
    cout << "Nms 3!" << endl;
    cout << "Bounding Box's size is:" << vrectBoundingBox.size() << endl;

    m = 0;
    vssTrimScore = vssScore;
    while(vssTrimScore.size()){
        cout << "vvsScore size is:" << vssTrimScore.size() <<endl;
        double dCurrentTime = static_cast<double>(getTickCount());
        if(vrectTrimBoundingBox.size()){
            vrectTrimBoundingBox.clear();
        }
        if(vssTempTrimScore.size()){
            vssTempTrimScore.clear();
        }
        nScoreNo = vssScore[m].nNo;
        cout << "vssScore[0].fScore" << vssScore[0].fScore << endl;
        viPick.push_back(nScoreNo);
        rectTemp = vrectBoundingBox[nScoreNo];
        nTempArea = rectTemp.area();
        //cout << "Nms 4!" << endl;

        for(n=vrectTempBoundingBox.size() - 1; n >= 0; n--){
            //double dCurrentTime = static_cast<double>(getTickCount());
            //cout << "vrectTempBoundingBox size is:" << vrectTempBoundingBox.size() <<endl;
            ptTlPoint = vrectTempBoundingBox[n].tl();
            ptBrPoint = vrectTempBoundingBox[n].br();
            if(n != nScoreNo){
                rectTempBoundingBox.x = max(rectTemp.x, ptTlPoint.x);
                rectTempBoundingBox.y = max(rectTemp.y, ptTlPoint.y);
                rectTempBoundingBox.width = min(rectTemp.x + rectTemp.width, ptBrPoint.x) -
                                            rectTempBoundingBox.x;
                rectTempBoundingBox.height = min(rectTemp.y + rectTemp.height, ptBrPoint.y) -
                                             rectTempBoundingBox.y;
                //cout << "rectTempBoundingBox" << rectTempBoundingBox.x << " " << rectTempBoundingBox.y << " " << rectTempBoundingBox.width << " " << rectTempBoundingBox.height << endl;
                                             
                rectTempBoundingBox.width = rectTempBoundingBox.width>0?rectTempBoundingBox.width:0;
                rectTempBoundingBox.height = rectTempBoundingBox.height>0?rectTempBoundingBox.height:0;
                
                //cout << "rectTempBoundingBox" << rectTempBoundingBox.x << " " << rectTempBoundingBox.y << " " << rectTempBoundingBox.width << " " << rectTempBoundingBox.height << endl;
                //cout << "Nms 4!" << endl;
                //vrectTrimBoundingBox.push_back(rectTempBoundingBox);
                if(strType == "Min"){
                    fRatio = rectTempBoundingBox.area() / min(rectTempBoundingBox.area(), vrectTempBoundingBox[n].area());
                }else{
                    //cout << "Bounding Box area is:" << rectTempBoundingBox.area() << endl;
                    //cout << "Bounding Box n area is:" << vrectTempBoundingBox[n].area() << endl;
                    //cout << "Bounding Box n area is:" << nTempArea << endl;
                    fRatio = rectTempBoundingBox.area() / (nTempArea + vrectTempBoundingBox[n].area() - rectTempBoundingBox.area());
                }
                //cout << "Nms 5!" << endl;
                //cout << "fRatio" << fRatio << endl;
                if(fRatio <= fThreshold){
                    //for(int m=0; m < vssScore.size(); m++){
                        //if(n == vssScore[m].nNo){
                            //vssScore.erase(vssScore.begin() + m);
                        //}
                    //}
                    //vrectTempBoundingBox.erase(vrectTempBoundingBox.begin() + n);
                    vrectTrimBoundingBox.push_back(vrectTempBoundingBox[n]);
                    for(int m=0; m < vssTrimScore.size(); m++){
                        if(n == vssTrimScore[m].nNo){
                            vssTempTrimScore.push_back(vssTrimScore[m]);
                        }
                    }
                    
                }

                
            }
            //else{
                //vssScore.erase(vssScore.begin());
                //vrectTempBoundingBox.erase(vrectTempBoundingBox.begin() + n);
            //}
            //cout << "n" << n << endl;
            //double dProcessTime = (static_cast<double>(getTickCount())-dCurrentTime) / getTickFrequency();
            //cout << dProcessTime <<endl;
        }
        vrectTempBoundingBox = vrectTrimBoundingBox;
        vssTrimScore = vssTempTrimScore;
        double dProcessTime = (static_cast<double>(getTickCount())-dCurrentTime) / getTickFrequency();
        cout << dProcessTime <<endl;
        m++;
    }
    cout << "Nms 6!" << endl;
}

void MTCNN_DETECTOR::InitializeScore(const vector<float>& vfScores, vector<sortScore>& vssScore){
    if(!vfScores.size()){
        cerr << "InitializeScore input score is empty!" << endl;
        exit(-1);
    }
    
    if(vssScore.size()){
        vssScore.clear();
    }
    int n;
    sortScore ssTempScore;
    for(n = 0; n < vfScores.size(); n++){
        ssTempScore.fScore = vfScores[n];
        ssTempScore.nNo = n;
        vssScore.push_back(ssTempScore);
    }
    
}

vector<vector<Point2f> > MTCNN_DETECTOR::GetFacePoints(){
    cout << "The FacePoints' size is:" << vvpt2fFacePoints.size() <<endl;
    return vvpt2fFacePoints;
}
vector<Rect> MTCNN_DETECTOR::GetFaceBoundingBox(){
    cout << "The FaceBoundingBox's is:" << vrectFaceBoundingBox.size() <<endl;
    return vrectFaceBoundingBox;
}
