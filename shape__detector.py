import cv2
import numpy as np
MIN_MATCH_COUNT=10

detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg_stop=cv2.imread("train_image/stop_sign_template_small_01.png",0)
trainKP_stop,trainDesc_stop=detector.detectAndCompute(trainImg_stop,None)

trainImg_limit5=cv2.imread("train_image/sign_speed-5.jpg",0)
trainKP_limit5,trainDesc_limit5=detector.detectAndCompute(trainImg_limit5,None)

cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)

    matches_stop=flann.knnMatch(queryDesc,trainDesc_stop,k=2)

    matches_limit5=flann.knnMatch(queryDesc,trainDesc_limit5,k=2)

    goodMatch_stop=[]
    for m,n in matches_stop:
        if(m.distance<0.45*n.distance):
            goodMatch_stop.append(m)
    if(len(goodMatch_stop)>MIN_MATCH_COUNT):
        tp_stop=[]
        qp_stop=[]
        for m in goodMatch_stop:
            tp_stop.append(trainKP_stop[m.trainIdx].pt)
            qp_stop.append(queryKP[m.queryIdx].pt)
        tp_stop,qp_stop=np.float32((tp_stop,qp_stop))
        H_stop,status=cv2.findHomography(tp_stop,qp_stop,cv2.RANSAC,3.0)
        h_stop,w_stop=trainImg_stop.shape
        trainBorder_stop=np.float32([[[0,0],[0,h_stop-1],[w_stop-1,h_stop-1],[w_stop-1,0]]])
        queryBorder_stop=cv2.perspectiveTransform(trainBorder_stop,H_stop)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder_stop)],True,(0,255,0),5)
        print ("found a stop sign")

    goodMatch_limit5=[]
    for m,n in matches_limit5:
        if(m.distance<0.45*n.distance):
            goodMatch_limit5.append(m)
    if(len(goodMatch_limit5)>MIN_MATCH_COUNT):
        tp_limit5=[]
        qp_limit5=[]
        for m in goodMatch_limit5:
            tp_limit5.append(trainKP_limit5[m.trainIdx].pt)
            qp_limit5.append(queryKP[m.queryIdx].pt)
        tp_limit5,qp_limit5=np.float32((tp_limit5,qp_limit5))
        H_limit5,status=cv2.findHomography(tp_limit5,qp_limit5,cv2.RANSAC,3.0)
        h_limit5,w_limit5=trainImg_limit5.shape
        trainBorder_limit5=np.float32([[[0,0],[0,h_limit5-1],[w_limit5-1,h_limit5-1],[w_limit5-1,0]]])
        queryBorder_limit5=cv2.perspectiveTransform(trainBorder_limit5,H_limit5)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder_limit5)],True,(255,0,0),5)
        print ("found a 5 speed limit")

    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

#def detectorsss():
