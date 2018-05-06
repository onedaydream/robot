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

trainImg_limit50=cv2.imread("train_image/sign_speed-50.jpg",0)
trainKP_limit50,trainDesc_limit50=detector.detectAndCompute(trainImg_limit50,None)

cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)

    matches_stop=flann.knnMatch(queryDesc,trainDesc_stop,k=2)

    matches_limit5=flann.knnMatch(queryDesc,trainDesc_limit5,k=2)

    matches_limit50=flann.knnMatch(queryDesc,trainDesc_limit50,k=2)

    goodMatch_stop=[]
    for m,n in matches_stop:
        if(m.distance<0.65*n.distance):
            goodMatch_stop.append(m)
    if(len(goodMatch_stop)>MIN_MATCH_COUNT):
        print ("found a stop sign")

    goodMatch_limit5=[]
    for m,n in matches_limit5:
        if(m.distance<0.45*n.distance):
            goodMatch_limit5.append(m)
    if(len(goodMatch_limit5)>MIN_MATCH_COUNT):
        print ("found a 5 speed limit")

    goodMatch_limit50=[]
    for m,n in matches_limit50:
        if(m.distance<0.45*n.distance):
            goodMatch_limit50.append(m)
    if(len(goodMatch_limit50)>MIN_MATCH_COUNT):
        print ("found a 55 speed limit")

    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

#def detectorsss():
