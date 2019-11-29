import cv2 as cv
import numpy as np
import math

def hough(img,R,thresh):
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	edges = cv.Canny(gray,50,150,apertureSize = 3)

	lines = cv.HoughLines(edges,1,np.pi/180,thresh)
	imgLines = np.copy(img)
	for k in range(lines.shape[0]):
	    rho = lines[k][0][0]
	    theta = lines[k][0][1]
	    a = np.cos(theta)
	    b = np.sin(theta)
	    x0 = a*rho
	    y0 = b*rho
	    x2 = int(x0 - R*(-b))
	    y2 = int(y0 - R*(a))
	    imgLines = cv.line(imgLines,(x0,y0),(x2,y2),(0,0,255),2)
	return lines

def drawContour(img,contours,k):
    imgCon = np.copy(img)
    cv.drawContours(imgCon, contours, k, (0,255,255))
    cv.imshow('Image',imgCon)

def numChildren(hierarchy,k):
    count = 0
    for hier in hierarchy[0]:
        if hier[3]==k:
            count+=1
    return count

def getContourCenter(contour):
    mu = cv.moments(contour)
    if mu['m00']!=0:
        c = (mu['m10']/mu['m00'], mu['m01']/mu['m00']) 
    else:
        c = (0,0)    

def getContourCirc(contour):
    P = cv.arcLength(contour,True);
    if P==0:
        return 0
    A = cv.contourArea(contour);
    return 4*math.pi*A/(P*P)

def getBoundingBox(contour):
    top = int(min(contour[:,0,0])-(max(contour[:,0,0])-min(contour[:,0,0]))*0.2)
    return (top,max(contour[:,0,0]),min(contour[:,0,1]),max(contour[:,0,1]))

def getBlobs(img,step,show,a,b):
    imgsmall = img
    while imgsmall.shape[0]*imgsmall.shape[1]>1000000:
        imgsmall = cv.resize(imgsmall,(0,0),fx=.5,fy=.5)
    imgHSV = cv.cvtColor(imgsmall,cv.COLOR_BGR2HSV)
    imgHue = imgHSV[:,:,0]
    imgSat = imgHSV[:,:,1]
    imgLit = (imgHSV[:,:,2]*(1-imgSat/511)).astype(np.uint8)
    blobImgs = []
    hues = []
    step = 30
    imgCon = np.copy(imgsmall)
    strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)) #Can adjust morph size
    for minHue in [-2,-1]+[k for k in range(0,180,step)]:
            if minHue==-2:
                imgLitM = cv.morphologyEx(255-imgLit, cv.MORPH_CLOSE, strel)
                ret1,_ = cv.threshold(imgLitM,55,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                _,imgBWM = cv.threshold(imgLitM,ret1*1.29,255,cv.THRESH_BINARY)
            elif minHue==-1:
                imgLitM = cv.morphologyEx(imgLit, cv.MORPH_CLOSE, strel)
                ret1,_ = cv.threshold(imgLitM,170,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                _,imgBWM = cv.threshold(imgLitM,ret1*1.6,255,cv.THRESH_BINARY)
            else:
                imgColor = imgSat*np.logical_and(imgHue>=minHue,imgHue<minHue+step)
                imgM = cv.morphologyEx(imgColor, cv.MORPH_CLOSE, strel)
                ret1,_ = cv.threshold(imgM,100,255,cv.THRESH_BINARY+cv.THRESH_OTSU) 
                _,imgBWM = cv.threshold(imgM,ret1*a+b,255,cv.THRESH_BINARY)  
            #contours0, hierarchy0 = cv.findContours(imgBW, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            contours, hierarchy = cv.findContours(imgBWM, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
            for k in range(len(contours)):
                if cv.contourArea(contours[k])>600 and hierarchy[0][k][3]<0 and 0.69 < getContourCirc(contours[k]) < 0.9: #Can adjust threshold
                    cv.drawContours(imgCon, contours, k, (255,0,0), 2)
                    bbox = getBoundingBox(contours[k])
                    blobImg = imgsmall[bbox[2]:bbox[3],bbox[0]:bbox[1]]
                    blobHSV = cv.cvtColor(blobImg,cv.COLOR_BGR2HSV)
                    if minHue<0:
                        hues.append(minHue)
                    else:
                        hues.append(np.mean(blobHSV[:,:,0]))
                    blobImgs.append(blobImg)
    if show:
        cv.imshow('Image',imgCon)
    try:
        blobs = np.array(blobImgs)
    except ValueError:
        return []
    blobs = blobs[np.argsort(hues)]
    return resizeBlobs(blobs)

def resizeBlobs(blobs):
    blobsSq = []
    for blob in blobs:
        f = min(50/blob.shape[0],60/blob.shape[1])
        blobRs = cv.resize(blob,(0,0),fx=f,fy=f)
        pads = ((50-blobRs.shape[0])/2,(60-blobRs.shape[1])/2)
        pad0 = (math.floor(pads[0]),math.ceil(pads[0]))
        pad1 = (math.floor(pads[1]),math.ceil(pads[1]))
        blobSq = np.pad(blobRs,(pad0,pad1,(0,0)),'edge')
        blobsSq.append(blobSq)
    return blobsSq

def montage(blobs):
    res = np.zeros((50,5,3)).astype(np.uint8)
    for blob in blobs:
        if blob.shape!=(50,60,3):
            return
        res = np.concatenate((res,blob,np.zeros((50,5,3)).astype(np.uint8)),axis=1)
    cv.imshow('Image',res)

def testLinearOtsu():
    for a in range(50,80,2):
        print('\t',a)
        for b in range(50,80,2):
            fail = False
            for k in range(20):
                img=cv.imread('C:\\Users\\joshm\\Documents\\bbMLg\\data1120a\\3d' + str(k) + '.png')
                blobs = getBlobs(img,30,False,a/100,b)
                if len(blobs)!=3:
                    fail=True
            img = cv.imread('C:\\Users\\joshm\\Documents\\bbMLg\\alldicephoto.jpg')
            blobs = getBlobs(img,30,False,a/100,b)
            if len(blobs)!=6:
                fail=True
            if not fail:
                print(a/100,b)
                
#for k in range(20):
#    img=cv.imread('C:\\Users\\joshm\\Documents\\bbMLg\\data1120a\\3d' + str(k) + '.png')
#    blobs = getBlobs(img,30,False,0.50,76) #Finely tuned parameters
#    print(str(k)+'\t'+'\t'.join([str(blob.shape[0])+' '+str(blob.shape[1]) for blob in blobs]))

    
#lines = hough(img,1000,100)
