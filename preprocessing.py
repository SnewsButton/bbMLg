import cv2 as cv
import numpy as np
import math

def hough(img,R,thresh):
	gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
	edges = cv.Canny(gray,50,150,apertureSize = 3)

	lines = cv.HoughLines(edges,1,np.pi/180,thresh)
	print(lines.shape[0])
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
	return imgLines

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
    return (min(contour[:,0,0]),max(contour[:,0,0]),min(contour[:,0,1]),max(contour[:,0,1]))

def getBlobs(img,show):
    imgsmall = img
    while imgsmall.shape[0]*imgsmall.shape[1]>1000000:
        imgsmall = cv.resize(imgsmall,(0,0),fx=.5,fy=.5)
    imgHSV = cv.cvtColor(imgsmall,cv.COLOR_BGR2HSV)
    imgSat = imgHSV[:,:,1]
    ret1,imgBW = cv.threshold(imgSat,127,255,cv.THRESH_BINARY) #Can adjust second value
    strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7)) #Can adjust morph size
    imgBWM = cv.morphologyEx(imgBW, cv.MORPH_CLOSE, strel)
    if show:
        cv.imshow('Image',imgBWM) ################
    contours0, hierarchy0 = cv.findContours(imgBW, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours, hierarchy = cv.findContours(imgBWM, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    blobImgs = []
    hues = []
    for k in range(len(contours)):
        if len(contours[k])>20 and hierarchy[0][k][3]<0 and 0.75 < getContourCirc(contours[k]) < 0.9: #Can adjust threshold
            bbox = getBoundingBox(contours[k])
            blobImg = imgsmall[bbox[2]:bbox[3],bbox[0]:bbox[1]]
            blobHSV = cv.cvtColor(blobImg,cv.COLOR_BGR2HSV)
            hues.append(np.mean(blobHSV[:,:,0]))
            blobImgs.append(blobImg)            
    blobs = np.array(blobImgs)
    return blobs[np.argsort(hues)]
    
for k in range(20):
    img=cv.imread('C:\\Users\\joshm\\Documents\\bbMLg\\data1120a\\3d' + str(k) + '.png')
    blobs = getBlobs(img,False)
    print(str(k)+'\t'+'\t'.join([str(blob.shape[0])+' '+str(blob.shape[1]) for blob in blobs]))
