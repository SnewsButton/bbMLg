import cv2 as cv
import numpy as np
import math
import os

def hough(img,step):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,150,apertureSize = 3)
    for thresh in range(step,300,step): 
            lines = cv.HoughLines(edges,1,np.pi/180,thresh)
            if lines is None:
                return []
            if len(lines)<=4:
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
    top = max(int(min(contour[:,0,1])-(max(contour[:,0,1])-min(contour[:,0,1]))*0.2),0)
    return (int(min(contour[:,0,0])),max(contour[:,0,0]),top,max(contour[:,0,1]))

def getBlobs(img,params,show):
    imgsmall = img
    while imgsmall.shape[0]*imgsmall.shape[1]>1000000:
        imgsmall = cv.resize(imgsmall,(0,0),fx=.5,fy=.5)
    imgHSV = cv.cvtColor(imgsmall,cv.COLOR_BGR2HSV)
    imgHue = imgHSV[:,:,0]
    imgSat = imgHSV[:,:,1]
    imgLit = (imgHSV[:,:,2]*(1-imgSat/511)).astype(np.uint8)
    blobImgs = []
    hues = []
    imgCon = np.copy(imgsmall)
    strel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9, 9)) #Can adjust morph size
    for minHue in [-2]+[k for k in range(0,180,params['step'])]:
            if minHue==-2:
                imgLitM = cv.morphologyEx(255-imgLit, cv.MORPH_CLOSE, strel)
                ret1,_ = cv.threshold(imgLitM,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                _,imgBWM = cv.threshold(imgLitM,ret1*params['black'],255,cv.THRESH_BINARY)
            elif minHue==-1:
                imgLitM = cv.morphologyEx(imgLit, cv.MORPH_CLOSE, strel)
                ret1,_ = cv.threshold(imgLitM,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
                _,imgBWM = cv.threshold(imgLitM,ret1*params['white'],255,cv.THRESH_BINARY)
            else:
                imgColor = imgSat*np.logical_and(imgHue>=minHue,imgHue<minHue+params['step'])
                imgM = cv.morphologyEx(imgColor, cv.MORPH_CLOSE, strel)
                ret1,_ = cv.threshold(imgM,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU) 
                _,imgBWM = cv.threshold(imgM,ret1*params['a']+params['b'],255,cv.THRESH_BINARY)  
            #contours0, hierarchy0 = cv.findContours(imgBW, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            contours, hierarchy = cv.findContours(imgBWM, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            for k in range(len(contours)):
                if 600<cv.contourArea(contours[k])<imgBWM.shape[0]*imgBWM.shape[1]/2 and hierarchy[0][k][3]<0 and params['circ_low'] < getContourCirc(contours[k]) < params['circ_high']: #Can adjust threshold
                    #print(hierarchy[0][k])
                    cv.drawContours(imgCon, contours, k, (255,0,0), 2)
                    bbox = getBoundingBox(contours[k])
                    blobImg = resizeBlob(imgsmall[bbox[2]:bbox[3],bbox[0]:bbox[1]])
                    blobHSV = cv.cvtColor(blobImg,cv.COLOR_BGR2HSV)
                    if minHue<0:
                        hues.append(minHue)
                    else:
                        hues.append(np.mean(blobHSV[:,:,0]))
                    blobImgs.append({'Image':blobImg,'Center':(bbox[3],int((bbox[0]+bbox[1])/2))})
    if show:
        cv.imshow('Image',imgCon)
    try:
        blobs = np.array(blobImgs)
    except ValueError:
        return []
    blobs = blobs[np.argsort(hues)]
    return blobs

def resizeBlob(blob):
    f = min(60/blob.shape[0],50/blob.shape[1])
    blobRs = cv.resize(blob,(0,0),fx=f,fy=f)
    pads = ((60-blobRs.shape[0])/2,(50-blobRs.shape[1])/2)
    pad0 = (math.floor(pads[0]),math.ceil(pads[0]))
    pad1 = (math.floor(pads[1]),math.ceil(pads[1]))
    blobSq = np.pad(blobRs,(pad0,pad1,(0,0)),'edge')
    return blobSq

def montage(blobs):
    res = np.zeros((60,5,3)).astype(np.uint8)
    for blob in blobs:
        if blob['Image'].shape!=(60,50,3):
            return
        res = np.concatenate((res,blob['Image'],np.zeros((60,5,3)).astype(np.uint8)),axis=1)
    cv.imshow('Image',res)
    cv.waitKey(0)

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

params = {'step':30,'a':0.50,'b':76,'black':1.29,'white':1.60,'circ_low':0.69,'circ_high':0.90}

def testImages(D,N):
    for k in range(N):
        filename = str(D) + 'd' + str(k) + '.png'
        if os.name == 'nt':
            filename = 'C:\\Users\\joshm\\Documents\\bbMLg\\bbMLG\\data1120b\\' + filename
        else:
            filename = './data1120b/' + filename
        img=cv.imread(filename)
        blobs = getBlobs(img,params,False)
        montage(blobs)
        print(str(k)+'\t'+'\t'.join([str(blob['Center']) for blob in blobs]))

N = 40
if os.name == 'nt':
    folder = 'C:\\Users\\joshm\\Documents\\bbMLg\\bbMLG\\data1120b\\'
else:
    folder = './data1120b/'
alldata = np.array([])
allblobs = np.array([])
alllines = np.array([])
for D in range(1,4):
	data = np.genfromtxt(folder + 'data' + str(D) + '.txt', delimiter=',')[:N]
	names = [folder + str(D) + 'd' + str(k) + '.png' for k in range(N)]
	imgs = [cv.imread(name) for name in names]
	mat = np.array([getBlobs(img,params,False) for img in imgs])
	lines = np.array([hough(img,10) for img in imgs])
	mask = [mat[i].shape[0]==D for i in range(mat.shape[0])]
	Nr = sum(mask)
	dataf = data[mask,1:]
	matf = mat[mask]
	linesf = lines[mask]
	datar = np.reshape(dataf,(Nr*D,))
	matr = np.concatenate(matf)
	linesr = np.reshape(np.stack((linesf,)*D, axis=-1),(Nr*D,))
	alldata = np.concatenate((alldata,datar))
	allblobs = np.concatenate((allblobs,matr))
	alllines = np.concatenate((alllines,linesr))
