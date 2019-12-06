import bpy
import numpy as np
from math import *
from random import random
from random import randint
phi = (1+sqrt(5))/2
dhangle = pi-acos(-sqrt(5)/3)
rotsideold = {1:'-+', 2:'u+-', 3:'u-', 4:'-+-+', 5:'--', 6:'u+-+-', 7:'-', 8:'+', 9:'u+',10:'++',
        11:'--+', 12:'+-+', 13:'u+-+', 14:'++-+', 15:'', 16:'++-', 17:'u', 18:'-+-', 19:'+-', 20:'u-+'}
rotside = {2:'-+', 3:'u+-', 1:'u-', 14:'-+-+', 10:'--', 6:'u+-+-', 12:'-', 5:'+', 17:'u+', 13:'++',
           8:'--+', 4:'+-+', 16:'u+-+', 9:'++-+', 15:'', 11:'++-', 7:'u', 20:'-+-', 18:'+-',19:'u-+'}


def randpos(obj):
    #z=0.278686
    #z=0.312754
    loc = (-(random())*4.8,(random())*4.8, 0.312754)
    obj.location.xyz = loc
    return loc

def randcampos():
    obj = bpy.data.objects['Camera.001']
    loc = ((2*random()-1)*10,(2*random()-1)*10,(random())*20)
    obj.location.xyz = loc
    return loc

def plusminus(coors):
    res = []
    if len(coors)==0:
        return [[]]
    else:
        for coor in plusminus(coors[1:]):
            res.append([coors[0]]+coor)
            if coors[0]!=0:
                res.append([-coors[0]]+coor)
        return res

def dot(u1,u2):
    return sum([u1[k]*u2[k] for k in range(len(u1))])

def norm(l1,l2):
    return sqrt(sum([(l1[k]-l2[k])*(l1[k]-l2[k]) for k in range(len(l1))]))

def rot0():
    tripts=[[0,1,phi],[0,-1,phi],[phi,0,1]]
    ctop = tripts[2]
    ctri = [sum([tripts[j][k] for j in range(3)])/3 for k in range(3)]
    return acos(dot(ctop,ctri)/sqrt(dot(ctop,ctop)*dot(ctri,ctri)))


def rotstring(obj,r0,s):
    obj.select = True
    obj.rotation_euler=r0
    for c in s:
        if c=='+':
            obj.rotation_euler.z += 2*pi/3
        elif c=='-':
            obj.rotation_euler.z -= 2*pi/3
        if c=='+' or c=='-' or c=='u':
            bpy.ops.transform.rotate(value=dhangle,axis=(0.0,1.0,0.0))
            obj.rotation_euler.z += pi
    obj.select = False

def randrot(obj,r0):
    side = randint(1,20)
    rotstring(obj,r0,rotside[side])
    obj.rotation_euler.z = random()*2*pi
    return side

def randcamrot():
    obj = bpy.data.objects['Camera.001']
    x = radians(randint(0,90))
    z = radians(randint(0,360))
    obj.rotation_euler.x = x
    obj.rotation_euler.z = z
    return (x,z)

def roll(n,folder,dice):
    for obj in bpy.context.selected_objects:
        obj.select = False
    objs = [bpy.data.objects['Icosphere.001'],bpy.data.objects['Icosphere'],bpy.data.objects['Icosphere.002']][:dice]
    r0 = (0,0.65235,0) #,(3.341, 0.309, 1.628)
    targets = {}
    sides = [-1 for _ in range(dice)]
    locs = [-1 for _ in range(dice)]
    i = 0
    K = np.array([[2100,0,1920/2],[0,2100,1080/2],[0,0,1]])
    corners = plusminus((5.56,5.56,0))
    while i<n:
        (camx,camz)=randcamrot()
        camPos = randcampos()
        rotMat = xzRotMat(camx,camz)
        M = extMat(rotMat,camPos)        
        for k in range(dice):
            sides[k] = randrot(objs[k],r0)
            locs[k] = randpos(objs[k])
        collide = False
        for l1 in locs:
            for l2 in locs:
                if l1!=l2 and norm(l1,l2)<2:
                    collide = True
        if collide:
            continue
        outFrame = False
        for loc in locs:
            if not check(M,K,loc,150):
                outFrame = True
        if outFrame:
            continue
        hasCorner = False
        for corner in corners:
            if check(M,K,corner,0):
                hasCorner = True
        if not hasCorner:
            continue
        bpy.data.scenes['Scene'].render.filepath = 'C:\\Users\\joshm\\Documents\\bbMLg\\' + folder + '\\' + str(dice) + 'd' + str(i) + '.png'
        bpy.ops.render.render(write_still=True)
        targets[i] = sides[:]
        i+=1
    no = 'C:\\Users\\joshm\\Documents\\bbMLg\\' + folder + '\\' + 'data' + str(dice) + '.txt'
    fo = open(no,'w+')
    for key in targets:
        fo.write(str(key) + ',' + ','.join([str(side) for side in targets[key]])+'\n')
    fo.flush()
    fo.close()

def icopoints():
    points = [[1,1,1],[0,phi,phi-1],[phi-1,0,phi],[phi,phi-1,0]]
    pts = []
    for point in points:
        pts+=plusminus(point)


def extMat(rotMat,camPos):
    posMat = np.array(camPos).reshape((3,1))
    Hwc = np.concatenate((rotMat,posMat),1)
    Hwc = np.concatenate((Hwc,np.array((0,0,0,1)).reshape(1,4)),0)
    Hcw = np.linalg.inv(Hwc)
    return Hcw[:3]

def xzRotMat(x,z):
    c1=cos(z)
    s1=sin(z)
    c3=cos(x)
    s3=sin(x)
    return -np.array([[-c1,-c3*s1,s1*s3],[-s1,c1*c3,-c1*s3],[0,s3,c3]])

def transform(M,K,point):
    P = np.concatenate((np.array(point).reshape((3,1)),np.array((1)).reshape((1,1))),0)
    Pw=K@(M@P)
    return (Pw[0][0]/Pw[2][0],Pw[1][0]/Pw[2][0])

def check(M,K,point,buf):
    Pi = transform(M,K,point)
    return Pi[0]>buf and Pi[0]<1920-buf and Pi[1]>buf and Pi[1]<1080-buf

roll(4000,'data1120a',1)
roll(4000,'data1120a',2)
roll(4000,'data1120a',3)
