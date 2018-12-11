# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 11:09:45 2018

@author: Ian, Bram
@version: 0.1

this coee uses 2 core features. 
1 is the diffrence list from the system. this list contains the all the size
diffrences between the coins. Secondly is a collour finder with selected 
wich will set a boundry between which values of the diffrences its going to look.
Wich means that it atleast 2 coins of diffrent value to work in the same picture. 
This is more to let the system work with variable distances. this will then 
select the largest coin in the picture and use that one as a referance of 
between the other coins. 
"""

#from __future__ import division
import cv2
import numpy as np
import copy
def nothing(*arg):
        pass

def get_circles(image, maskrange, kernel,minRadius,maxRadius):
    colorLow = np.array([ maskrange[0], maskrange[1], maskrange[2]])
    colorHigh = np.array([ maskrange[3], maskrange[4], maskrange[5]])
    mask = cv2.inRange(image, colorLow, colorHigh)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    threshold = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 5)
    circles = cv2.HoughCircles(threshold, cv2.HOUGH_GRADIENT, 2, 80,
              param1=45,
              param2=50,
              minRadius=minRadius,
              maxRadius=maxRadius)
    circles = np.uint16(np.around(circles))
    return circles
# CV parameters
font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
power = 1
resize = 0.25
# coin list
coins = [
        ["Copper 1 cent",0.01],
        ["Copper 2 cent",0.02],
        ["Copper 5 cent",0.05],
        ["Messing 10 cent",0.10],
        ["Messing 20 cent",0.20],
        ["Messing 50 cent",0.50],
        ["Silver 1 Euro",1],
        ["Silver 2 Euro",2]
        ]
# size diffrences
diffrence = np.array([
    [1.000,	1.170,	1.307,	1.227,	1.386,	1.500,	1.432,	1.500],
    [0.854, 1.000,	1.117,	1.049,	1.184,	1.282,	1.223,	1.282],
    [0.765,	0.896,	1.000,	0.939,	1.061,	1.148,	1.096,	1.148],
    [0.815,	0.954,	1.065,	1.000,	1.130,	1.222,	1.167,	1.222],
    [0.721,	0.844,	0.943,	0.885,	1.000,	1.082,	1.033,	1.082],
    [0.667,	0.780,	0.871,	0.818,	0.924,	1.000,	0.955,	1.000],
    [0.698,	0.817,	0.913,	0.857,	0.968,	1.048,	1.000,	1.048],
    [0.667,	0.780,	0.871,	0.818,	0.924,	1.000,	0.955,	1.000]
    ])
#increasing the power to enlarge the saclings diffrence. works best with 2 euro coins
for x in range(0,len(diffrence)):
        for y in range(0,len(diffrence[0])):
            diffrence[x][y]=diffrence[x][y]**power

#colour spaces that the system works in
icol = []
icolC = (8, 41, 0, 19, 255, 255)  #Copper
icolY = (20, 31, 0, 35, 255, 255) # Yellow
icolS = (20, 28, 0, 31, 70, 255)    # Silver
icol.append([icolC,icolY,icolS])
icolC = (8, 41, 0, 19, 255, 255)  #Copper
icolY = (22, 31, 0, 35, 255, 255) # Yellow
icolS = (20, 28, 0, 31, 65, 255)    # Silver
icol.append([icolC,icolY,icolS])

cv2.namedWindow('colorTest')

#list of images

images = [
          ["Coin2",".jpg",0],
          ["Coin3",".jpg",0],
          ["Coin4",".jpg",0],
          ["Coin5",".jpg",0],
          ["Coin6",".jpg",0],
          ["Coin7",".jpg",1]]


#frame = cv2.resize(frame, (0,0), fx=1, fy=1)
 
for pic in range(0,len(images)):
    frame = cv2.imread((images[pic][0]+images[pic][1]))

    framecopy = copy.copy(frame)

    frameBGR = cv2.GaussianBlur(frame, (7, 7), 0)
    frameBGR = cv2.medianBlur(frameBGR, 15)

    
    hsv = cv2.cvtColor(frameBGR, cv2.COLOR_BGR2HSV)

    colorLow = np.array([icolC[0],icolC[1],icolC[2]])
    colorHigh = np.array([icolC[3],icolC[4],icolC[5]])
    mask = cv2.inRange(hsv, colorLow, colorHigh)
    colorLow = np.array([icolY[0],icolY[1],icolY[2]])
    colorHigh = np.array([icolY[3],icolY[4],icolY[5]])
    mask2 = cv2.inRange(hsv, colorLow, colorHigh)
 
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    circles1 = get_circles(hsv, icol[images[pic][2]][0], kernal,80, 140)
    circles2 = get_circles(hsv, icol[images[pic][2]][1], kernal,100, 140)
    circles3 = get_circles(hsv, icol[images[pic][2]][2], kernal,100, 140)

    data = []
    error = 25
    #extract all copper coins out the list
    for c in circles1[0,:]:
        data.append( [c[2],'C', 1.0, c[0], c[1]])
    #extract all messing coins out the list
    for c in circles2[0,:]:
        
        data.append( [c[2],'Y', 1.0, c[0], c[1]])
    #look trough the list of messing coins that are actually silver
    for d in data:
        if (d[1]=='Y'):
            for c in circles3[0,:]:
                if(c[0]-error < d[3] and d[3] < c[0]+error and
                   c[1]-error < d[4] and d[4] < c[1]+error):
                    d[1]='S'
    
    #find smallest and largest in picture
    sizesave=[data[0][0],data[1][0]]
    locationsave=[0,1]
    for x in range(0,len(data)):
        if (data[x][0]<sizesave[0]):
            sizesave[0]=data[x][0]
            locationsave[0]=x
        elif (data[x][0]>sizesave[1]):
            sizesave[1]=data[x][0]
            locationsave[1]=x
    
    for x in range(0,len(data)):
        data[x][2] = (data[locationsave[1]][0]/data[x][0])**power
    
    circles=np.hstack((circles1,circles2))
    temp = copy.deepcopy(diffrence)
    lowest = 1
    ysave = 7
    
    #find what collum there needs to be worked in
    for x in range(0,len(diffrence)):
        for y in range(0,len(diffrence[0])):
            temp[x][y]=abs(diffrence[x][y]-data[locationsave[0]][2])
            if (abs(temp[x][y])<lowest):
                ysave = y
                lowest = abs(temp[x][y])
    
    
    tempvalue = 0
    typesave = 0
    #Identify coin and add to som
    for n in range(0,len(data)):
        lowest = 10000000
        if (data[n][1]=='C'):
            for x in range(0,3):
                tempvalue=diffrence[x][ysave]-data[n][2]
                if (abs(tempvalue)<lowest): 
                    lowest = abs(tempvalue)
                    typesave = x
        elif (data[n][1]=='Y'):
            for x in range(3,6):
                tempvalue=diffrence[x][ysave]-data[n][2]
                if (abs(tempvalue)<lowest): 
                    lowest = abs(tempvalue)
                    typesave = x
        elif (data[n][1]=='S'):
            for x in range(6,8):
                tempvalue=diffrence[x][ysave]-data[n][2]
                if (abs(tempvalue)<lowest): 
                    lowest = abs(tempvalue)
                    typesave = x
        data[n].append(coins[typesave][0])
        data[n].append(coins[typesave][1])
        data[n].append(lowest)  
        data[n].append(typesave)
    counter = 0
    value = 0

    for i in circles1[0,:]:
        cv2.circle(frameBGR,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(frameBGR,(i[0],i[1]),2,(0,0,255),3)
    for i in circles[0,:]:
        cv2.circle(framecopy,(i[0],i[1]),i[2],(0,255,0),2)
        cv2.circle(framecopy,(i[0],i[1]),2,(0,0,255),3)
        
        cv2.putText(framecopy, "coin : {}, {}".format(counter,data[counter][5]), 
                    (i[0],i[1]), 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
        value+=data[counter][6]
        counter+=1
    
    #print found value of coins
    print ("found: {:5.2f} Euro worth of coins".format(value))
    
    #show results
    
    result = cv2.bitwise_and(frame, frame, mask = mask)
    framecopy = cv2.resize(framecopy, (0,0), fx=resize*2.5, fy=resize*2.5)
    
    cv2.imshow('found', framecopy)
    frameBGR = cv2.resize(frameBGR, (0,0), fx=resize, fy=resize)
    cv2.imshow('blurred', frameBGR)
    result = cv2.resize(result, (0,0), fx=resize, fy=resize)
    cv2.imshow('colorTest', result)
    cv2.imwrite((images[pic][0]+"_found"+images[pic][1]),framecopy)
    
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    
#cv2.destroyAllWindows()