# -*- coding: utf-8 -*-
import numpy as np
from os import listdir
import cv2


Folders = ['Dataset']

Images = []
ImgName = []
Labels = []

for i in range(0,len(Folders)):
    Floder = Folders[i]    
    Link = "./"+Floder+"/"
    PhotoN = listdir(Link)
    for j in range(0,len(PhotoN)):
        ori = cv2.imread(Link + PhotoN[j])
        #ori = cv2.cvtColor(ori, cv2.COLOR_BGR2RGB) #Covert BGR to RGB.
        ori = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY) #Covert BGR to GRAY.
        output = cv2.resize(ori,(224, 224)) #Resize image to 224 by 224
        #cv2.imwrite("Resized"+PhotoName[j],output) #Save resized images
        #save image's array, name, and label
        ImgName.append(PhotoN[j])
        Images.append(output)
        Labels.append(i)



np.save("./" + 'ImgName', ImgName) #Save image name
np.save("./" + 'ImagesArray', Images) #Save image array   
#np.save("./" + 'Labels', Labels) #Can be used in classificaiton    
    
