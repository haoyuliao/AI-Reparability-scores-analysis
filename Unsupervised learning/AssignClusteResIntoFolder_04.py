import pandas as pd
import numpy as np
import os
import shutil

resClu = pd.read_excel('./ORB_Res250Features/ORB_4.xlsx')
saveFolder = 'AssignLabelFolder'
if not os.path.exists(saveFolder):
    os.makedirs(saveFolder)

       
for i in range(len(resClu)):
    picName = resClu['Name'][i]
    if picName == 'Shift_6_5': #Modify incorrect name image.
        picName = 'Shift_6_5.1_2015'
    subfloder = resClu['Label'][i]
    saveSubFolder = './%s/%s/' %(saveFolder, subfloder)
    if not os.path.exists(saveSubFolder):
        os.makedirs(saveSubFolder)
    srcImg ='./Dataset/%s.jpg' %(picName)
    copyImg = saveSubFolder+picName+'.jpg'
    shutil.copy(srcImg, copyImg) 
    #print(picName, floder)
