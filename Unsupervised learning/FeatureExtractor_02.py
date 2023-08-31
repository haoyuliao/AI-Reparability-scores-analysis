import numpy as np
import matplotlib.pyplot as plt
import cv2, random

def ORB(img):
  # Create ORB model
  orb = cv2.ORB_create()
  kp, des = orb.detectAndCompute(img,None)
  return kp, des

def SIFT(img):
  # Create SIFT model
  sift = cv2.SIFT_create()
  kp, des = sift.detectAndCompute(img,None)
  return kp, des

'''
def SURF(img):
  # Create ORB model
  surf = cv2.SURF_create()
  kp, des = surf.detectAndCompute(img,None)
  return kp, des
'''

imgs = np.load('ImagesArray.npy')
imgsName = np.load('ImgName.npy')
nf = 250 #Number of feature to select. None means take all.
shuffle = False
#featureEx = [ORB, SIFT]
#featureExName = ['ORB', 'SIFT']
featureEx = [ORB]
featureExName = ['ORB']
for fe in range(len(featureEx)):
    features = []
    featuresName = []
    for i in range(len(imgs)):
        img = imgs[i]
        imgName = imgsName[i].split('.')[0]
        kp, des = featureEx[fe](img)
        kp, des = np.array(kp), np.array(des)
        if len(kp) < 100:
            continue
        idx = np.arange(len(kp))
        if shuffle:
          np.random.seed(42)
          np.random.shuffle(idx)
        kpsub = kp[idx[:nf]]
        dessub = des[idx[:nf]]
        print(dessub.shape)
        if dessub.shape[0] != 250:
          continue
        dessub = dessub.reshape((dessub.shape[0]*dessub.shape[1])) #Flatten features as 1 d.
        features.append(dessub)
        featuresName.append(imgName)
        #draw keypoints location
        plt.cla()
        plt.clf()
        drawKps = cv2.drawKeypoints(img, kpsub, None, color=(0,255,0), flags=0)
        plt.imshow(drawKps)#, plt.show()
        plt.savefig("./Drawkps250Features/%s_drawKps_%s.jpg" %(featureExName[fe],imgName))
        #print(dessub.shape)
    np.save("./" + '%s_250Features' %(featureExName[fe]), features) #Save image features
    np.save("./" + '%s_250FeaturesImgName' %(featureExName[fe]), featuresName) #Save image name
