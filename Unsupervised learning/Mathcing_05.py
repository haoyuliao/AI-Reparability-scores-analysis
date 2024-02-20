import cv2, os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#The programming is following by the artice, and adding new writing program.
#https://www.analyticsvidhya.com/blog/2019/10/detailed-guide-powerful-sift-technique-image-matching-python/

def randomShuffle(kp, des, nf):
    kp, des = np.array(kp), np.array(des)
    idx = np.arange(len(kp))
    np.random.seed(42)
    np.random.shuffle(idx)
    kpsub = kp[idx[:nf]]
    dessub = des[idx[:nf]]

    return kpsub, dessub

brandLst1 = [  #The matching smartphone modes.#0
'Apple_6_iPhone 14 Pro_2022',
'BlackBerry_8_Z10_2013',
'Essential_1_Phone_2017',
'Fairphone_10_2_2015',
'Fairphone_10_3_2019',
'Google_5_Pixel 6 Pro_2021',
'Google_6_Pixel 3a_2019',
'Google_6_Pixel 4a_2020',
'Google_7_Pixel XL_2016',
'HTC_1_One_2013',
'HTC_2_One M8_2014',
'Huawei_4_Mate 20 Pro_2018',
'Huawei_7_P9_2016',
'iPhone_2_1st Generation_2007',
'iPhone_6_13 Pro_2021',
'iPhone_6_4S_2011',
'iPhone_6_5c_2013',
'iPhone_6_5s_2013',
'iPhone_6_8_2017',
'iPhone_7_3GS_2009',
'iPhone_7_5_2012',
'iPhone_7_6s Plus_2015',
'iPhone_7_6s_2015',
'iPhone_7_6_2014',
'iPhone_7_7 Plus_2016',
'LG_8_G4_2015',
'Meizu_7_MX6_2016',
'Microsoft_2_Surface Duo_2020',
'Motorola_1_razr_2020',
'Motorola_4_Droid RAZR_2011',
'Motorola_6_Droid 3_2011',
'Nexus_7_4_2012',
'Samsung_2_Galaxy Fold_2019',
'Samsung_3_Galaxy Note 20 Ultra_2020',
'Samsung_3_Galaxy S10_2019',
'Samsung_3_Galaxy S20 Ultra_2020',
'Samsung_3_Galaxy S7_2016',
'Samsung_4_Galaxy A51_2020',
'Samsung_4_Galaxy S8_2017',
'Samsung_5_Galaxy S5 Mini_2014',
'Samsung_5_Galaxy S5_2014',
'Samsung_8_Galaxy Note II_2012',
'Samsung_8_Galaxy Note_2011',
'Samsung_8_Galaxy S III_2012',
'Samsung_8_Galaxy S4_2013',
'Shift_6_5.1_2015',
'Wiko_7_Pulp 4G Phone_2015'
]

brandLst2 = [  #The matching smartphone modes. #1
'Amazon_3_Fire Phone_2014',
'Apple_7_iPhone 14_2022',
'Google_4_Pixel 3 XL_2018',
#'Huawei_4_Mate 10 Pro_2017',
'Huawei_4_Mate 20 X 5G_2019',
'Huawei_4_Mate 40 Pro_2020',
'Huawei_5_Mate 9_2016',
'iPhone_6_11 Pro Max_2019',
'iPhone_6_12 mini_2020',
'iPhone_6_12 Pro Max_2020',
'iPhone_6_12 Pro_2020',
'iPhone_6_SE 2020_2020',
'iPhone_6_XR_2018',
'iPhone_6_XS_2018',
'iPhone_7_7_2016',
'LG_5_G6_2017',
'LG_8_G5_2016',
'Nexus_7_5X_2015',
'OnePlus_7_2_2015',
'Samsung_4_Galaxy Note7_2016',
'Samsung_4_Galaxy Note8_2017',
'Samsung_6_Galaxy S 4G_2011'
]


brandLst2 = [  #The matching smartphone modes. #1
'LG_5_G6_2017',
]

brandLst3 = [  #The matching smartphone modes. #2
'Apple_7_iPhone 14 Plus_2022',
'Fairphone_7_1_2013',
'Google_4_Pixel 4 XL_2019',
'Google_6_Pixel 2 XL_2017',
'Google_7_Pixel_2016',
'HTC_2_One M9_2015',
'iPhone_6_SE_2016',
'iPhone_7_6 Plus_2014',
'Mi_4_11_2021',
'Motorola_4_Droid 4_2012',
'Nexus_7_6_2014',
'Nexus_8_5_2013',
'Nokia_8_N8_2010',
'OnePlus_5_6_2018',
'Samsung_3_Galaxy Note10 Plus 5G_2019',
'Samsung_3_Galaxy S7 Edge_2016',
'Samsung_4_Galaxy S6_2015',
'Samsung_5_Galaxy Alpha_2014'
]

brandLst4 = [  #The matching smartphone modes. #3
'Fairphone_10_4_2021',
'HTC_5_Surround_2010',
'Huawei_4_P20 Pro_2018',
'iPhone_6_11_2019',
'iPhone_6_8 Plus_2017',
'iPhone_6_X_2017',
'iPhone_7_3G_2008',
'Motorola_7_Moto X 1st Generation_2013',
'Motorola_9_Atrix 4G_2011',
'Motorola_9_Droid Bionic_2011',
'Nexus_2_6P_2015',
'Nexus_7_S_2010',
'OnePlus_5_One_2014',
'Samsung_2_Galaxy Z Flip_2020',
'Samsung_3_Galaxy S22 Ultra_2022',
'Samsung_3_Galaxy S22_2022',
'Samsung_3_Galaxy S6 Edge_2015',
'Samsung_4_Galaxy Note Fan Edition_2017',
'Samsung_4_Galaxy S8 Plus_2017',
'Samsung_4_Galaxy S9 Plus_2018',
'Samsung_4_Galaxy S9_2018',
'Samsung_7_Galaxy Nexus_2011',
'Samsung_8_Galaxy S II_2011',
'Xiaomi_8_Redmi Note 3_2015'
]


blts = [brandLst1, brandLst2, brandLst3, brandLst4]
name = ['brandLst1', 'brandLst2', 'brandLst3', 'brandLst4']
blts = [brandLst2]
g = 0
for brandLst in blts:
    print(g)
    nf = 10 #Number of feature to select. None means take all.
    cals = []
    print(cals)
    results = {}
    results['Name'] = []
    for brd in brandLst:
        # read images
        NameImg1 = 'Huawei_4_Mate 10 Pro_2017' #The selected one to match others #Belong to 2 (brandLst1).
        #NameImg1 = 'Samsung_3_Galaxy S22 Ultra_2022' #The selected one to match others #Belong to 2 (brandLst1).
        
        NameImg2 = brd
        results['Name'].append(brd)
        img1 = cv2.imread('./Dataset/%s.jpg' %(NameImg1))  
        img2 = cv2.imread('./Dataset/%s.jpg' %(NameImg2)) 

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.resize(img1,(224, 224))
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        img2 = cv2.resize(img2,(224, 224))
        
        #Creat ORB model
        ORB = cv2.ORB_create()
        
        keypoints_1, descriptors_1 = ORB.detectAndCompute(img1,None)
        keypoints_1, descriptors_1 = np.array(keypoints_1), np.array(descriptors_1)
        keypoints_1, descriptors_1 = keypoints_1[:nf], descriptors_1[:nf]
        #keypoints_1, descriptors_1 = randomShuffle(keypoints_1, descriptors_1, 100)
        keypoints_2, descriptors_2 = ORB.detectAndCompute(img2,None)
        keypoints_2, descriptors_2 = np.array(keypoints_2), np.array(descriptors_2)
        keypoints_2, descriptors_2 = keypoints_2[:nf], descriptors_2[:nf]
        #keypoints_2, descriptors_2 = randomShuffle(keypoints_2, descriptors_2, 100)
        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

        matches = bf.match(descriptors_1,descriptors_2)
        matches = sorted(matches, key = lambda x:x.distance)
        #print(len(keypoints_1))
        #print(len(keypoints_2))
        #print(len(matches))
        cals.append(len(matches))
        
        plt.cla()
        plt.clf()
        plt.close('all')
        plt.figure(figsize=(16, 8), dpi=160)
        img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:], img2, flags=2)
        plt.imshow(img3)#,plt.show()
        plt.xticks([])
        plt.yticks([])
        if not os.path.exists('MatchingFeaturesRes'):
            os.makedirs('MatchingFeaturesRes')
        plt.savefig('./MatchingFeaturesRes/%s_%s.png' %(NameImg1, NameImg2))
        
        

                
    cals = np.array(cals)
    print(np.mean(cals))
    print(np.std(cals))
    results['Number'] = cals
    #pd.DataFrame(results).to_excel('./'+'Res%s.xlsx' %(name[g]))
    g+=1
    
