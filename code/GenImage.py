import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

arr = []
newImg = None
flag = True
cnt = 0

names = []

for folder in os.listdir('G:/AI/data/raw_data'):
    files = os.listdir('G:/AI/data/raw_data/' + folder)
    names.append(folder)
    img = cv2.imread('G:/AI/data/raw_data/' + folder + '/' + files[8], 1)
    img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = cv2.resize(img, (100, 100))
    if flag:
        arr = np.array(img)
        flag = False
    else:
        arr = np.concatenate((arr, img), axis = 1)
    cnt += 1
    if cnt == 8:
        cnt = 0
        flag = True
        if newImg is None:
            newImg = np.array(arr)
        else:
            newImg = np.concatenate((newImg, arr), axis = 0)

img = np.ones((100, 100, 3))
#print(arr.shape, img.shape)
while cnt <= 8:    
    if flag:
        arr = np.array(img)
        flag = False
    else:
        arr = np.concatenate((arr, img), axis = 1)

    cnt += 1
    if cnt == 8:
        if newImg is None:
            newImg = np.array(arr)
        else:
            newImg = np.concatenate((newImg, arr), axis = 0)
            
plt.imshow(newImg)
plt.axis('off')
plt.show()
#img = cv2.imread('img.png')
#vis = np.concatenate((img1, img2), axis=1)
cv2.imwrite('G:/AI/data/out.png', newImg)
i = 1
for name in names:
    name = name.replace('__', ', ')
    name = name.replace('_', ' ')
    name = name.title()
    print("(%d) %s"%(i, name), end=" ")
    i += 1
    

