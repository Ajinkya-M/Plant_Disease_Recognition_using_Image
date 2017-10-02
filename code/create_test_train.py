import os
import random
import math
import shutil

path = "testing"

basePath = "data"

test = "test"
train = "train"

if not os.path.exists(path):
    os.mkdir(path)

if not os.path.exists(path + "/" + test):
    os.mkdir(path + "/" + test)

if not os.path.exists(path + "/" + train):
    os.mkdir(path + "/" + train)

allFolder = [f for f in os.listdir(basePath) if os.path.isdir(os.path.join(basePath, f))]

print(len(allFolder))

k = 100

trainRatio = 0.8
total = 0

for subFolder in allFolder:
    src = basePath + "/" + subFolder
    trainPath = path + "/" + train + "/" + subFolder
    testPath = path + "/" + test + "/" + subFolder
    if not os.path.exists(trainPath):
        os.mkdir(trainPath)
    if not os.path.exists(testPath):
        os.mkdir(testPath)

    allFiles = [f for f in os.listdir(src)]
    #print(len(allFiles))
    rand_files = random.sample(allFiles, min(len(allFiles), k))

    total = total + len(rand_files)
    
    for i in range(0, math.floor(len(rand_files) * trainRatio)):
        shutil.copy(src + "/" + rand_files[i], trainPath)

    for i in range(math.floor(len(rand_files) * trainRatio), len(rand_files)):
        shutil.copy(src + "/" + rand_files[i], testPath)

print("Total images = ", total)
