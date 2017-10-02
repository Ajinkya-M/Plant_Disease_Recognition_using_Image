import os
import random
import math
import shutil

path = "testing"

basePath = "data"

if not os.path.exists(path):
    os.mkdir(path)

allFolder = [f for f in os.listdir(basePath) if os.path.isdir(os.path.join(basePath, f))]

print(len(allFolder))

k = 100

for subFolder in allFolder:
    src = basePath + "/" + subFolder
    dest = path + "/" + subFolder
    if not os.path.exists(dest):
        os.mkdir(dest)

    allFiles = [f for f in os.listdir(src)]
    #print(len(allFiles))
    rand_files = random.sample(allFiles, min(len(allFiles), k))

    for f in rand_files:
        shutil.copy(src + "/" + f, dest)
