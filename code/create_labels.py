import os
import random
import math
import shutil
import numpy as np

path = "testing"

test = "test"
train = "train"

labels = ['apple__apple_scab', 'apple__black_rot', 'apple__cedar_apple_rust', 'apple__healthy', 'banana__banana_speckle', 'banana__black_sigatoka_black_leaf_streak', 'banana__healthy', 'blueberry__healthy', 'cabbage_red_white_savoy__black_rot', 'cabbage_red_white_savoy__healthy', 'cantaloupe__healthy', 'cassava_manioc__brown_leaf_spot', 'cassava_manioc__cassava_green_spider_mite', 'celery__early_blight_cercospora_leaf_spot_cercospora_blight', 'cherry_including_sour__healthy', 'cherry_including_sour__powdery_mildew', 'corn_maize__cercospora_leaf_spot_gray_leaf_spot', 'corn_maize__common_rust', 'corn_maize__healthy', 'corn_maize__northern_leaf_blight', 'cucumber__downy_mildew', 'cucumber__healthy', 'eggplant__healthy', 'gourd__downy_mildew', 'grape__black_rot', 'grape__esca_(black_measles_or_spanish_measles)', 'grape__healthy', 'grape__leaf_blight_(isariopsis_leaf_spot)', 'onion__healthy', 'orange__huanglongbing_(citrus_greening)', 'peach__bacterial_spot', 'peach__healthy', 'pepper_bell__bacterial_spot', 'pepper_bell__healthy', 'potato__early_blight', 'potato__healthy', 'potato__late_blight', 'pumpkin__cucumber_mosaic', 'raspberry__healthy', 'soybean__downy_mildew', 'soybean__frogeye_leaf_spot', 'soybean__healthy', 'soybean__septoria_leaf_blight', 'squash__healthy', 'squash__powdery_mildew', 'strawberry__healthy', 'strawberry__leaf_scorch', 'tomato__bacterial_spot', 'tomato__early_blight', 'tomato__healthy', 'tomato__late_blight', 'tomato__leaf_mold', 'tomato__septoria_leaf_spot', 'tomato__spider_mites_two_spotted_spider_mite', 'tomato__target_spot', 'tomato__tomato_mosaic_virus', 'tomato__tomato_yellow_leaf_curl_virus', 'watermelon__healthy']
def getLabel(folder):
    return labels.index(folder)       

allFolder = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
#allFolder = [f for f in os.listdir(os.path.join(path, train))]
#print(allFolder)
#test and train

for f in allFolder:
    file_ptr = open("%s/%s.csv" % (path, f), "w")
    allSubFolder = [folder for folder in os.listdir(os.path.join(path, f))]
    #all folders inside test and train
    for sf in allSubFolder:
        allFiles = [ssf for ssf in os.listdir(os.path.join(path, f, sf))]
        
        for x in allFiles:
            path_x = os.path.join(path, f, sf, x)
            label_x = getLabel(sf)
            file_ptr.write(path_x + "," + str(label_x) + "\n")
    file_ptr.close()
