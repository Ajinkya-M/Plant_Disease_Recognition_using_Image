
# coding: utf-8

# In[2]:

import tensorflow as tf
import numpy as np
import tflearn
import os
import random
import glob
import math
import time
import cv2
from tqdm import tqdm
from tflearn.data_utils import image_preloader
import matplotlib.pyplot as plt


# In[15]:

IMAGE_FOLDER = 'G:/AI/Project/data/raw_data'
TRAIN_DATA = 'G:/AI/Project/data/training_data.txt'
TEST_DATA = 'G:/AI/Project/data/test_data.txt'
VALIDATION_DATA = 'G:/AI/Project/data/validation_data.txt'
MODEL_PATH = "G:/AI/Project/model/AI_In_Agri.ckpt"
LOG_PATH = "G:/AI/Project/logs/"

IMAGE_SIZE = 64
LR = 1e-3

NUM_CHANNEL = 3
NUM_CLASS = 58

train_proportion=0.3
test_proportion=0.6
validation_proportion=0.1


# In[4]:

labels = ['apple__apple_scab', 'apple__black_rot', 'apple__cedar_apple_rust', 'apple__healthy', 'banana__banana_speckle', 'banana__black_sigatoka_black_leaf_streak', 'banana__healthy', 'blueberry__healthy', 'cabbage_red_white_savoy__black_rot', 'cabbage_red_white_savoy__healthy', 'cantaloupe__healthy', 'cassava_manioc__brown_leaf_spot', 'cassava_manioc__cassava_green_spider_mite', 'celery__early_blight_cercospora_leaf_spot_cercospora_blight', 'cherry_including_sour__healthy', 'cherry_including_sour__powdery_mildew', 'corn_maize__cercospora_leaf_spot_gray_leaf_spot', 'corn_maize__common_rust', 'corn_maize__healthy', 'corn_maize__northern_leaf_blight', 'cucumber__downy_mildew', 'cucumber__healthy', 'eggplant__healthy', 'gourd__downy_mildew', 'grape__black_rot', 'grape__esca_(black_measles_or_spanish_measles)', 'grape__healthy', 'grape__leaf_blight_(isariopsis_leaf_spot)', 'onion__healthy', 'orange__huanglongbing_(citrus_greening)', 'peach__bacterial_spot', 'peach__healthy', 'pepper_bell__bacterial_spot', 'pepper_bell__healthy', 'potato__early_blight', 'potato__healthy', 'potato__late_blight', 'pumpkin__cucumber_mosaic', 'raspberry__healthy', 'soybean__downy_mildew', 'soybean__frogeye_leaf_spot', 'soybean__healthy', 'soybean__septoria_leaf_blight', 'squash__healthy', 'squash__powdery_mildew', 'strawberry__healthy', 'strawberry__leaf_scorch', 'tomato__bacterial_spot', 'tomato__early_blight', 'tomato__healthy', 'tomato__late_blight', 'tomato__leaf_mold', 'tomato__septoria_leaf_spot', 'tomato__spider_mites_two_spotted_spider_mite', 'tomato__target_spot', 'tomato__tomato_mosaic_virus', 'tomato__tomato_yellow_leaf_curl_virus', 'watermelon__healthy']

def getLabel(folder):
    return labels.index(folder)    

def generate_all_files():
    if os.path.exists(TRAIN_DATA):
        return
    
    allFolder = os.listdir(IMAGE_FOLDER)
    #print(allFolder)

    filenames_image = []

    for f in tqdm(allFolder):
        allFiles = os.listdir(os.path.join(IMAGE_FOLDER, f))        
        for x in allFiles:
            path_x = IMAGE_FOLDER + "/" + f + "/" + x
            label_x = getLabel(f)
            filenames_image.append(path_x + " " + str(label_x) + "\n")

    #print(len(filenames_image))
    random.shuffle(filenames_image)
    #print(filenames_image[0]) 
    
    #total number of images
    total = len(filenames_image)
    ##  *****training data******** 
    fp = open(TRAIN_DATA, 'w')
    train_files = filenames_image[0: int(train_proportion*total)]
    for filename in train_files:
        fp.write(filename)

    fp.close()
    print("Training Data is created on %s" % TRAIN_DATA)
    ##  *****testing data******** 
    fp = open(TEST_DATA, 'w')
    test_files = filenames_image[int(math.ceil(train_proportion*total)):int(math.ceil((train_proportion+test_proportion)*total))]
    for filename in test_files:
        fp.write(filename)

    fp.close()
    print("Testing Data is created on %s" % TEST_DATA)

    ##  *****validation data******** 
    fp = open(VALIDATION_DATA, 'w')
    valid_files = filenames_image[int(math.ceil((train_proportion+test_proportion)*total)):total]
    for filename in valid_files:
        fp.write(filename)

    fp.close()
    print("Validation Data is created on %s" % VALIDATION_DATA)


# In[5]:

def getOneHot(lbl):
    arr = np.zeros((NUM_CLASS))
    arr[int(lbl)] = 1
    #print(arr)
    return arr

def load_data(filepath):
    fp = open(filepath)
    data = []
    for line in tqdm(fp):
        data.append(line)
    return data

def separate_path_labels(batch):
    paths = []
    classes  = []
    for line in batch:
        x = line.split(" ")
        path = x[0]
        lbl = x[1]
        paths.append(path)
        classes.append(lbl)
    return paths, classes

def load_batch(batch, lbl):
    images = []
    classes  = []
    for i in range(len(batch)):
        path = batch[i]
        l = lbl[i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        onehot = getOneHot(l)
        #data.append([np.array(img), np.array(onehot)])
        images.append(img)
        classes.append(np.array(onehot))
    return images, classes


# In[6]:

X = tf.placeholder(tf.float32,shape=[None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL], name='input_image') 
#input class
Y_ = tf.placeholder(tf.float32,shape=[None, NUM_CLASS], name='input_class')


# In[7]:

input_layer = X
#convolutional layer 1 --convolution+RELU activation
conv_layer1 = tflearn.layers.conv.conv_2d(input_layer, nb_filter=64, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu', regularizer="L2", name='conv_layer_1')

#2x2 max pooling for layer 1 with kernel size = 2
out_layer1 = tflearn.layers.conv.max_pool_2d(conv_layer1, 2) 


#second convolutional layer 
conv_layer2 = tflearn.layers.conv.conv_2d(out_layer1, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2", name='conv_layer_2')

#2x2 max pooling for layer 2 with kernel size = 2
out_layer2 = tflearn.layers.conv.max_pool_2d(conv_layer2, 2)

# third convolutional layer
conv_layer3 = tflearn.layers.conv.conv_2d(out_layer2, nb_filter=128, filter_size=5, strides=[1,1,1,1],
                                        padding='same', activation='relu',  regularizer="L2", name='conv_layer_3')

#2x2 max pooling for layer 3 with kernel size = 2
out_layer3 = tflearn.layers.conv.max_pool_2d(conv_layer3, 2)

#fully connected layer1
fcl = tflearn.layers.core.fully_connected(out_layer3, 4096, activation='relu' , name='FCL-1')
fcl_dropout_1 = tflearn.layers.core.dropout(fcl, 0.8)

#fully connected layer2
fc2 = tflearn.layers.core.fully_connected(fcl_dropout_1, 1024, activation='relu' , name='FCL-2')
fcl_dropout_2 = tflearn.layers.core.dropout(fc2, 0.8)
#softmax layer output
y_predicted = tflearn.layers.core.fully_connected(fcl_dropout_2, NUM_CLASS, activation='softmax', name='output')


# In[8]:

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y_ * tf.log(y_predicted + np.exp(-10)), reduction_indices=[1]))
#tf.summary.scalar("loss", cross_entropy)
#optimiser -
train_step = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
#calculating accuracy of our model 
correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(Y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#tf.summary.scalar("accuracy", accuracy)


# In[9]:

# session parameters
sess = tf.InteractiveSession()
#initialising variables
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver()


# In[10]:

epoch = 2
#change batch size according to your hardware's power. For GPU's use batch size in powers of 2 like 2,4,8,16...
batch_size = 20


# In[12]:

def train(epoch = 1, batch_size = 20, snapshot = 500):
    #writer = tf.train.SummaryWriter(LOG_PATH, sess.graph)

    st_time = time.time()
    train_data = load_data(TRAIN_DATA)
    test_data = load_data(TEST_DATA)

    for epoch_i in range(epoch):
        ######   
        random.shuffle(train_data)
        random.shuffle(test_data)

        X_train, Y_train = separate_path_labels(train_data)
        X_test, Y_test = separate_path_labels(test_data)    

        #X_train = X_train[:500]
        #Y_train = Y_train[:500]
        #X_test = X_test[:60]
        #Y_test = Y_test[:60]
        ######
        print("Epoch %d started..." % (epoch_i + 1))
        previous_batch = 0
        start_time = time.time()
        num_batches = math.ceil(len(X_train) / batch_size)
        for i in range(num_batches):
            #batch wise training 
            if previous_batch >= len(X_train) : #total --> total number of training images
                previous_batch = 0    
            current_batch = previous_batch + batch_size
            if current_batch > len(X_train) :
                current_batch = len(X_train)

            x_images, y_label = load_batch(X_train[previous_batch : current_batch], Y_train[previous_batch : current_batch])

            #x_images = np.reshape(np.array(x_images), [current_batch - previous_batch, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
            #y_label = np.reshape(np.array(y_label), [current_batch - previous_batch, NUM_CLASS])

            previous_batch = previous_batch + batch_size
            _, loss = sess.run([train_step, cross_entropy], feed_dict = {X: x_images, Y_: y_label}) 
            if i % snapshot == 0 or (i + 1) == num_batches:
                n = 50 #number of test samples
                # increase the number of test samples with higher RAM. if you have less RAM, limit your test sample or 
                # run test accross larger samples once in every 1000 epochs or so..  
                x_test_images, y_test_labels = load_batch(X_test[0 : n], Y_test[0 : n])
                #x_test_images = np.reshape(x_test_images, [n, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
                #y_test_labels = np.reshape(y_test_labels, [n, NUM_CLASS])

                Accuracy = sess.run(accuracy, feed_dict = {X: x_test_images, Y_: y_test_labels})
                print("Iteration no : %d, Accuracy : %f, Loss : %f" % (i, Accuracy, loss))
                #saver.save(sess, MODEL_PATH, global_step = i)
            elif i % 100 == 0:   
                print("Iteration no : %d, Loss : %f" % (i, loss))

        saver.save(sess, MODEL_PATH)
        print("Time required for epoch : %d is %f min\n" % (epoch_i + 1, (time.time() - start_time) / 60.0))

    print("Total time required is %f min" % ((time.time() - st_time) / 60.0))

def retrain(epoch = 7, batch_size = 40, snapshot = 500):
    saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
    saver.restore(sess, MODEL_PATH)
    train(epoch, batch_size, snapshot)
    #all_vars = tf.get_collection('vars')
    #print(all_vars)
    test_data = load_data(TEST_DATA)
    X_test, Y_test = separate_path_labels(test_data)

    batch_size = 500
    total_accuracy = 0.0
    num_batches = math.ceil(len(X_test) / batch_size)
    previous_batch = 0
    for i in range(num_batches):
        #batch wise testing 
        if previous_batch >= len(X_test) : #total --> total number of training images
            previous_batch = 0    
        current_batch = previous_batch + batch_size
        if current_batch > len(X_test) :
            current_batch = len(X_test)

        x_test_images, y_test_labels = load_batch(X_test[previous_batch : current_batch], Y_test[previous_batch : current_batch])
        #x_images = np.reshape(np.array(x_images), [current_batch - previous_batch, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
        #y_label = np.reshape(np.array(y_label), [current_batch - previous_batch, NUM_CLASS])

        previous_batch = previous_batch + batch_size

        Accuracy = sess.run(accuracy, feed_dict = {X: x_test_images, Y_: y_test_labels})
        total_accuracy += Accuracy

    print("Accuracy on test data : %f" % (total_accuracy / num_batches))

def cal_accuracy():
    saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
    saver.restore(sess, MODEL_PATH)

    test_data = load_data(TEST_DATA)
    X_test, Y_test = separate_path_labels(test_data)

    #X_test = X_test[:500]
    #Y_test = Y_test[:500]
    
    batch_size = 100
    total_accuracy = 0.0
    num_batches = math.ceil(len(X_test) / batch_size)
    previous_batch = 0
    for i in range(num_batches):
        #batch wise testing 
        if previous_batch >= len(X_test) : #total --> total number of training images
            previous_batch = 0    
        current_batch = previous_batch + batch_size
        if current_batch > len(X_test) :
            current_batch = len(X_test)

        x_test_images, y_test_labels = load_batch(X_test[previous_batch : current_batch], Y_test[previous_batch : current_batch])
        #x_images = np.reshape(np.array(x_images), [current_batch - previous_batch, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
        #y_label = np.reshape(np.array(y_label), [current_batch - previous_batch, NUM_CLASS])

        previous_batch = previous_batch + batch_size

        Accuracy = sess.run(accuracy, feed_dict = {X: x_test_images, Y_: y_test_labels})
        print("Accuracy = %f" % (Accuracy))
        total_accuracy += Accuracy

    print("Accuracy on test data : %f" % (total_accuracy / num_batches))


# In[16]:

generate_all_files()


# In[19]:

#retrain(epoch = 10, batch_size = 20)
#cal_accuracy()


# In[20]:

saver = tf.train.import_meta_graph(MODEL_PATH + '.meta')
saver.restore(sess, MODEL_PATH)


# In[21]:

def get_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.normalize(img, img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = np.reshape(img, [1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
    return img

def predict(path):
    #test your own images 
    test_image = get_image(path)

    predicted_array = sess.run(y_predicted, feed_dict={X: test_image})
    predicted_class = np.argmax(predicted_array)
    #print("Prediction class = %d, Label = %s" %(predicted_class, labels[predicted_class]))
    prediction = labels[predicted_class]
    crop_name = prediction.split("__")[0]
    crop_disease = prediction.split("__")[1]
    crop_name = crop_name.replace("_", ' ').title()
    crop_disease = crop_disease.replace("_", ' ').title()
    return crop_name, crop_disease


# In[23]:

crop_name, crop_disease = predict('G:/AI/Project/data/raw_data/corn_maize__healthy/6434e093-507d-42fa-8626-5a549a572083_DSC03431_resized.JPG')
print("Crop Name = %s\nCrop Disease = %s" % (crop_name, crop_disease))

