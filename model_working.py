import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

from keras.backend import tf as ktf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import skimage.transform as sktransform
import random
import matplotlib.image as mpimg

def gen_hist_plot(angles):
    # Prints a histogram for "class" distribution overview
    num_bins = 25
    hist, bins = np.histogram(angles, num_bins)
    width = 0.6 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

########################### Augmentation Methods ###########################

def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -1.0*steering_angle
    return image, steering_angle

def random_translate(image, steering_angle, range_x=10, range_y=10):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = 320 * np.random.rand(), 0
    x2, y2 = 320 * np.random.rand(), 160
    xm, ym = np.mgrid[0:160, 0:320]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

########################### Pipeline ###########################

def resize(img):
    """
    Resizes the images in the supplied tensor to the original dimensions of the NVIDIA model (66x200)
    """
    return ktf.image.resize_images(img, [66, 200])

def read_driving_logs():

    print("Reading all driving logs...")

    driving_logs=["data_fwd/driving_log.csv",
          "data_rev/driving_log.csv",
          "data_recov_fwd/driving_log.csv",
          "data_recov_rev/driving_log.csv",]

    lines = []

    for driving_log in driving_logs:
        with open(driving_log) as csvfile:
            reader = csv.reader(csvfile)
            next(reader) #skip header
            for line in reader:
                lines.append(line)

    #mix data sets
    lines = shuffle(lines)

    return lines

def generator(samples, batch_size=128, augment=True):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:

                source_path = batch_sample[0]
                filename = source_path.split('/')[-1]
                folder = source_path.split('/')[-3]
                current_path = folder + "/IMG/" + filename
                center_image = cv2.imread(current_path)

                #convert to YUV
                #center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                center_angle = float(batch_sample[3])

                #append original data
                angles.append(center_angle)
                images.append(center_image)

            #augment batch
            if augment:
                #augment one qurter of the batch
                for i in range(int(batch_size/16)):

                    #add random shadow
                    index = random.randint(0, len(images)-1)
                    images[index] = random_shadow(images[index])

                    #add random brightness
                    index = random.randint(0, len(images)-1)
                    images[index] = random_brightness(images[index])

                    #randomly flip
                    index = random.randint(0, len(images)-1)
                    images[index], angles[index] = random_flip(images[index], angles[index])

                    #randomly translate
                    index = random.randint(0, len(images)-1)
                    images[index], angles[index] = random_translate(images[index], angles[index])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def build_model():
    
    print("4. Build model...")

    #input
    model = Sequential()
    model.add(Lambda(lambda x: ((x / 127.5) - 1), input_shape=(160,320,3), output_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Lambda(resize))

    #layer1
    model.add(Convolution2D(24,(5,5), strides=(2,2), activation="relu"))
    #layer2
    model.add(Convolution2D(36,(5,5), strides=(2,2), activation="relu"))
    #layer3
    model.add(Convolution2D(48,(5,5), strides=(2,2), activation="relu"))
    #layer4
    model.add(Convolution2D(64,(3,3), activation="relu"))
    #layer5
    model.add(Convolution2D(64,(3,3), activation="relu"))

    #fully connected 0
    model.add(Flatten())

    #fully connected 1
    model.add(Dense(100))

    #dropout
    model.add(Dropout(0.5))
    model.add(Activation('relu'))

    #fully connected 2
    model.add(Dense(50))

    #dropout
    model.add(Dropout(0.25))
    model.add(Activation('relu'))

    #fully connected 3
    model.add(Dense(10))

    #fully connected 4
    model.add(Dense(1))

    #print model topology
    model.summary()

    return model
 
def train_model(model, train_generator, validation_generator, lr=0.0001, epochs=30):
    
    print("5. Train model...")
    
    model.compile(loss="mse", optimizer=Adam(lr=lr))

    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    history_object = model.fit_generator(train_generator,
        steps_per_epoch= len(train_samples)/5,
        validation_data=validation_generator,
        validation_steps=len(validation_samples)/5,
        epochs=epochs,
        verbose = 1,
        callbacks=[checkpoint])

    model.save("model.h5")
    
    return history_object

########################### Preprocessing ###########################

if __name__ == '__main__':

    lines = read_driving_logs()

    #plot histogram
    angles=[]
    
    for line in lines:
        angles.append(float(line[3]))

    gen_hist_plot(angles)


    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=128)
    validation_generator = generator(validation_samples, batch_size=128, augment=False)

    ######################## Building & Training ########################
    # Build model from NVIDIA paper with dropout
    model = build_model()

    # Train the model
    #history_object = train_model(model, train_generator=train_generator, train_generator=validation_generator)
    history_object = train_model(model, train_generator, validation_generator)

    ############################# Evaluation #############################
    # Plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model MSE loss')
    plt.ylabel('MSE loss')
    plt.xlabel('Epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()








# determine keep probability for each bin: if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
# to number of samples above the average, so as to bring the number of samples for that bin down to the average
# keep_probs = []
# target = avg_samples_per_bin * .5
# for i in range(num_bins):
#     if hist[i] < target:
#         keep_probs.append(1.)
#     else:
#         keep_probs.append(1./(hist[i]/target))
# remove_list = []
# for i in range(len(angles)):
#     for j in range(num_bins):
#         if angles[i] > bins[j] and angles[i] <= bins[j+1]:
#             # delete from X and y with probability 1 - keep_probs[j]
#             if np.random.rand() > keep_probs[j]:
#                 remove_list.append(i)
# image_paths = np.delete(image_paths, remove_list, axis=0)
# angles = np.delete(angles, remove_list)