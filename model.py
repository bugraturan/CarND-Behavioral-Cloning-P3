########################### Model script #####################################
########################### Behavioral cloning ###############################
########################### Udacity CarND-Term1-P3 ###########################
########################### Author: Bugra Turan ##############################

#main imports
import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import argparse

#import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

#import tensorflow keras backend
from keras.backend import tf as ktf

#import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

########################### Helper Methods #################################

def gen_hist_plot(angles):
    """
    Prints a histogram for steering angle distribution overview
    """
    num_bins = 25
    hist, bins = np.histogram(angles, num_bins)
    width = 0.6 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

def augment_discriminator(angle, glob_augment_probs, glob_bins):
    """
    Decide if augmentation needs to be applied
    """
    for i in range(len(glob_bins)):
        #check in which angle bin the angle is
        if angle > glob_bins[i] and angle <= glob_bins[i+1]:
            return (np.random.rand() > glob_augment_probs[i])

def random_augment_operation(image, angle):
    """
    Execute random augmentation operation
    """
    op = random.choice([1,2,3,4])

    #change to <= for multiple augmentations on same image
    if op ==1:
        #add random brightness
        image = shadow(image)
    if op ==2:
        #add random brightness
        image = brightness(image)
    if op ==3:
        #randomly flip
        image, angle = vflip(image, angle)
    if op ==4:
        #randomly translate
        image, angle = translate(image, angle)

    return image, angle

########################### Augmentation Methods ###########################

def vflip(image, steering_angle):
    """
    Vertically flip image
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = steering_angle*(-1.0)
    return image, steering_angle

def translate(image, steering_angle, dx=10, dy=10):
    """
    Translates image within defined pixel range
    """
    shiftx = dx * (np.random.rand() - 0.5)
    shifty = dy * (np.random.rand() - 0.5)
    steering_angle += shiftx * 0.002
    mask = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, mask, (width, height))
    return image, steering_angle

def shadow(image):
    """
    Apply random shadow
    """
    x1, y1 = 320 * np.random.rand(), 0
    x2, y2 = 320 * np.random.rand(), 160
    xm, ym = np.mgrid[0:160, 0:320]

    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

def brightness(image):
    """
    Change brightness of image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

########################### Pipeline Functions #############################

def resize(img):
    """
    Resize directly within the pipeline
    """
    return ktf.image.resize_images(img, [66, 200])

def read_driving_logs(driving_logs):
    """
    Read all driving logs from the given paths
    """
    print('-' * 50)
    print("1. Reading all driving logs...")
    print('-' * 50)

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

def generator(samples, batch_size, augment=False, augmentation_divider=16, use_angle_dep_augmentation=False, glob_augment_probs=None, glob_bins=None):
    """
    Data batch generator
    """

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
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)
                center_angle = float(batch_sample[3])

                #append original data
                angles.append(center_angle)
                images.append(center_image)

            #augment batch?
            if augment:
                #how much of the current batch do we want to augment?
                for i in range(int(batch_size/augmentation_divider)):

                    if use_angle_dep_augmentation:
                        #choose random data from current batch
                        index = random.randint(0, len(images)-1)

                        #do we want to augment this angle?
                        if augment_discriminator(angles[index], glob_augment_probs, glob_bins):
                            #apply random augmentation operation
                            images[index], angles[index] = random_augment_operation(images[index], angles[index])
                    else:
                        #add random shadow
                        index = random.randint(0, len(images)-1)
                        images[index] = shadow(images[index])

                        #add random brightness
                        index = random.randint(0, len(images)-1)
                        images[index] = brightness(images[index])

                        #randomly flip
                        index = random.randint(0, len(images)-1)
                        images[index], angles[index] = vflip(images[index], angles[index])

                        #randomly translate
                        index = random.randint(0, len(images)-1)
                        images[index], angles[index] = translate(images[index], angles[index])

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

def build_model(keep_prob1, keep_prob2, crop_top, crop_bot):
    
    print('-' * 50)
    print("4. Building model...")
    print('-' * 50)

    #input
    model = Sequential()
    model.add(Lambda(lambda x: ((x / 127.5) - 1), input_shape=(160,320,3), output_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((crop_top,crop_bot), (0,0))))
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
    model.add(Dropout(keep_prob1))
    model.add(Activation('relu'))

    #fully connected 2
    model.add(Dense(50))

    #dropout
    model.add(Dropout(keep_prob2))
    model.add(Activation('relu'))

    #fully connected 3
    model.add(Dense(10))

    #fully connected 4
    model.add(Dense(1))

    #print model topology
    model.summary()

    return model
 
def train_model(model, train_generator, validation_generator, train_samples,  validation_samples, lr, epochs, samples_per_epoch_divider):
    
    print('-' * 50)
    print("5. Training model...")
    print('-' * 50)
    
    model.compile(loss="mse", optimizer=Adam(lr=lr))

    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    history_object = model.fit_generator(train_generator,
        steps_per_epoch= len(train_samples)/samples_per_epoch_divider,
        validation_data=validation_generator,
        validation_steps=len(validation_samples)/samples_per_epoch_divider,
        epochs=epochs,
        verbose = 1,
        callbacks=[checkpoint])

    model.save("model.h5")
    
    return history_object

########################### Main Part #######################################

def main():

    #argument parser
    parser = argparse.ArgumentParser(description='Behavioral Cloning')
    parser.add_argument('-keep1', help='drop out probability', dest='keep_prob1', type=float, default=0.5)
    parser.add_argument('-keep2', help='drop out probability', dest='keep_prob2', type=float, default=0.25)
    parser.add_argument('-epochs', help='number of epochs', dest='epochs', type=int,   default=5)
    parser.add_argument('-s_div', help='samples per epoch divider', dest='samples_per_epoch_divider', type=int, default=5)
    parser.add_argument('-batch_size', help='batch size', dest='batch_size', type=int, default=128)
    parser.add_argument('-batch_aug_div', help='batch augmentation divider', dest='augmentation_divider', type=int, default=16)
    parser.add_argument('-angDep', help='Use angle dep augmentation', dest='use_angle_dep_augmentation', type=bool, default=True)
    parser.add_argument('-lr', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-c_top', help='crop top', dest='crop_top', type=int, default=70)
    parser.add_argument('-c_bot', help='crop bottom', dest='crop_bot', type=int, default=25)
    args = parser.parse_args()

    #print parameters
    print('-' * 50)
    print('0. Read all parameters')
    print('-' * 50)
    for key, value in vars(args).items():
        print('{:<30} := {}'.format(key, value))

    #read the driving logs of all measurements
    driving_logs=["data_fwd/driving_log.csv",
          "data_rev/driving_log.csv",
          "data_recov_fwd/driving_log.csv",
          "data_recov_rev/driving_log.csv",]

    lines = read_driving_logs(driving_logs)

    #plot histogram of steering angle distribution
    print('-' * 50)
    print("2. Plot steering angle distribution...")
    print('-' * 50)
    angles=[]
    for line in lines:
        angles.append(float(line[3]))
    gen_hist_plot(angles)

    #create list with inverse of the steering angle distribution
    print('-' * 50)
    print("3. Define augmentation policy...")
    print('-' * 50)
    num_bins = 25
    hist, glob_bins = np.histogram(angles, num_bins)
    glob_augment_probs = []
    target = sum(hist)/num_bins * 0.5
    for i in range(num_bins):
        if hist[i] < target:
            glob_augment_probs.append(1.0)
        else:
            glob_augment_probs.append(target/hist[i])

    plt.plot(glob_augment_probs)
    plt.show()
    
    #split log lines set
    train_samples, validation_samples = train_test_split(lines, test_size=0.2)

    #define generator functions
    train_generator = generator(train_samples, 
        args.batch_size, 
        augment=True,
        augmentation_divider=args.augmentation_divider,
        use_angle_dep_augmentation=args.use_angle_dep_augmentation,
        glob_augment_probs=glob_augment_probs, 
        glob_bins=glob_bins)

    validation_generator = generator(validation_samples, args.batch_size)

    #build model from NVIDIA paper with dropout
    model = build_model(args.keep_prob1, args.keep_prob2, args.crop_top, args.crop_bot)

    #train the model
    history_object = train_model(model, 
        train_generator, 
        validation_generator,
        train_samples,
        validation_samples,
        args.learning_rate, 
        args.epochs,
        args.samples_per_epoch_divider)

    #plot the training and validation results
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('MSE loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()