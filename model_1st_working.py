import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import itertools

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

from keras.backend import tf as ktf

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def gen_hist_plot(y_train):
    # Prints a histogram for "class" distribution overview
    num_bins = 25
    hist, bins = np.histogram(y_train, num_bins)
    width = 0.6 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()

def random_shadow(image):

    IMAGE_WIDTH = np.array(image).shape[1]
    IMAGE_HEIGHT = np.array(image).shape[0]

    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

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
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def data_augment(images, measurements):
    #print("3. Data augmentation...")
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        if abs(measurement)>0:
            augmented_images.append(image)
            augmented_measurements.append(measurement)
            augmented_images.append(cv2.flip(image,1))
            augmented_measurements.append(-1.0*measurement)
            if abs(measurement)>0.25:
                augmented_images.append(random_shadow(image))
                augmented_measurements.append(measurement)
                if abs(measurement)>0.5:
                    augmented_images.append(random_brightness(image))
                    augmented_measurements.append(measurement)



    return augmented_images, augmented_measurements

def read_driving_logs(paths):
     lines = []
     print("Reading driving logs...")
     for path in paths:
         with open(path) as csvfile:
             reader = csv.reader(csvfile)
             next(reader, None) #skip first line
             for line in reader:
                 lines.append(line)
     return lines

def resize(img):
    """
    Resizes the images in the supplied tensor to the original dimensions of the NVIDIA model (66x200)
    """
    return ktf.image.resize_images(img, [66, 200])

def generator2(lines, batch_size=128):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):

            lines_batch = lines[offset:offset+batch_size]

            images = []
            measurements = []

            for line in lines_batch:
                steering_center = float(line[3])
                steering_left = steering_center + 0.25
                steering_right = steering_center - 0.25

                for i in range(1): #for center, left, and right image
                    source_path = line[i]
                    filename = source_path.split('/')[-1]


                    if len(source_path.split('/'))>2:
                        foldername = source_path.split('/')[-3]
                        current_path = foldername + "/IMG/" + filename
                    else:
                        current_path =  "data/IMG/" + filename

                    image = cv2.imread(current_path)
                    
                    if not image is None: #if image could be read
                        images.append(image)
                        if i==0: measurements.append(steering_center)
                        if i==1: measurements.append(steering_left)
                        if i==2: measurements.append(steering_right)

            #images, measurements = data_augment(images, measurements)

            X = np.array(images)
            y = np.array(measurements)

            yield shuffle(X, y)
            
def build_model(keep_prob=0.8, crop_top=70, crop_bottom=25):
    
    print("4. Build model...")

    #input
    model = Sequential()
    model.add(Lambda(lambda x: ((x / 255.0) - 0.5), input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0))))
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

    #dropout
    #model.add(Dropout(keep_prob))

    #fully connected 0
    model.add(Flatten())

    #fully connected 1
    model.add(Dense(100))

    #fully connected 2
    model.add(Dense(50))

    #fully connected 3
    model.add(Dense(10))

    #fully connected 4
    model.add(Dense(1))

    #print model topology
    model.summary()

    return model
 
def train_model(model, X=None, y=None, train_generator=None, validation_generator=None, lr=0.001, epochs=5):
    
    print("5. Train model...")
    
    model.compile(loss="mse", optimizer=Adam())

    #checkpoint = ModelCheckpoint('model{epoch:02d}.h5')

    #history_object = model.fit_generator(train_generator,
        #steps_per_epoch= len(test_samples),
        #validation_data=validation_generator,
        #validation_steps=len(validation_samples),
        #epochs=epochs,
        #verbose = 1,
        #callbacks=[checkpoint])
    history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=epochs)


    model.save("model.h5")
    
    return history_object




def read_dataset(paths, correction=0.25):

    lines = []
    images = []
    measurements = []

    print("Reading data sets...")
    for path in paths:
        with open(path) as csvfile:
            reader = csv.reader(csvfile)

            next(reader) #skip first line

            for line in reader:

                steering_center = float(line[3])
                steering_left = steering_center + correction
                steering_right = steering_center - correction

                for i in range(1): #for center, left, and right image
                    source_path = line[i]
                    filename = source_path.split('/')[-1]
                    folder = path.split('/')[0]
                    current_path = folder + "/IMG/" + filename
                    image = cv2.imread(current_path)
                
                    images.append(image)
                    if i==0: measurements.append(steering_center)
                    if i==1: measurements.append(steering_left)
                    if i==2: measurements.append(steering_right)

    images, measurements = data_augment(images, measurements)
    gen_hist_plot(measurements)
    return np.array(images), np.array(measurements)

def generator(X, y, batch_size=128):
    num_samples = len(X)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):

            X_batch = X[offset:offset+batch_size]
            y_batch = y[offset:offset+batch_size]

            yield shuffle(X_batch, y_batch)

########################### Preprocessing ###########################
#1 Get all the paths
dirs=["data/driving_log.csv"]
  #    "data_fwd/driving_log.csv",
   ##   "data_rev/driving_log.csv"]
  #    "data_track2_fwd/driving_log.csv",
  #    "data_track2_rev/driving_log.csv"]

#dirs=["data/driving_log.csv"]

#train_samples, validation_samples = train_test_split(complete_log, test_size=0.2)
#flattened_list  = list(itertools.chain(*list_of_lists))

#lines = read_driving_logs(dirs)
#test_samples, validation_samples = train_test_split(lines, test_size=0.2)

# create the generators
#train_generator = generator2(test_samples, batch_size=128)
#validation_generator = generator2(validation_samples, batch_size=128)


X_train, y_train = read_dataset(dirs)
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

# create the generators
#train_generator = generator(X_train, y_train, batch_size=128)
#validation_generator = generator(X_valid, y_valid, batch_size=128)

######################## Building & Training ########################
# Build model from NVIDIA paper with dropout
model = build_model()

# Train the model
#history_object = train_model(model, train_generator=train_generator, train_generator=validation_generator)
history_object = train_model(model, X=X_train, y=y_train)

############################# Evaluation #############################
# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model MSE loss')
plt.ylabel('MSE loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()