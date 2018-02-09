import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

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

    IMAGE_WIDTH = image.shape

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

def read_driving_log(path):
    samples = []

    print("1. Reading driving log...")
    with open(path) as csvfile:
        reader = csv.reader(csvfile)

        next(reader) #skip first line

        for line in reader:
            samples.append(line)

    return samples

def read_dataset(lines, correction=0.2, bUseLeftRight=True):

    images = []
    measurements = []

    print("2. Reading the dataset...")
    for line in lines:

        steering_center = float(line[3])
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        if bUseLeftRight:

            for i in range(3): #for center, left, and right image
                source_path = line[i]
                filename = source_path.split("/")[-1]
                current_path = "data/IMG/" + filename
                image = cv2.imread(current_path)
                images.append(image)

                if i==0: measurements.append(steering_center)
                if i==1: measurements.append(steering_left)
                if i==2: measurements.append(steering_right)
        else:
            source_path = line[0]
            filename = source_path.split("/")[-1]
            current_path = "data/IMG/" + filename
            image = cv2.imread(current_path)
            images.append(image)
            measurements.append(steering_center)

    #gen_hist_plot(measurements)

    return images, measurements

def data_augment(images, measurements):
    print("3. Data augmentation...")
    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        
        augmented_images.append(image)
        augmented_measurements.append(measurement)

        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement*-1.0)
    return augmented_images, augmented_measurements

def generator(features, labels, batch_size=128):
    num_samples = len(features)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):

            features_batch = features[offset:offset+batch_size]
            labels_batch = labels[offset:offset+batch_size]

            yield shuffle(features_batch, labels_batch)

def build_model(keep_prob=0.5, crop_top=70, crop_bottom=25):
    
    print("4. Build model...")

    #input
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((crop_top,crop_bottom), (0,0))))

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
    model.add(Dropout(keep_prob))

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
 
def train_model(model, train_generator, validation_generator, lr=0.001, epochs=10):
    
    print("5. Train model...")
    
    model.compile(loss="mse", optimizer=Adam(lr=lr))

    history_object = model.fit_generator(train_generator,
        steps_per_epoch= len(aug_images)/64,
        validation_data=validation_generator,
        validation_steps=len(aug_measurements)/64,
        epochs=epochs,
        verbose = 1)
    model.save("model.h5")
    
    return history_object


########################### Preprocessing ###########################
#1 Get all the paths
log_lines = read_driving_log("data/driving_log.csv")

images, measurements = read_dataset(log_lines)
aug_images, aug_measurements = data_augment(images, measurements)

# Split the data into train and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(aug_images, aug_measurements, test_size=0.2)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_valid = np.array(X_valid)
y_valid = np.array(y_valid)

# create the generators
train_generator = generator(X_train, y_train, batch_size=128)
validation_generator = generator(X_valid, y_valid, batch_size=128)

######################## Building & Training ########################
# Build model from NVIDIA paper with dropout
model = build_model()

# Train the model
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