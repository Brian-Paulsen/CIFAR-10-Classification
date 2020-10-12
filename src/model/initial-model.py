from loaddata import create_master_dataset

import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


dataDir = '../../data/raw/cifar-10-batches-py'
trainXTemp, trainYTemp, testXTemp, testYTemp = create_master_dataset(dataDir)

trainSize = int(trainXTemp.shape[0] * 0.8)
valSize = trainXTemp.shape[0] - trainSize
testSize = testXTemp.shape[0]

allTrainSet = np.moveaxis(trainXTemp.reshape([trainSize+valSize, 3, 32, 32]), 1, -1)

trainX = allTrainSet[:trainSize] / 255
trainY = to_categorical(trainYTemp[:trainSize])
valX = allTrainSet[trainSize:] / 255
valY = to_categorical(trainYTemp[trainSize:])
testX = np.moveaxis(testXTemp.reshape([testSize, 3, 32, 32]) / 255, 1, -1)
testY = to_categorical(testYTemp)

# data augmentation
imageGenerator = ImageDataGenerator(
        horizontal_flip=True)
trainAug = imageGenerator.flow(trainX, trainY)

modelNum = 9

if modelNum == 1:
    model = Sequential()
    model.add(layers.Input(shape=(32, 32, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(trainX, trainY, epochs=40, validation_data=(valX, valY), verbose=1)
    
elif modelNum == 2:
    # BatchNormalization after activation
    model = Sequential()
    
    model.add(layers.Input(shape=(32, 32, 3)))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dense(10, activation='softmax'))
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(trainX, trainY, epochs=30, validation_data=(valX, valY), verbose=1)
    
elif modelNum == 3:
    # BatchNormalization before activation
    model = Sequential()
    
    model.add(layers.Input(shape=(32, 32, 3)))
    
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(32))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    model.add(layers.Dense(10))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('softmax'))
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(trainX, trainY, epochs=30, validation_data=(valX, valY), verbose=1)
    
elif modelNum == 4:
    # After the activation performed better for batch normalization
    # Add dropout after all layers expect input
    model = Sequential()
    
    model.add(layers.Input(shape=(32, 32, 3)))
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(trainX, trainY, epochs=30, validation_data=(valX, valY), verbose=1)
    
elif modelNum == 5:
    # Tries different architecture using inception modules
    # Let's not use dropout for now

    input_ = layers.Input(shape=(32, 32, 3))
    
    conv0 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_)
    batch0 = layers.BatchNormalization()(conv0)
    
    dropout1 = layers.Dropout(0.3)(batch0)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout1)
    batch1 = layers.BatchNormalization()(conv1)
    
    pool1 = layers.MaxPooling2D((2, 2))(batch1)
    
    dropout2 = layers.Dropout(0.3)(pool1)
    conv2a = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout2)
    conv2b = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(dropout2)
    conv2c = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(dropout2)
    conv2 = layers.Concatenate()([conv2a, conv2b, conv2c])
    batch2 = layers.BatchNormalization()(conv2)
    
    pool2 = layers.MaxPooling2D((2, 2))(batch2)
    
    dropout3 = layers.Dropout(0.3)(pool2)
    conv3a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(dropout3)
    conv3b = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout3)
    conv3c = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(dropout3)
    conv3 = layers.Concatenate()([conv3a, conv3b, conv3c])
    batch3 = layers.BatchNormalization()(conv3)
    
    dropout4 = layers.Dropout(0.3)(batch3)
    conv4a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(dropout4)
    conv4b = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout4)
    conv4c = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(dropout4)
    conv4 = layers.Concatenate()([conv4a, conv4b, conv4c])
    batch4 = layers.BatchNormalization()(conv4)
    
    pool4 = layers.MaxPooling2D((2, 2))(batch4)
    flat4 = layers.Flatten()(pool4)
    
    dropout5 = layers.Dropout(0.3)(flat4)
    dense5 = layers.Dense(32, activation='relu')(dropout5)
    batch5 = layers.BatchNormalization()(dense5)
    
    output = layers.Dense(10, activation='softmax')(batch5)
    
    model = Model(inputs=[input_], outputs=[output])
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(trainX, trainY, epochs=30, validation_data=(valX, valY), verbose=1)
    
elif modelNum == 6:
    # try reducing parameters by adding 1x1 convolutions in layer
    # 4 inception module
    
    input_ = layers.Input(shape=(32, 32, 3))
    
    conv0 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_)
    batch0 = layers.BatchNormalization()(conv0)
    
    dropout1 = layers.Dropout(0.3)(batch0)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout1)
    batch1 = layers.BatchNormalization()(conv1)
    
    pool1 = layers.MaxPooling2D((2, 2))(batch1)
    
    dropout2 = layers.Dropout(0.3)(pool1)
    conv2a = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout2)
    conv2b = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(dropout2)
    conv2c = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(dropout2)
    conv2 = layers.Concatenate()([conv2a, conv2b, conv2c])
    batch2 = layers.BatchNormalization()(conv2)
    
    pool2 = layers.MaxPooling2D((2, 2))(batch2)
    
    dropout3 = layers.Dropout(0.3)(pool2)
    conv3a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(dropout3)
    conv3b = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout3)
    conv3c = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(dropout3)
    conv3 = layers.Concatenate()([conv3a, conv3b, conv3c])
    batch3 = layers.BatchNormalization()(conv3)
    
    dropout4 = layers.Dropout(0.3)(batch3)
    conv4a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(dropout4)
    conv4b1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout4)
    conv4b = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv4b1)
    conv4c1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout4)
    conv4c = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(conv4c1)
    conv4 = layers.Concatenate()([conv4a, conv4b, conv4c])
    batch4 = layers.BatchNormalization()(conv4)
    
    pool4 = layers.MaxPooling2D((2, 2))(batch4)
    flat4 = layers.Flatten()(pool4)
    
    dropout5 = layers.Dropout(0.3)(flat4)
    dense5 = layers.Dense(32, activation='relu')(dropout5)
    batch5 = layers.BatchNormalization()(dense5)
    
    output = layers.Dense(10, activation='softmax')(batch5)
    
    model = Model(inputs=[input_], outputs=[output])
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(trainX, trainY, epochs=45, validation_data=(valX, valY), verbose=1)
    
    model.evaluate(testX, testY) # 0.8317
    
elif modelNum == 7:

    # removing layer 4 of previous
    
    input_ = layers.Input(shape=(32, 32, 3))
    
    conv0 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_)
    batch0 = layers.BatchNormalization()(conv0)
    
    dropout1 = layers.Dropout(0.3)(batch0)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout1)
    batch1 = layers.BatchNormalization()(conv1)
    
    pool1 = layers.MaxPooling2D((2, 2))(batch1)
    
    dropout2 = layers.Dropout(0.3)(pool1)
    conv2a = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout2)
    conv2b = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(dropout2)
    conv2c = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(dropout2)
    conv2 = layers.Concatenate()([conv2a, conv2b, conv2c])
    batch2 = layers.BatchNormalization()(conv2)
    
    pool2 = layers.MaxPooling2D((2, 2))(batch2)
    
    dropout3 = layers.Dropout(0.3)(pool2)
    conv3a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(dropout3)
    conv3b = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout3)
    conv3c = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(dropout3)
    conv3 = layers.Concatenate()([conv3a, conv3b, conv3c])
    batch3 = layers.BatchNormalization()(conv3)
    
    pool4 = layers.MaxPooling2D((2, 2))(batch3)
    flat4 = layers.Flatten()(pool4)
    
    dropout5 = layers.Dropout(0.3)(flat4)
    dense5 = layers.Dense(32, activation='relu')(dropout5)
    batch5 = layers.BatchNormalization()(dense5)
    
    output = layers.Dense(10, activation='softmax')(batch5)
    
    model = Model(inputs=[input_], outputs=[output])
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(trainX, trainY, epochs=45, validation_data=(valX, valY), verbose=1)

elif modelNum == 8:
    # Reduce parameters further by changing padding in first two layers
    
    input_ = layers.Input(shape=(32, 32, 3))
    
    conv0 = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(input_)
    batch0 = layers.BatchNormalization()(conv0)
    
    dropout1 = layers.Dropout(0.3)(batch0)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(dropout1)
    batch1 = layers.BatchNormalization()(conv1)
    
    pool1 = layers.MaxPooling2D((2, 2))(batch1)
    
    dropout2 = layers.Dropout(0.3)(pool1)
    conv2a = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout2)
    conv2b = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(dropout2)
    conv2c = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(dropout2)
    conv2 = layers.Concatenate()([conv2a, conv2b, conv2c])
    batch2 = layers.BatchNormalization()(conv2)
    
    pool2 = layers.MaxPooling2D((2, 2))(batch2)
    
    dropout3 = layers.Dropout(0.3)(pool2)
    conv3a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(dropout3)
    conv3b = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout3)
    conv3c = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(dropout3)
    conv3 = layers.Concatenate()([conv3a, conv3b, conv3c])
    batch3 = layers.BatchNormalization()(conv3)
    
    dropout4 = layers.Dropout(0.3)(batch3)
    conv4a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(dropout4)
    conv4b1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout4)
    conv4b = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv4b1)
    conv4c1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout4)
    conv4c = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(conv4c1)
    conv4 = layers.Concatenate()([conv4a, conv4b, conv4c])
    batch4 = layers.BatchNormalization()(conv4)
    
    pool4 = layers.MaxPooling2D((2, 2))(batch4)
    flat4 = layers.Flatten()(pool4)
    
    dropout5 = layers.Dropout(0.3)(flat4)
    dense5 = layers.Dense(32, activation='relu')(dropout5)
    batch5 = layers.BatchNormalization()(dense5)
    
    output = layers.Dense(10, activation='softmax')(batch5)
    
    model = Model(inputs=[input_], outputs=[output])
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(trainX, trainY, epochs=10, validation_data=(valX, valY), verbose=1)
    
elif modelNum == 9:
    # Add data augmentation
    
    input_ = layers.Input(shape=(32, 32, 3))
    
    conv0 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_)
    batch0 = layers.BatchNormalization()(conv0)
    
    dropout1 = layers.Dropout(0.3)(batch0)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout1)
    batch1 = layers.BatchNormalization()(conv1)
    
    pool1 = layers.MaxPooling2D((2, 2))(batch1)
    
    dropout2 = layers.Dropout(0.3)(pool1)
    conv2a = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout2)
    conv2b = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(dropout2)
    conv2c = layers.Conv2D(16, (5, 5), activation='relu', padding='same')(dropout2)
    conv2 = layers.Concatenate()([conv2a, conv2b, conv2c])
    batch2 = layers.BatchNormalization()(conv2)
    
    pool2 = layers.MaxPooling2D((2, 2))(batch2)
    
    dropout3 = layers.Dropout(0.3)(pool2)
    conv3a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(dropout3)
    conv3b = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(dropout3)
    conv3c = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(dropout3)
    conv3 = layers.Concatenate()([conv3a, conv3b, conv3c])
    batch3 = layers.BatchNormalization()(conv3)
    
    dropout4 = layers.Dropout(0.3)(batch3)
    conv4a = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(dropout4)
    conv4b1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout4)
    conv4b = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv4b1)
    conv4c1 = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(dropout4)
    conv4c = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(conv4c1)
    conv4 = layers.Concatenate()([conv4a, conv4b, conv4c])
    batch4 = layers.BatchNormalization()(conv4)
    
    pool4 = layers.MaxPooling2D((2, 2))(batch4)
    flat4 = layers.Flatten()(pool4)
    
    dropout5 = layers.Dropout(0.3)(flat4)
    dense5 = layers.Dense(32, activation='relu')(dropout5)
    batch5 = layers.BatchNormalization()(dense5)
    
    output = layers.Dense(10, activation='softmax')(batch5)
    
    model = Model(inputs=[input_], outputs=[output])
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['categorical_accuracy'])
    model.fit(trainAug, epochs=45, validation_data=(valX, valY), verbose=1)