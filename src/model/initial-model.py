from loaddata import create_master_dataset

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


dataDir = '../../data/raw/cifar-10-batches-py'
trainXTemp, trainYTemp, testXTemp, testYTemp = create_master_dataset(dataDir)

trainSize = int(trainXTemp.shape[0] * 0.8)
valSize = trainXTemp.shape[0] - trainSize
testSize = testXTemp.shape[0]

allTrainSet = trainXTemp.reshape([trainSize+valSize, 32, 32, 3])

trainX = allTrainSet[:trainSize]
trainY = to_categorical(trainYTemp[:trainSize])
valX = allTrainSet[trainSize:]
valY = to_categorical(trainYTemp[trainSize:])
testX = testXTemp.reshape([testSize, 32, 32, 3])
testY = to_categorical(testYTemp)

model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                        input_shape=(32, 32, 3)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                        input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                        input_shape=(32, 32, 3)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                        input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                        input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['categorical_accuracy'])
model.fit(trainX, trainY, epochs=10, validation_data=(valX, valY), verbose=2)