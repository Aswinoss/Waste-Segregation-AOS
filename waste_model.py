import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Activation,Dropout,Dense,MaxPooling2D,Flatten

# pickle to train in local machine not needed for kaggle training
#features=pickle.load(open("X.pickle","rb"))
#labels=pickle.load(open("Y.pickle","rb"))

features=features/255.0 #image normalising with maximum values


model = Sequential()

# hidden layer1
model.add(Conv2D(256, (3, 3), input_shape=features.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# hidden layer2
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# hidden layer3 (not necessarily needed)
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))

#output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

model.fit(features,labels,epochs=10,batch_size=64,validation_split=0.3)
model.save("waste_segregation.model")
