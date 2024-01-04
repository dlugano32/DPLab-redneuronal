## Simple tf.keras Sequential model

import numpy as np
import tensorflow as tf
import plot_confusion_matrix as pcm

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy

from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix

train_labels = []
train_samples = []

##dummy data

for i in range(50):
    # The ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # The ~5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # The ~95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # The ~95% of older individuals who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_labels, train_samples = shuffle(train_labels, train_samples) #shuffle both train_labels and train_samples  to remove any order that was imposed on the data during the creation process

scaler = MinMaxScaler(feature_range=(0,1))  #scale all of the data down from a scale ranging from 13 to 100 to be on a scale from 0 to 1
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1)) #reshape the data as a technical requirement just since the fit_transform() function doesn't accept 1D data by default

#model is an instance of a Sequential object. A tf.keras.Sequential model is a linear stack of layers. It accepts a list, and each element in the list should be a layer.
#A Dense layer is our standard fully-connected or densely-connected neural network layer
#The input data defines the input layer shape
#The model first layer defines the first hidden layer

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),   #the input_shape data is one-dimensional in this case
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')                    #the output layer will have 2 neurons because we have two possible outputs: either a patient experienced side effects, or the patient did not experience side effects.
])


model.summary()

model.compile(optimizer=Adam(learning_rate=0.1), loss='sparse_categorical_crossentropy', metrics=['accuracy'])  #This function configures the model for training

#model.compile tear apart
##Adam optimization is a stochastic gradient descent (SGD) method
##The next parameter we specify is loss. We'll be using sparse_categorical_crossentropy
##The last parameter we specify is metrics. This parameter expects a list of metrics that we'd like to be evaluated by the model during training and testing. We'll set this to a list that contains the string ‘accuracy'.

#Train using fit
#model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, verbose=2)

#model.fit tear apart
#The batch size is the number of samples that are passed to the network at once.
#Lastly, we specify verbose=2. This just specifies how much output to the console we want to see during each epoch of training. The verbosity levels range from 0 to 2, so we're getting the most verbose output.

# IF we use a validation set, using a 10% of the training data, the model will not train on it, but evaluate the loss and any model metrics on this data at the end of each epoch.
model.fit(
      x=scaled_train_samples
    , y=train_labels
    , validation_split=0.1
    , batch_size=10
    , epochs=30
    , verbose=2
)

### Inference
## Testing the Training of the Neural Network with a data set

test_labels =  []
test_samples = []

for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

#We quest the Neural Netowork with the test set

predictions = model.predict(
      x=scaled_test_samples
    , batch_size=10
    , verbose=0 # The output from the predictions won't be relevant for us, so we're setting verbose=0 for no output.
)  

rounded_predictions = np.argmax(predictions, axis=-1)   #most probable prediction.

## Confusion matrix: it will able us to read the predictions from the model easily

cm=confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)     #we define the labels for the confusion matrix
cm_plot_labels=['no_side_effects', 'had side effects']
pcm.plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')  # plot the matrix with the function given

# model.save() is for saving a model at its current state after it was trained so that we could make use of it later, we pass in the file path and name of the file we want to save the model to with an h5 extension

model.save('models/medical_trial_model.h5')