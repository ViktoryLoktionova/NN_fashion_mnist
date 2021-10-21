from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_train = x_train / 255

y_train = utils.to_categorical(y_train, 10)

classes = ['футболка', 'брюки', 'свитер','платье','пальто','туфли','рубашка','кроссовки','сумка','ботинки' ]
print(y_train[0])

y_train  = utils.to_categorical(y_train, 10)

print(y_train[0])

model = Sequential()
model.add(Dense(800, input_dim = 784, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"]) #стохастический градиентный спуск

print(model.summary())

model.fit(x_train, y_train,
          batch_size=200,
          epochs=100,
          verbose=1)

predictions = model.predict(x_train)
n = 0
plt.imshow(x_train[n].reshape(28, 28), cmap=plt.cm.binary)
plt.show()

print(predictions[n])
np.argmax(predictions[n])
classes[np.argmax(predictions[n])]
np.argmax(y_train[n])
classes[np.argmax(y_train[n])]