import tensorflow
import numpy as np
from keras.datasets import mnist
from tensorflow.keras import models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
#%%
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train/255 
x_test = x_test/255
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
#%%
model = Sequential()
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = x_train.shape[1:]))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(32, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = CategoricalCrossentropy(), metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 5, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#%%
# Classify a few samples from the set
num_samples = 5
#predictions = model.predict(x_test[:num_samples])
#predicted_classes = np.argmax(predictions, axis=-1)

for i in range(num_samples):
  # Show image
  plt.imshow(x_test[i])
  plt.show()
  plt.savefig(r'D:\College Files\Flask Codes\mnist\test1.png')
  print("Actual label:", y_test[i])
  #print("Predicted label:", predicted_classes[i])
  print("\n\n")
#%%
# Save the model
model.save(r"D:\College Files\Flask Codes\mnist\CNNModel.h5")

# Load
model = models.load_model("CNNModel.h5")
#%%
plt.imshow(x_test[1])

plt.savefig(r'D:\College Files\Flask Codes\mnist\test1.png')
plt.show()
