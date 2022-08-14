from keras.utils import to_categorical
from tensorflow import keras
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras import layers
from keras.datasets import mnist


(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape(60000, 28, 28).astype("float32") / 255
test_X = test_X.reshape(10000, 28, 28).astype("float32") / 255

train_y = to_categorical(train_y)
test_y = to_categorical(test_y)


model = keras.models.Sequential([
  layers.Flatten(input_shape=(28,28)),
  Dense(64, activation='relu'),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax')
])
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(0.001),
              metrics=['categorical_accuracy'])

training = model.fit(train_X, train_y, batch_size=16, epochs=10)
results = model.evaluate(test_X, test_y, batch_size=2)
print(results)



metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15, 3))


ax[0].set(title="Training")
ax11 = ax[0].twinx()
ax[0].plot(training.history['loss'], color='black')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss', color='black')
for metric in metrics:
    ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
ax11.legend()


plt.show()

