import numpy as np
from keras import models, layers
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mae', metrics=['mae'])
    return model


k = 4
num_val_samples = len(train_data)//k
num_epochs = 300
all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i*num_val_samples],
         train_data[(i+1)*num_val_samples:]],
        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i*num_val_samples],
         train_targets[(i+1)*num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=1)
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)
"""     model.fit(partial_train_data, partial_train_targets,
              epochs=num_epochs, batch_size=1, verbose=1)
    val_mse, val_mae = model.evaluate(val_data,val_targets,verbose=1)
    all_scores.append(val_mae) """


average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]

plt.figure()
plt.plot(range(16, len(average_mae_history[15:])+16), average_mae_history[15:])
plt.xlabel('Epochs')
plt.ylabel('Val MAE')
plt.show()


def smooth_curve(points, factor=0.9):
    smooth_points = []
    for point in points:
        if smooth_points:
            previous = smooth_points[-1]
            smooth_points.append(previous*factor+point*(1-factor))
        else:
            smooth_points.append(point)
    return smooth_points


smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.figure()
plt.plot(range(1, len(smooth_mae_history)+1), smooth_mae_history)

plt.xlabel('Epochs')
plt.ylabel('Val MAE')
plt.show()
