
# 用循环神经网络预测天气

## 下载数据

[耶拿天气数据](https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip)

## 导入相关依赖

```python
from keras import layers
from keras.models import Sequential
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
```

## 导入数据

原始数据第一栏是时间，第二栏是气压，第三栏才是温度

```python
data_dir = '/root/MySource/jena'
fname = os.path.join(data_dir, 'climate.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
float_data = np.zeros((len(lines), len(header) - 1))
# the first column of the data is Date
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
```

## 标准化数据

```python
# standardize the data
# take the first 200000 time steps as training data
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std
```

## 定义数据生成器

因为循环网络的数据有大量冗余，我们不一次性将数据导入内存，而是使用迭代器分别读取数据

```python
def generator(data, lookback, delay, min_index, max_index,
            shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
            # samples = np.zeros((len(rows),          # num of [input for the next procces]
            #                     lookback // step,   # num of previous data
            #                     data.shape[-1]))    # num of features
            # num of [output for the next procces]
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets
```

迭代器的数据是按下图方式产生的：

![illustrate](https://s1.ax1x.com/2020/05/12/YNud4s.png)

## 设置迭代器

```python
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=0,
                    max_index=200000,
                    shuffle=True,
                    step=step,
                    batch_size=batch_size)
val_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=200001,
                    max_index=300000,
                    step=step,
                    batch_size=batch_size)
test_gen = generator(float_data,
                    lookback=lookback,
                    delay=delay,
                    min_index=300001,
                    max_index=None,
                    step=step,
                    batch_size=batch_size)
# to see the whole dataset, how many times to sample from val_gen
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size
```

训练并保存LSTM模型

```python
##### train a LSTM with dropout #####
model = Sequential()
model.add(layers.LSTM(32,
                    dropout=0.1,
                    recurrent_dropout=0.5,
                    return_sequences=True,
                    input_shape=(None, float_data.shape[-1])))
model.add(layers.LSTM(64, activation='tanh',
                    dropout=0.1,recurrent_dropout=0.5))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                            steps_per_epoch=500,
                            epochs=40,
                            validation_data=val_gen,
                            validation_steps=val_steps)
model.save('/root/MySource/LSTM0.h5')
```

## 展示结果

```python
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./images/LSTM_mpl')
```

结果如下图：

![LSTM loss](https://s1.ax1x.com/2020/05/12/YNuUEQ.png)

## 另一种模型：GRU

GRU和LSTM相比少了一个控制门，训练更快  
只用把以上代码的所有“LSTM”改成“GRU”就行了  

结果如下图：

![GRU loss](https://s1.ax1x.com/2020/05/12/YNM6k4.png)

更快训练速度的代价是更大的loss。。。  

## 预测天气

获得两个模型的预测数据

```python
LSTM_pre = []
GRU_pre = []
y_real = []
for i in range(8):
    x_temp, y_temp = next(test_gen)
    LSTM_temp = LSTM_model.predict(x_temp)
    LSTM_pre.extend(LSTM_temp.tolist())
    GRU_temp = GRU_model.predict(x_temp)
    GRU_pre.extend(GRU_temp.tolist())
    y_real.extend(y_temp.tolist())
```

画图

```python
days = range(1, len(y_real)+1)
plt.figure()
plt.plot(days, y_real, 'black', alpha=1, label='Real Date')
plt.plot(days, LSTM_pre, 'r', alpha=0.6, label='LSTM Predict')
plt.plot(days, GRU_pre,'b',alpha=0.5, label='GRU Predict')
plt.title('Real And Predict Curve')
plt.xlabel("days")
plt.ylabel("temp")
plt.legend()
plt.savefig('./images/L_G_predict')
```

结果如下：

![LSTM & GRU](https://s1.ax1x.com/2020/05/12/YNuaNj.png)

感觉还不错哈哈哈哈
