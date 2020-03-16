
# 用少量数据+数据增强训练CNN

## 引入相关依赖

    from keras.preprocessing import image
    import matplotlib.pyplot as plt
    from keras.preprocessing.image import ImageDataGenerator
    from keras import optimizers
    from keras import models, layers
    import os
    import shutil

## 整理数据

使用和鲸社区上下载的猫狗图片集  
下载链接：[cats and dogs](https://www.kesci.com/home/dataset/5d11bb1a38dc33002bd6f1f1)

从其中的train目录里提取部分样本

    original_dataset_dir = '/home/ylxiong/Documents/kaggle1902/train'

把他们放到自己的base_dir目录下

    base_dir = '/home/ylxiong/Documents/base_dir'
    os.mkdir(base_dir)

在base_dir目录下对猫和狗的数据分别创建以下文件夹：train, test, validation

    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, 'cats')
    os.mkdir(train_cats_dir)
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.mkdir(train_dogs_dir)

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.mkdir(validation_cats_dir)
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.mkdir(validation_dogs_dir)

    test_cats_dir = os.path.join(test_dir, 'cats')
    os.mkdir(test_cats_dir)
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(test_dogs_dir)

从原始目录开始复制数据至上述目录

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)


    fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

检查一下结果

    print("total training cat images:", len(os.listdir(train_cats_dir)))
    print("total training dog images:", len(os.listdir(train_dogs_dir)))
    print("total validation cat images:", len(os.listdir(validation_cats_dir)))
    print("total validation dog images:", len(os.listdir(validation_dogs_dir)))
    print("total test cat images:", len(os.listdir(test_cats_dir)))
    print("total test dog images:", len(os.listdir(test_dogs_dir)))

## 建立模型

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

看一看建好的模型

    model.summary()

编译模型

    model.compile(loss='binary_crossentropy',
                optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

## 定义数据增强器

对训练数据：

    train_datagen = ImageDataGenerator(
        rescale=1./255,                 # 正规化数据
        rotation_range=40,              # 旋转角度
        width_shift_range=0.2,          # 宽度变化
        height_shift_range=0.2,         # 高度变化
        shear_range=0.2,                # 切剪
        zoom_range=0.2,                 # 缩放比例
        horizontal_flip=True,           # 镜像反转
        fill_mode='nearest')            # 填充方式

对测试数据要保留原始！仅进行正规化

    test_datagen = ImageDataGenerator(rescale=1./255)

## 生成数据

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

## 训练模型，保存结果

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=50)

将训练好的模型保存为这个名字，方便日后调用

    model.save('cats_and_dogs_small_2.h5')

## 分析结果

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc)+1)

绘制图像

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.savefig('/root/my_code2242787668/ML_works/cat_or_dog/images/cat_dog_plt')

    plt.show()

结果如图：

![cat_or_dog_model](http://q5ioolwed.bkt.clouddn.com/cat_dog_plt.png)
