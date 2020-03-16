
# 将卷基层识别的结果可视化

## 引入相关资源

    from keras import models
    import matplotlib.pyplot as plt
    import numpy as np
    from keras.preprocessing import image
    from keras.models import load_model

## 载入模型和图片

本文用的模型和数据均参见[这篇文章](http://101.133.217.104/%e7%94%a8%e5%b0%91%e9%87%8f%e6%95%b0%e6%8d%ae%e6%95%b0%e6%8d%ae%e5%a2%9e%e5%bc%ba%e8%ae%ad%e7%bb%83cnn/)

载入模型

    model = load_model('cats_and_dogs_small_2.h5')
    save_fig_dict = '/home/ylxiong/Documents/'

导入图片

    img_path = '/home/ylxiong/Documents/base_dir/test/cats/cat.1777.jpg'
    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # tensor shape: (1, 150, 150, 3)
    img_tensor /= 255.

可以用matplotlib看一看载入的图像

    plt.imshow(img_tensor[0])
    plt.show()

结果应该是这样：

![cat1777](http://q5ioolwed.bkt.clouddn.com/cat1777_mpl.png)

## 显示模型输出

提取每一层的模型

    # extract outputs from the first 8 layers
    layer_outputs = [layer.output for layer in model.layers[:8]]

定义获取模型的输出的方法

    # def a model, which return a certain layer's output(given input)
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

获得上述方法对于给定图片的输出

    # return 8 numpy lists(the output of\
    # each activation layer(given a certain image))
    # as the following shape:
    # (n'th img(i.e.,1), width, height, n'th channel)
    activations = activation_model.predict(img_tensor)

## 展示图片

先画一张大图，把每个neuron的输出画在对应层上

    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)
    # print(layer_names)
    # result :
    # ['conv2d_1', 'max_pooling2d_1', 'conv2d_2', 'max_pooling2d_2',
    #     'conv2d_3', 'max_pooling2d_3', 'conv2d_4', 'max_pooling2d_4']

    img_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):
        # shape is (1, size, size, channel)
        n_features = layer_activation.shape[-1]

        size = layer_activation.shape[1]

        n_cols = n_features // img_per_row
        # init a full 0 canvas
        display_grid = np.zeros((size*n_cols, img_per_row*size))

        for col in range(n_cols):
            for row in range(img_per_row):
                channel_image = layer_activation[0, :, :, col*img_per_row+row]
                # beautify the image
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                # print the image on the canvas
                display_grid[col*size:(col+1)*size,
                            row*size:(row+1)*size] = channel_image
        scale = 1./size
        plt.figure(figsize=(scale*display_grid.shape[1],
                            scale*display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='plasma')
        plt.savefig(save_fig_dict+layer_name)

    plt.show()

结果应该类似这样：

![results](http://q5ioolwed.bkt.clouddn.com/cat_dog.png)
