
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')
save_fig_dict = '/home/ylxiong/Documents/'

# model.summary()

img_path = '/home/ylxiong/Documents/base_dir/test/cats/cat.1777.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
# tensor shape: (1, 150, 150, 3)
img_tensor /= 255.

# plt.imshow(img_tensor[0])
# plt.show()


# extract outputs from the first 8 layers
layer_outputs = [layer.output for layer in model.layers[:8]]

# def a model, which return a certain layer's output(given input)
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

# return 8 numpy lists(the output of\
# each activation layer(given a certain image))
# as the following shape:
# (n'th img(i.e.,1), width, height, n'th channel)
activations = activation_model.predict(img_tensor)


# # show one image
# plt.matshow(activations[0][0, :, :, 2], cmap="plasma")
# plt.show()


# show all the images

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
