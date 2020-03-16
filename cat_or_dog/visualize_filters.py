
import numpy as np
from keras.applications import VGG16
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt

# model = load_model('cats_and_dogs_small_2.h5')
model = VGG16(weights='imagenet', include_top=False)


# in case of getting a number out of range: (0,255)
# we need to preprocess the image

def deprocess_image(x):
    # s.t. mean = 0, stdErr = 0.1
    x -= x.mean()
    x /= (x.std()+1e-5)
    x *= 0.1
    # turn x into [0,1]
    x += 0.5
    x = np.clip(x, 0, 1)
    # turn x into RGB
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# def a function to visualize the filters

def generate_pattern(layer_name, filter_index, size=150):
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # return a list of tensor, in this case the length is 1
    grads = K.gradients(loss, model.input)[0]

    # normalization
    grads /= K.sqrt(K.mean(K.square(grads))) + 1e-5  # in case of 0

    iterate = K.function([model.input], [loss, grads])
    loss_value, grads_value = iterate([np.zeros((1, size, size, 3))])

    # stocastic gradient descent
    # from a noisy image
    input_img_data = np.random.random((1, size, size, 3))*20+128.

    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value*step

    img = input_img_data[0]
    return deprocess_image(img)


# plt.imshow(generate_pattern('conv2d_1', 0))
# plt.show()


size = 64
margin = 5
num = 8  # 5*5 filters
fig_size = num * size + (num-1)*margin

# layer_name = 'conv2d_1'


def save_pattern(layer_name):
    results = np.zeros((fig_size, fig_size, 3))
    for i in range(num):
        for j in range(num):
            filter_img = generate_pattern(layer_name, i+j*num, size=size)

            horizontal_start = i*size + i*margin
            horizontal_end = horizontal_start + size
            vertical_start = j*size + j*margin
            vertical_end = vertical_start + size
            results[horizontal_start:horizontal_end,
                    vertical_start:vertical_end, :] = filter_img

    plt.figure(figsize=(20, 20))
    results = np.clip(results, 0, 255).astype('uint8')
    plt.imshow(results)
    plt.savefig('/root/workingplace/my_code2242787668/'+layer_name+'_pattern')
    # plt.show()

for i in range(1,6):
    layer_name = 'block'+str(i)+'_conv1'
    save_pattern(layer_name)
    print(layer_name,' has been saved!')
