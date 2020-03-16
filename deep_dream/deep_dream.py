from keras.preprocessing import image
import scipy  # scipy==1.2.1
import numpy as np
from keras import backend as K
from keras.applications import inception_v3

model = inception_v3.InceptionV3(weights='imagenet', include_top=False)

# disable training
K.set_learning_phase(0)

layer_contributions = {
    'mixed5': 2,
    'mixed6': 3.,
    'mixed7': 2.,
    'mixed8': 0.2,
}

layer_dict = dict([(layer.name, layer) for layer in model.layers])

loss = 0.

for layer_name in layer_contributions:
    coeff = layer_contributions[layer_name]
    activation = layer_dict[layer_name].output

    scaling = K.prod(K.cast(K.shape(activation), 'float32'))
    # prevent the shadow at edges
    loss += coeff*K.sum(K.square(activation[:, 2:-2, 2:-2, :]))/scaling


dream = model.input

grads = K.gradients(loss, dream)[0]

grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grads_value = outs[1]
    return loss_value, grads_value


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grads_value = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('...loss value at ', i, ': ', loss_value)
        x += step*grads_value
    return x


step = 0.01
num_octave = 3
octave_scale = 1.4
iterations = 20

# if loss is too large, stop training
max_loss = 10.

base_image_dir = './images/me.jpg'


def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


img = preprocess_image(base_image_dir)

# zoom the image while layers goes lower
original_shape = img.shape[1:3]
succesive_shape = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim/(octave_scale**i))
                   for dim in original_shape])
    succesive_shape.append(shape)

# reverse the list
succesive_shape = succesive_shape[::-1]


def resize_img(img, size):
    img = np.copy(img)
    factors = (1, float(size[0])/img.shape[1],
               float(size[1])/img.shape[2], 1)
    return scipy.ndimage.zoom(img, factors, order=1)


def save_img(img, fname):
    pil_img = deprocess_image(np.copy(img))
    scipy.misc.imsave(fname, pil_img)


def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


original_img = np.copy(img)
shrunk_original_img = resize_img(img, succesive_shape[0])

for shape in succesive_shape:
    print('Proccesing image shape: ', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img, iterations=iterations,
                          step=step, max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)
    # save_img(img, fname='./images/dream_at_scale_'+str(shape)+'.png')

save_img(img, fname='./images/me_dream.png')
