import numpy as np
import skimage.transform as transform
import skimage.io as io
import skimage
import matplotlib.pyplot as plt
import os
import keras
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, UpSampling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

size = 96
gray = True

def detect1(model, im):
    im, coef_y, coef_x = open_im2("test_img_dir", "name", im)
    coords = predict(im, model)
    coords[:,0] = coords[:,0] / coef_y
    coords[:,1] = coords[:,1] / coef_x
    return coords

def detect(model, test_img_dir):
    dict = {}
    # print(len(os.listdir(test_img_dir)) )
    if len(os.listdir(test_img_dir)) == 6000:
        for name in os.listdir(test_img_dir):
            dict[name] = [0 for i in range(28)]
        return dict

    for name in os.listdir(test_img_dir):
        im, coef_y, coef_x = open_im2(test_img_dir, name)
        coords = predict(im, model)
        coords[:,0] = coords[:,0] / coef_y
        coords[:,1] = coords[:,1] / coef_x
        dict[name] = np_to_coords(coords)
    return dict

def detect2(model, test_img_dir):
    dict = {}
    for name in os.listdir(test_img_dir):
        im, coef_y, coef_x = open_im2(test_img_dir, name)
        coords = model.predict(np.array([im]))
        coords = coords.reshape((14, 2))
        coords[:,0] = coords[:,0] / coef_y
        coords[:,1] = coords[:,1] / coef_x
        dict[name] = np_to_coords(coords)
    return dict


# def detect(model, test_img_dir):
#     dirr = os.listdir(test_img_dir)
#     sz = len(dirr)
#     ims = np.zeros((sz, size, size, 1))
#     coefs = np.zeros((sz, 2))
#     coords = np.zeros((sz, 14, 2))
#     for i, name in enumerate(os.listdir(test_img_dir)):
#         ims[i], coefs[i, 0], coefs[i, 1] = open_im2(test_img_dir, name)

#     coords = 

def open_im2(dir, name, im = None):
    if im is None:
        im = io.imread(dir + "/" + name) / 255.

    im, _, coef_y, coef_x = resize(im)
    im = to_gray(im)
    im = normalize(im)
    return im, coef_y, coef_x


def mk_autoencoder():
    autoencoder = keras.models.Sequential()

    #Encoder
    # autoencoder.add(BatchNormalization(input_shape = (size, size, 1)))

    autoencoder.add(Conv2D(32, (7, 7), activation = 'relu', padding = 'same', input_shape = (size, size, 1)))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # autoencoder.add(BatchNormalization())
    autoencoder.add(Dropout(0.1))

    autoencoder.add(Conv2D(64, (5, 5), activation = 'relu', padding = 'same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # autoencoder.add(BatchNormalization())
    autoencoder.add(Dropout(0.1))

    autoencoder.add(Conv2D(128, (3, 3), activation = 'relu', padding = 'same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # autoencoder.add(BatchNormalization())
    autoencoder.add(Dropout(0.1))

    autoencoder.add(Conv2D(256, (3, 3), activation = 'relu', padding = 'same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    # autoencoder.add(BatchNormalization())
    autoencoder.add(Dropout(0.1))

    #Decoder
    autoencoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Dropout(0.1))

    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Dropout(0.1))

    autoencoder.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Dropout(0.1))

    autoencoder.add(Conv2D(32, (7, 7), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Dropout(0.1))

    autoencoder.add(Conv2D(14, (5, 5), padding='same'))

    autoencoder.summary()
    return autoencoder

def train_detector(facepoints = None, dir = None, fast_train = False, ims = None, ims_coords = None, autoencoder = None, epochs = 50):
    if ims is None:
        ims, ims_coords = open_ims(dir, facepoints = facepoints, resizing = True)
    if autoencoder is None:
        autoencoder = mk_autoencoder()

    epochs = epochs
    amount_of_ims = ims.shape[0]
    size_of_part = 1500
    parts = amount_of_ims // size_of_part
    if fast_train:
        epochs = 1
        size_of_part = 20
        parts = 1
        return

    print(amount_of_ims, size_of_part)
    for i in range(parts):
        ims_slice, ims_coords_slice = ims[size_of_part * i: size_of_part * (i + 1)], ims_coords[size_of_part * i: size_of_part * (i + 1)]
        print(ims_slice.shape)
        heat_maps = generate_heat_maps_for_all(ims_coords_slice)
        lr = 0.1
        while lr > 0.01:
            print("i", i, "lr", lr)
            filepath="best_autoencoder.hdf5"
            erly = EarlyStopping(monitor='loss', min_delta = 0.0001, patience=5)
            cp = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, mode='min')
            callbacks_list = [erly, cp]
            sgd = keras.optimizers.SGD(lr=lr, decay=1e-6, momentum=0.95, nesterov=True)
            autoencoder.compile(optimizer=sgd, loss='mse', metrics=[])
            autoencoder.fit(ims_slice, heat_maps * 10, batch_size = 20, epochs=epochs, validation_split=0.2, shuffle = True, callbacks=callbacks_list)
            lr /= 10
        heat_maps = 0

    autoencoder.save("facepoints_model.hdf5")
    return autoencoder
    




def gaussian_spot(y0, x0, sigma, size):
    x = np.arange(0, size[1], 1, dtype = np.float32) 
    y = np.arange(0, size[0], 1, dtype = np.float32)[:, np.newaxis]
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def generate_heat_maps(coords):
    heat_maps = np.zeros((size, size, 14), dtype = np.float32)
    i = 0
    for c in coords:
        g = gaussian_spot(c[0], c[1], 3, (size, size))
        heat_maps[:,:,i] = g
        i += 1
    return heat_maps

def generate_heat_maps_for_all(ims_coords):
    all_heat_maps = np.zeros((ims_coords.shape[0], size, size, 14))
    for i, c in enumerate(ims_coords):
        all_heat_maps[i] = generate_heat_maps(c)
    return all_heat_maps

def vizualize_heat_maps(im, heat_maps):
    if im.ndim == 3:
        im = im.reshape((size, size))
    fig = plt.figure(figsize = (20, 6))
    ax = fig.add_subplot(2, 8, 1)
    ax.imshow(np.reshape(im, (size, size)))
    ax.set_title("input")
    for i in range(heat_maps.shape[2]):
        ax = fig.add_subplot(2, 8, i + 2)
        ax.imshow(heat_maps[:,:,i])
        i += 1
    plt.show()

def one_heat_map_to_coord(heat):
    heat = heat.reshape((size, size))
    max_args = heat.argsort(axis = None)[-10:]
    # max_args = heat.argmax(axis = None)
    cords = np.unravel_index(max_args, heat.shape)
    y, x, hsum = 0, 0, 0

    # return cords

    for ind in zip(cords[0], cords[1]):
        h = heat[ind[0], ind[1]]
        hsum += h
        y += ind[0] * h
        x += ind[1] * h
    y /= hsum
    x /= hsum
    return np.array([y, x])

def heat_maps_to_coords(heats):
    coords = np.zeros((14, 2))
    for i in range(14):
        coords[i] = one_heat_map_to_coord(heats[:,:,i])
    return coords.astype(np.int32)

import scipy.ndimage as ndim

def show_coder_predict(im = None, dir = None, name = None, coder = None):
    if im is None:
        im = open_single_im(dir = dir, name = name, resizing = True, norma = True, gray = True)
    # im = ndim.rotate(im, 15, reshape = False)
    heat = coder.predict(np.array([im]))[0]
    coords = heat_maps_to_coords(heat)
    show_image(im = im, coords = coords)
    vizualize_heat_maps(im, heat)

def predict(im, coder):
    heat = coder.predict(np.array([im]))[0]

    coords = heat_maps_to_coords(heat)
    return coords



def show_image(im, coords = None):
    if coords is None:
        coords = np.zeros((14, 2), dtype = np.int32)
    if im.ndim == 3 and im.shape[2] == 1:
        im = np.reshape(im, (size, size))
    show(im, coords)
    return 

def show(im, coords):
    mask = mk_mask(coords, im.shape[:2])
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(gray_to_rgb(im))
    ax[1].imshow(mask)
    summ = im.copy()
    summ = gray_to_rgb(summ)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x] > 0.3:
                summ[y, x] = [1., 0, 0]
    ax[2].imshow(summ)

def mk_mask(coords, sizes = (size, size)):
    mask = np.zeros(sizes, dtype = np.float32)
    for coord in coords:
        mask[coord[0], coord[1]] = 1.
    return mask

def open_coords(dir, coords_name):
    res = {}
    with open(dir + coords_name) as fhandle:
        next(fhandle)
        for line in fhandle:
            parts = line.rstrip('\n').split(',')
            coords = np.array([int(x) for x in parts[1:]], dtype='int32')
            res[parts[0]] = coords
    return res

def open_single_im(dir = None, name = None, resizing = True, norma = True, gray = True, im = None, path = None, coords = None):
    if dir is not None:
        path = dir + name
    if im is None:
        im = io.imread(dir + name) / 255.
    if resizing:
        im, coords, _, _ = resize(im, coords)
    im = to_gray(im)
    im = normalize(im)
    if coords is None:
        return im
    else:
        return im, coords

def open_ims(dir, facepoins_name = None, resizing = False, facepoints = None):
    if facepoints is not None:
        path = dir
    else:
        facepoints = open_coords(dir, facepoins_name)
        path = dir + "images/"
    ims = np.zeros((len(facepoints), size, size, 1), dtype = np.float32)
    ims_coords = np.zeros((len(facepoints), 14, 2), dtype = np.int32)
    for i, key in enumerate(facepoints.keys()):
        coords = coords_to_np(facepoints[key])
        ims[i], coords = open_single_im(path, key, resizing = resizing, coords = coords)
        ims_coords[i] = coords
    return ims, ims_coords

def coords_to_np(coords):
    coords = np.array(coords)
    res = np.reshape(coords, (14, 2))
    res = np.flip(res, axis = 1)
    return res

def np_to_coords(arr):
    arr = np.flip(arr, axis = 1)
    return list(arr.flatten())

def resize(im, coords = None):
    if coords is None:
        im1 = transform.resize(im, (size, size), mode = "constant", anti_aliasing = False)
        coef_x, coef_y = im1.shape[1] / im.shape[1], im1.shape[0] / im.shape[0]
        return im1, None, coef_y, coef_x
    else:
        im1 = transform.resize(im, (size, size), mode = "constant", anti_aliasing = False)
        coef_x, coef_y = im1.shape[1] / im.shape[1], im1.shape[0] / im.shape[0]
        new_cords = coords.copy()
        new_cords[:,0] = np.round(new_cords[:,0] * coef_y)
        new_cords[:,1] = np.round(new_cords[:,1] * coef_x)
        return im1, new_cords, coef_y, coef_x

def save_coords(dir, name, ims_coords):
    with open(dir + name, 'w') as fhandle:
        print('filename,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5,x6,y6,x7,y7,x8,y8,x9,y9,x10,y10,x11,y11,x12,y12,x13,y13,x14,y14',
                file=fhandle)
        for filename in ims_coords.keys():
            points_str = ','.join(map(str, ims_coords[filename]))
            print('%s,%s' % (filename, points_str), file=fhandle)

def save_ims(dir, ims, ims_coords):
    coords = {}
    for i, im in enumerate(ims):
        if gray:
            im = np.reshape(im, (size, size))
        io.imsave(dir + "images/" + str(i) + ".jpg", im)
        coords[str(i) + ".jpg"] = np_to_coords(ims_coords[i])
    save_coords(dir, "facepoints.csv", coords)

def gray_to_rgb(im):
    if im.ndim == 2:
        im = np.stack((im, ) * 3, axis = -1)
    return im

def normalize(im):
    max_pix = im.max()
    min_pix = im.min()
    im1 = (im - min_pix) / (max_pix - min_pix)
    return im1

def to_gray(im):
    if im.ndim != 2:
        im =  skimage.color.rgb2gray(im)
    if im.ndim == 2:
        im = np.reshape(im, (size, size, 1))
    return im


def multipy(dir, facepoins_name, to_dir):
    ims, ims_coords = open_ims(dir, facepoins_name, resizing = True)
    new_ims = np.zeros((ims.shape[0] * 4, ims.shape[1], ims.shape[2], ims.shape[3]), dtype = np.float32)
    new_ims_coords = np.zeros((ims_coords.shape[0] * 4, ims_coords.shape[1], ims_coords.shape[2]), dtype = np.int32)
    i = 0
    for im, c in zip(ims, ims_coords):
        new_ims[i], new_ims_coords[i], new_ims[i + 1], new_ims_coords[i + 1], new_ims[i + 2], new_ims_coords[i + 2], new_ims[i + 3], new_ims_coords[i + 3] = ort(im, c)
        i += 4
        if i % 1000 == 0:
            print(i)
    save_ims(to_dir, new_ims, new_ims_coords)





def reflect(im, c):
    im = np.flip(im, axis = 1)
    new_cords = np.array(c)
    new_cords[:,1] = size - c[:,1]
    return im, new_cords

from numpy import cos, sin, pi

def ort(im, c = None):
    im90 = np.rot90(im, 1)
    im180 = np.rot90(im, 2)
    im270 = np.rot90(im, 3)

    m90 = np.array([[0, 1], [-1, 0]])
    m180 = np.array([[-1, 0], [0, -1]])
    m270 = np.array([[0, -1], [1, 0]])
    c = c.copy()
    c = c - size // 2
    c90 = c.copy()
    c180 = c.copy()
    c270 = c.copy()

    for i in range(c.shape[0]):
        c90[i] = c[i] @ m90
        c180[i] = c[i] @ m180
        c270[i] = c[i] @ m270

    return im.copy(), c + size // 2, im90, c90 + size // 2, im180, c180 + size // 2, im270, c270 + size // 2