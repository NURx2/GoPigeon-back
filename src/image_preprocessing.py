from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np


def image_preprocessing(image_path):
    # load an image in PIL format
    print(f"loading image from {image_path}")
    original = load_img(image_path, target_size=(224, 224))
    print('PIL image size', original.size)

    # convert the PIL image to a numpy array
    # IN PIL - image is in (width, height, channel)
    # In Numpy - image is in (height, width, channel)
    numpy_image = img_to_array(original)
    print('numpy array size', numpy_image.shape)

    # Convert the image / images into batch format
    # expand_dims will add an extra dimension to the data at a particular axis
    # We want the input matrix to the network to be of the form (batchsize, height, width, channels)
    # Thus we add the extra dimension to the axis 0.
    image_batch = np.expand_dims(numpy_image, axis=0)
    print('image batch size', image_batch.shape)

    return image_batch
