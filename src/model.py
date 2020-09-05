from keras.applications import vgg16

model = None


def load_model():
    global model
    # Load the VGG model
    model = vgg16.VGG16(weights='imagenet')


def get_model():
    if model is None:
        load_model()
    return model
