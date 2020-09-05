from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from .model import get_model


def predict(image_batch):
    processed_image = vgg16.preprocess_input(image_batch.copy())

    # get the predicted probabilities for each class
    predictions = get_model().predict(processed_image)
    # print predictions

    # convert the probabilities to class labels
    # We will get top 5 predictions which is the default
    label = decode_predictions(predictions, top=10)[0]

    print(f"predicted labels - {label}")

    return label
