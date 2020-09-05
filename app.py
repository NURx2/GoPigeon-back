from flask import Flask, request, abort, jsonify
from datetime import datetime
import os
from src.preprocessing import image_preprocessing
from src.prediction import predict
from src.helpers import fix_na
import pandas as pd

app = Flask(__name__)

UPLOADER_FOLDER = "request_images"

DEBUG = False

ALLOWED_SPECIES = ['insert', 'values']  # TODO: fill it


@app.route("/image", methods=["POST"])
def get_image():
    print(dir(request))

    print(request.files)

    print(request.files.get("photo"))
    # request.files - json (dict) which we get from front
    # get method from dict
    image = request.files.get("photo", "")

    print(dir(image))

    try:
        if DEBUG:
            filename = os.path.join(UPLOADER_FOLDER, "2019-10-26T22:13:18.942507.jpg")
        else:
            filename = os.path.join(UPLOADER_FOLDER, f"{datetime.now().isoformat()}.jpg")

            image.save(filename)

        print("save completed")

        image_batch = image_preprocessing(filename)

        print("start predicting")
        labels = predict(image_batch)

        print("reading csv")
        data = pd.read_csv("imagenet.csv").fillna("")

        for label in labels:
            if label not in ALLOWED_SPECIES:
                continue
            print(label)
            result = {"class_name": label[0], "tag": label[1], "probability": float(label[2])}
            row = data[data.class_id == label[0]].iloc[0]
            print(row)
            result["rus_name"] = fix_na(row.rus_name)
            result["photo_link"] = fix_na(row.photo_link)
            result["food"] = fix_na(row.food)
            result["size"] = fix_na(row.size)
            result["description"] = fix_na(row.description)
            return jsonify(result)

        return {
            "class_name": "",
            "tag": "",
            "probability": "",
            "rus_name": "",
            "photo_link": "",
            "food": "",
            "size": "",
            "description": "",
        }

    except Exception as ex:
        print("vse naebnulos")

        print(repr(ex))

        abort(400)


def main():
    os.makedirs(UPLOADER_FOLDER, exist_ok=True)
    app.run(
        port=7733
    )


if __name__ == "__main__":
    main()
