from flask import Flask, request, abort, jsonify
from datetime import datetime
import os
from src.preprocessing import image_preprocessing
from src.predict import predict
from src.fix_na import fix_na
from src.data_loading import get_imagenet, get_allowed_species

app = Flask(__name__)

UPLOADER_FOLDER = "request_images"

DEBUG = True


@app.route("/image", methods=["POST"])
def get_image():
    data = get_imagenet()
    allowed_species = get_allowed_species()

    print(dir(request))

    print(request.files)

    print(request.files.get("photo"))
    # request.files - json (dict) which we get from front
    # get method from dict
    image = request.files.get("photo", "")

    print(dir(image))

    try:
        filename = os.path.join(UPLOADER_FOLDER, f"{datetime.now().isoformat()}.jpg")
        image.save(filename)
        print("save completed")

        image_batch = image_preprocessing(filename)

        print("start predicting")
        labels = predict(image_batch)

        for label in labels:
            print(label)
            if label[1] not in allowed_species:
                continue
            result = {"category_name": label[1], "id": label[0], "probability": float(label[2])}
            row = data[data.category_name == label[1]].iloc[0]
            print(row)
            result["rus_name"] = fix_na(row.rus_name)
            result["photo_link"] = fix_na(row.photo_link)
            result["food"] = fix_na(row.food)
            result["description"] = fix_na(row.description)
            result["brief"] = fix_na(row.brief)
            return jsonify(result)

        return {
            "category_name": "undefined_bird",
            "id": "no001",
            "probability": 0,
            "rus_name": "курлык",
            "photo_link": "https://docs.google.com/spreadsheets/d/"
                          "16nyN8OfUAbsQEGavjEuQtZoSecxyyo5KoBR9ZzGThqI/edit#gid=571539848",
            "food": "подсушенный белый хлеб, зерна, крупы, семечки, овсяные хлопья",
            "description": "Человек приручил дикого сизого голубя более 5000 лет тому назад. С тех пор голубеводы "
                           "вывели более 800 пород домашних голубей, различных по цвету, форме тела и назначению.",
            "brief": "Мы не уверены, но возможно это Голубь!",
        }

    except Exception as ex:
        print("vse naebnulos")

        print(repr(ex))

        abort(400)


def main():
    os.makedirs(UPLOADER_FOLDER, exist_ok=True)
    app.run(
        port=7733,
        debug=DEBUG
    )


if __name__ == "__main__":
    main()
