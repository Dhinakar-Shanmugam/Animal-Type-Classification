from flask import Flask, render_template, request
import os

from predict import predict_image
from database import collection, save_result   # 🔥 import both

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    result = None
    image_path = None
    error = None   # 🔥 NEW

    if request.method == "POST":

        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]

        if file.filename == "":
            return "No file selected"

        if file:

            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            # 🔥 Predict
            result = predict_image(path)
            image_path = path

            image_hash = result["imageHash"]

            # 🔐 CHECK DUPLICATE
            existing = collection.find_one({"imageHash": image_hash})

            if existing:
                error = "⚠️ Duplicate Image! This animal already exists."
                result = None   # don't show result

            else:
                # ✅ Save only if new
                save_result(result)

    return render_template(
        "analyze.html",
        result=result,
        image_path=image_path,
        error=error   # 🔥 pass error
    )


@app.route("/records")
def records():

    data = list(collection.find())

    return render_template(
        "records.html",
        records=data
    )


if __name__ == "__main__":
    app.run(debug=True)