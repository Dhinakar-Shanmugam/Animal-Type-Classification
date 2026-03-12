from flask import Flask, render_template, request
import os

from predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image_path = None

    if request.method == "POST":

        # Check file exists in request
        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]

        # Check file selected
        if file.filename == "":
            return "No file selected"

        if file:

            # Save file
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            print("Saved image:", image_path)   # Debug line

            # Predict
            result = predict_image(image_path)

    return render_template(
        "index.html",
        result=result,
        image_path=image_path
    )


if __name__ == "__main__":
    app.run(debug=True)