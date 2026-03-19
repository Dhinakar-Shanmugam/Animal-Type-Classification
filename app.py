from flask import Flask, render_template, request, redirect, url_for
import os
from bson.objectid import ObjectId
from predict import predict_image
from database import collection

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Default placeholder image
DEFAULT_IMAGE = "default.png"
default_image_path = os.path.join(app.config["UPLOAD_FOLDER"], DEFAULT_IMAGE)
if not os.path.exists(default_image_path):
    # Create a gray placeholder if it doesn't exist
    from PIL import Image
    img = Image.new("RGB", (300, 300), color=(200, 200, 200))
    img.save(default_image_path)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded"
        file = request.files["file"]
        if file.filename == "":
            return "No file selected"
        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            # Model prediction
            result = predict_image(path)
            image_hash = result["imageHash"]

            # Duplicate check
            existing = collection.find_one({"imageHash": image_hash})
            if existing:
                return render_template(
                    "analyze.html",
                    error="⚠️ Duplicate Image! Already exists."
                )

            # Save only filename for frontend
            result["image_filename"] = file.filename

            # Add date_time if not present
            if "date_time" not in result:
                from datetime import datetime
                result["date_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save to DB
            inserted = collection.insert_one(result)
            record_id = str(inserted.inserted_id)

            return render_template(
                "result.html",
                result=result,
                image_filename=result["image_filename"],
                record_id=record_id
            )

    return render_template("analyze.html")


# Records page
@app.route("/records")
def records():
    data = list(collection.find())
    # Ensure all records have 'image_filename'
    for rec in data:
        if "image_filename" not in rec or not rec["image_filename"]:
            rec["image_filename"] = DEFAULT_IMAGE
    return render_template("records.html", records=data)


# Delete record
@app.route("/delete/<id>")
def delete_record(id):
    collection.delete_one({"_id": ObjectId(id)})
    return redirect(url_for("records"))


# View record details
@app.route("/view/<id>")
def view_record(id):
    record = collection.find_one({"_id": ObjectId(id)})
    # Safe access with fallback
    image_filename = record.get("image_filename", DEFAULT_IMAGE)
    return render_template(
        "result.html",
        result=record,
        image_filename=image_filename,
        record_id=str(record["_id"])
    )


if __name__ == "__main__":
    app.run(debug=True)