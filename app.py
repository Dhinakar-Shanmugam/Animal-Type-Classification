from flask import Flask, render_template, request, redirect, url_for
import os
from bson.objectid import ObjectId
from predict import predict_image
from database import collection

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_IMAGE = "default.png"


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("analyze.html", error="No file selected")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        result = predict_image(filepath)

        # 🔐 Duplicate check
        existing = collection.find_one({"imageHash": result["imageHash"]})
        if existing:
            return render_template(
                "analyze.html",
                error="⚠️ Duplicate Image! Already exists."
            )

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


# ✅ RECORDS + FILTER
@app.route("/records")
def records():

    animal = request.args.get("animal")
    quality = request.args.get("quality")

    query = {}

    if animal:
        query["animal"] = animal

    if quality:
        query["atc_tag"] = quality

    data = list(collection.find(query))

    # fallback image
    for rec in data:
        if "image_filename" not in rec:
            rec["image_filename"] = DEFAULT_IMAGE

    return render_template("records.html", records=data)


# ✅ DELETE
@app.route("/delete/<id>")
def delete_record(id):
    collection.delete_one({"_id": ObjectId(id)})
    return redirect(url_for("records"))


# ✅ VIEW DETAILS
@app.route("/view/<id>")
def view_record(id):

    record = collection.find_one({"_id": ObjectId(id)})

    return render_template(
        "result.html",
        result=record,
        image_filename=record.get("image_filename", DEFAULT_IMAGE),
        record_id=str(record["_id"])
    )


if __name__ == "__main__":
    app.run(debug=True)