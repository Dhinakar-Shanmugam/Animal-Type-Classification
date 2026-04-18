from flask import Flask, render_template, request, redirect, url_for
import os
from bson.objectid import ObjectId

# ML + DB
from predict import predict_image
from database import collection, users_collection

# AUTH
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = "secret123"

# Upload
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEFAULT_IMAGE = "default.png"

# ================= LOGIN =================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, user):
        self.id = str(user["_id"])
        self.username = user["username"]


@login_manager.user_loader
def load_user(user_id):
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    return User(user) if user else None


# ================= AUTH =================
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        if users_collection.find_one({"username": username}):
            return render_template("signup.html", error="User already exists")

        users_collection.insert_one({
            "username": username,
            "password": password
        })

        return redirect(url_for("login"))

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users_collection.find_one({"username": username})

        if user and check_password_hash(user["password"], password):
            login_user(User(user))
            return redirect(url_for("dashboard"))

        return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# ================= DASHBOARD =================
@app.route("/")
@login_required
def dashboard():

    data = list(collection.find({"user_id": current_user.id}))

    total = len(data)

    avg_score = round(
        sum(d["atc"]["Total Score"] for d in data) / total, 2
    ) if total else 0

    cattle = sum(1 for d in data if d["animal"] == "cattle")
    buffalo = sum(1 for d in data if d["animal"] == "buffalo")

    return render_template(
        "dashboard.html",
        total=total,
        avg_score=avg_score,
        cattle=cattle,
        buffalo=buffalo,
        data=data
    )


# ================= ANALYZE =================
@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():

    if request.method == "POST":

        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("analyze.html", error="No file selected")

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        result = predict_image(filepath)

        # 🔐 Duplicate check (per user)
        existing = collection.find_one({
            "imageHash": result["imageHash"],
            "user_id": current_user.id
        })

        if existing:
            return render_template(
                "analyze.html",
                error="⚠️ Duplicate Image! Already exists."
            )

        # Attach user
        result["user_id"] = current_user.id

        inserted = collection.insert_one(result)
        record_id = str(inserted.inserted_id)

        return render_template(
            "result.html",
            result=result,
            image_filename=result["image_filename"],
            record_id=record_id
        )

    return render_template("analyze.html")


# ================= RECORDS =================
@app.route("/records")
@login_required
def records():

    animal = request.args.get("animal")
    quality = request.args.get("quality")

    query = {"user_id": current_user.id}

    if animal:
        query["animal"] = animal

    if quality:
        query["atc_tag"] = quality

    data = list(collection.find(query))

    for rec in data:
        if "image_filename" not in rec:
            rec["image_filename"] = DEFAULT_IMAGE

    return render_template("records.html", records=data)


# ================= DELETE =================
@app.route("/delete/<id>")
@login_required
def delete_record(id):

    collection.delete_one({
        "_id": ObjectId(id),
        "user_id": current_user.id
    })

    return redirect(url_for("records"))


# ================= VIEW =================
@app.route("/view/<id>")
@login_required
def view_record(id):

    record = collection.find_one({
        "_id": ObjectId(id),
        "user_id": current_user.id
    })

    return render_template(
        "result.html",
        result=record,
        image_filename=record.get("image_filename", DEFAULT_IMAGE),
        record_id=str(record["_id"])
    )


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)