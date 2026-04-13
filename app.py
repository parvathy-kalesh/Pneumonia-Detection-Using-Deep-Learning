from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load model
model = load_model("final_cnn_model.keras", compile=False)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Prediction function
def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return f"PNEUMONIA ({prediction*100:.2f}%)"
    else:
        return f"NORMAL ({(1-prediction)*100:.2f}%)"


# 🏠 Home page
@app.route("/")
def home():
    return render_template("home.html")


# 🩻 Prediction page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    img_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            result = predict_pneumonia(path)
            img_path = path

    return render_template("predict.html", result=result, img_path=img_path)


if __name__ == "__main__":
    app.run(debug=True)