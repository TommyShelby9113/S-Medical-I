from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import openai
import os

app = Flask(__name__)

# 🔑 OpenAI API Key
openai.api_key = "YOUR_OPENAI_API_KEY"

# 🧠 Load CNN Model
model = tf.keras.models.load_model("model/skin_cancer_cnn.h5")

def predict_skin_cancer(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]
    confidence = round(prediction * 100, 2)

    if prediction > 0.5:
        return "مشکوک به سرطان پوست", confidence
    else:
        return "خوش‌خیم", 100 - confidence


def medical_chat(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional medical AI assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


@app.route("/", methods=["GET", "POST"])
def index():
    result = confidence = chat_response = None

    if request.method == "POST":
        if "skin_image" in request.files:
            image = request.files["skin_image"]
            path = "static/upload.jpg"
            image.save(path)
            result, confidence = predict_skin_cancer(path)

        if "chat_prompt" in request.form:
            chat_response = medical_chat(request.form["chat_prompt"])

    return render_template("index.html",
                           result=result,
                           confidence=confidence,
                           chat_response=chat_response)


if __name__ == "__main__":
    app.run(debug=True)
