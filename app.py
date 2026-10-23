from flask import Flask, request, jsonify, render_template, send_file
from transformers import SwinForImageClassification, AutoImageProcessor
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import torch
import io

app = Flask(__name__)

MODEL_PATH = "oral_cancer_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = SwinForImageClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image using the 'image' field."}), 400

    try:
        image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            probabilities = torch.softmax(model(**inputs).logits, dim=1)[0]

        class_id = probabilities.argmax().item()

        return jsonify({
            "prediction": model.config.id2label[class_id],
            "confidence": round(probabilities[class_id].item() * 100, 2),
            "note": "Educational prototype only; not a medical diagnosis."
        })
    except Exception:
        return jsonify({"error": "Could not process this image."}), 400

@app.route("/download-report", methods=["POST"])
def download_report():
    prediction = request.form.get("prediction", "Not provided")
    confidence = request.form.get("confidence", "Not provided")

    pdf_buffer = io.BytesIO()
    pdf = canvas.Canvas(pdf_buffer, pagesize=A4)

    pdf.setTitle("Oral Image Classification Report")

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 790, "Oral Image Classification Report")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, 750, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pdf.drawString(50, 720, f"Prediction: {prediction}")
    pdf.drawString(50, 695, f"Model confidence: {confidence}%")

    pdf.drawString(50, 650, "Purpose: Educational image classification only.")
    pdf.drawString(50, 630, "This result is not a medical diagnosis.")
    pdf.drawString(50, 610, "Consult a qualified healthcare professional for concerns.")

    pdf.save()
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="oral_image_report.pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)

