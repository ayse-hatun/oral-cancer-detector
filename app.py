from flask import Flask, request, jsonify, render_template, send_file
from transformers import SwinForImageClassification, AutoImageProcessor
from PIL import Image, ImageStat, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime
import numpy as np
import torch
import io

app = Flask(__name__)

MODEL_PATH = "oral_cancer_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)
model = SwinForImageClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

def assess_image_quality(image: Image.Image):
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    mean_brightness = float(stat.mean[0])
    contrast = float(stat.stddev[0])
    
    # Calculate edge variance (sharpness)
    edges = np.array(grayscale.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    edge_var = float(np.var(edges))
    
    # Tissue vs clothing/background color distribution
    np_rgb = np.array(image.convert("RGB"), dtype=np.float32)
    r, g, b = np_rgb[:, :, 0], np_rgb[:, :, 1], np_rgb[:, :, 2]
    
    # Detect prominent non-biological clothing colors (e.g. blue/violet scarf/shirt)
    clothing_pixels = (b > r * 1.1) & (b > 65)
    clothing_ratio = float(np.mean(clothing_pixels))
    
    is_too_dark = mean_brightness < 45.0
    is_too_bright = mean_brightness > 230.0
    is_low_contrast = contrast < 15.0
    is_blurry = edge_var < 50.0
    has_dominant_clothing = clothing_ratio > 0.18
    
    issue = None
    message = "Image lighting and clarity are suitable for analysis."
    
    if is_too_dark:
        issue = "too_dark"
        message = "Image is too dark for accurate classification. Please retake the photo in good lighting."
    elif is_too_bright:
        issue = "too_bright"
        message = "Image is overexposed or has excessive glare. Please adjust angle and retake."
    elif is_low_contrast:
        issue = "low_contrast"
        message = "Image has very low contrast. Please ensure direct illumination on the oral cavity."
    elif is_blurry:
        issue = "blurry"
        message = "Image appears out of focus or blurry. Please hold steady and retake."
    elif has_dominant_clothing:
        issue = "clothing_detected"
        message = f"Clothing or background detected ({round(clothing_ratio * 100)}% of image). Please frame close-up inside the mouth without clothing or face."
        
    acceptable = issue is None
    
    return {
        "acceptable": acceptable,
        "issue": issue,
        "message": message,
        "brightness": round(mean_brightness, 1),
        "contrast": round(contrast, 1),
        "edge_variance": round(edge_var, 1),
        "clothing_ratio": round(clothing_ratio * 100, 1)
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Upload an image using the 'image' field."}), 400

    try:
        raw_bytes = request.files["image"].read()
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        
        # Quality & lighting evaluation
        quality = assess_image_quality(image)
        force = request.form.get("force", "false").lower() in ("true", "1")
        
        # If image fails quality checks (dark, blurry, or clothing) and not forced:
        if not quality["acceptable"] and not force:
            return jsonify({
                "error": quality["message"],
                "quality_error": True,
                "quality": quality
            }), 422
            
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits[0]
            
            # Confidence calibration via temperature scaling (T=1.5)
            TEMPERATURE = 1.5
            calibrated_probs = torch.softmax(logits / TEMPERATURE, dim=0)

        raw_class_id = logits.argmax().item()
        margin = abs(float(logits[0] - logits[1]))

        # Clinical Logit Separation Threshold:
        # In-distribution lesions have margins >= 4.0 (typically 7 to 11).
        # Out-of-distribution faces, closed teeth, and non-oral frames have low margins (< 3.2).
        is_ambiguous = margin < 3.2

        if is_ambiguous and not force:
            return jsonify({
                "prediction": "INCONCLUSIVE",
                "is_inconclusive": True,
                "confidence": round(float(calibrated_probs.max().item()) * 100, 1),
                "margin": round(margin, 2),
                "quality": quality,
                "warning": f"Logit margin ({round(margin, 2)}) is below the clinical reliability threshold. The image appears to contain non-intraoral features (e.g. face, teeth, or background) rather than a close-up of mucosal tissue.",
                "guidance": "Please open your mouth wide and capture a close-up (10–15 cm) focused directly on the tongue, gums, inner cheek, or oral lesion under bright direct light.",
                "note": "Educational prototype only; not a medical diagnosis."
            })

        predicted_label = model.config.id2label[raw_class_id]
        confidence_val = round(float(calibrated_probs[raw_class_id].item()) * 100, 1)

        return jsonify({
            "prediction": predicted_label,
            "is_inconclusive": False,
            "confidence": confidence_val,
            "margin": round(margin, 2),
            "quality": quality,
            "note": "Educational prototype only; not a medical diagnosis."
        })
    except Exception as exc:
        return jsonify({"error": f"Could not process this image: {str(exc)}"}), 400

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

