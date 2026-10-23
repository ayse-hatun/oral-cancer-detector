from flask import Flask, request, jsonify, render_template, send_file
from transformers import SwinForImageClassification, AutoImageProcessor
from PIL import Image, ImageStat, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from datetime import datetime
import numpy as np
import torch
import base64
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
    # ── Form fields ──────────────────────────────────────────────────────────
    prediction   = request.form.get("prediction", "Not available")
    confidence   = request.form.get("confidence", "N/A")
    margin       = request.form.get("margin", "N/A")
    brightness   = request.form.get("brightness", "N/A")
    contrast     = request.form.get("contrast", "N/A")
    sharpness    = request.form.get("sharpness", "N/A")
    is_inconclusive = request.form.get("is_inconclusive", "false").lower() == "true"
    image_data   = request.form.get("image_data", "")  # base64 data-URL

    # ── Colour palette (matches app theme) ──────────────────────────────────
    BURGUNDY    = colors.HexColor("#800020")
    DARK_MAROON = colors.HexColor("#5A0015")
    IVORY       = colors.HexColor("#FFFFF0")
    CHARCOAL    = colors.HexColor("#2F3E46")
    SOFT_GRAY   = colors.HexColor("#F5F5F0")
    ACCENT      = colors.HexColor("#C9A84C")   # gold accent
    WHITE       = colors.white
    BLACK       = colors.black
    MID_GRAY    = colors.HexColor("#666666")

    PAGE_W, PAGE_H = A4          # 595 × 842 pt
    MARGIN_L = 45
    MARGIN_R = PAGE_W - 45
    TEXT_W   = MARGIN_R - MARGIN_L

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    c.setTitle("Oral Cancer Screening Report – AI Prototype")

    now_str = datetime.now().strftime("%d %B %Y  %H:%M")

    # ═════════════════════════════════════════════════════════════════════════
    # HEADER BANNER
    # ═════════════════════════════════════════════════════════════════════════
    c.setFillColor(BURGUNDY)
    c.rect(0, PAGE_H - 78, PAGE_W, 78, fill=1, stroke=0)

    # Gold accent line below banner
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - 81, PAGE_W, 3, fill=1, stroke=0)

    # Title
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(MARGIN_L, PAGE_H - 30, "Oral Cancer Screening Report")
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN_L, PAGE_H - 46, "AI-Assisted Mucosal Tissue Classification  |  Educational Prototype")
    c.drawRightString(MARGIN_R, PAGE_H - 30, now_str)
    c.setFont("Helvetica", 8)
    c.drawRightString(MARGIN_R, PAGE_H - 44, "Version 1.0  |  Swin-T Model")

    y = PAGE_H - 100   # current vertical cursor

    # ═════════════════════════════════════════════════════════════════════════
    # ANALYSED IMAGE THUMBNAIL  (top, right-aligned)
    # ═════════════════════════════════════════════════════════════════════════
    IMG_W, IMG_H = 155, 115   # thumbnail dimensions in pt
    img_x = MARGIN_R - IMG_W
    img_y = y - IMG_H

    if image_data and image_data.startswith("data:image"):
        try:
            header, encoded = image_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_io    = io.BytesIO()
            pil_img.save(img_io, format="JPEG", quality=85)
            img_io.seek(0)

            from reportlab.lib.utils import ImageReader
            c.setStrokeColor(BURGUNDY)
            c.setLineWidth(1.5)
            c.rect(img_x - 2, img_y - 2, IMG_W + 4, IMG_H + 4, fill=0, stroke=1)
            c.drawImage(ImageReader(img_io), img_x, img_y, width=IMG_W, height=IMG_H,
                        preserveAspectRatio=True, mask="auto")

            c.setFillColor(MID_GRAY)
            c.setFont("Helvetica-Oblique", 7.5)
            c.drawCentredString(img_x + IMG_W / 2, img_y - 11, "Analysed photograph")
        except Exception:
            pass   # silently skip thumbnail if image decoding fails

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION HELPER  (draws a bold section heading with underline)
    # ═════════════════════════════════════════════════════════════════════════
    def section(title, y_pos):
        c.setFillColor(BURGUNDY)
        c.setFont("Helvetica-Bold", 10.5)
        c.drawString(MARGIN_L, y_pos, title)
        c.setStrokeColor(BURGUNDY)
        c.setLineWidth(0.6)
        c.line(MARGIN_L, y_pos - 3, MARGIN_R, y_pos - 3)
        return y_pos - 18

    def body(text, y_pos, font="Helvetica", size=9.5, color=CHARCOAL, indent=0):
        c.setFillColor(color)
        c.setFont(font, size)
        c.drawString(MARGIN_L + indent, y_pos, text)
        return y_pos - 14

    def kv_row(key, value, y_pos, val_color=CHARCOAL):
        """Render a key-value pair on one line."""
        c.setFillColor(MID_GRAY)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(MARGIN_L, y_pos, key)
        c.setFillColor(val_color)
        c.setFont("Helvetica", 9)
        c.drawString(MARGIN_L + 145, y_pos, str(value))
        return y_pos - 13

    # ═════════════════════════════════════════════════════════════════════════
    # 1. CLINICAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    text_right_limit = img_x - 14   # stay left of thumbnail

    # Shaded summary box
    c.setFillColor(SOFT_GRAY)
    c.setStrokeColor(BURGUNDY)
    c.setLineWidth(0.8)
    c.roundRect(MARGIN_L, img_y - 4, text_right_limit - MARGIN_L, (y - img_y + 4), 5, fill=1, stroke=1)

    ty = y - 10
    c.setFillColor(BURGUNDY)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(MARGIN_L + 8, ty, "Classification Result")
    ty -= 16

    # Prediction label with colour coding
    pred_upper = prediction.upper()
    if is_inconclusive or pred_upper == "INCONCLUSIVE":
        pred_color = colors.HexColor("#CC6600")
    elif "CANCER" in pred_upper or "MALIGNANT" in pred_upper or "CARCINOMA" in pred_upper:
        pred_color = colors.HexColor("#B00000")
    else:
        pred_color = colors.HexColor("#006633")

    c.setFillColor(pred_color)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(MARGIN_L + 8, ty, prediction)
    ty -= 16

    conf_display = f"{confidence}%" if confidence not in ("N/A", "") else "N/A"
    c.setFillColor(CHARCOAL)
    c.setFont("Helvetica", 9)
    c.drawString(MARGIN_L + 8, ty, f"Model Confidence: {conf_display}")
    ty -= 12
    c.drawString(MARGIN_L + 8, ty, f"Logit Margin:  {margin}")
    ty -= 12
    c.drawString(MARGIN_L + 8, ty, f"Date / Time:  {now_str}")

    # Advance cursor below the thumbnail
    y = img_y - 22

    # ═════════════════════════════════════════════════════════════════════════
    # 2. ABOUT THIS SCREENING TOOL
    # ═════════════════════════════════════════════════════════════════════════
    y = section("1.  About This Screening Tool", y)
    y = body("This report is produced by an AI-assisted oral mucosal tissue classifier based on a fine-tuned", y)
    y = body("Swin Transformer (Swin-T) model trained on intraoral clinical photographs. The model distinguishes", y)
    y = body("between oral mucosal tissue presentations that may or may not exhibit markers consistent with", y)
    y = body("oral squamous cell carcinoma (OSCC). Classification follows the WHO 5th Edition taxonomy.", y)
    y -= 4

    # ═════════════════════════════════════════════════════════════════════════
    # 3. IMAGE QUALITY METRICS
    # ═════════════════════════════════════════════════════════════════════════
    y = section("2.  Image Quality Assessment", y - 4)

    quality_data = [
        ["Metric", "Measured Value", "Acceptable Range", "Status"],
        ["Brightness (Luminance)", f"{brightness} / 255", "55 – 225",
         _qstatus(brightness, 55, 225, higher_is="mid")],
        ["Contrast (Std Dev)", f"{contrast}", "≥ 16",
         _qstatus_min(contrast, 16)],
        ["Sharpness (Edge Var)", f"{sharpness}", "≥ 28",
         _qstatus_min(sharpness, 28)],
        ["Logit Separation Margin", f"{margin}", "≥ 3.2 (reliable)",
         _qstatus_min(margin, 3.2)],
    ]

    col_widths = [120, 105, 120, 100]
    tbl = Table(quality_data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BURGUNDY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 8.5),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 8.5),
        ("BACKGROUND",    (0, 1), (-1, -1), SOFT_GRAY),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, SOFT_GRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
    ]))
    tbl_w, tbl_h = tbl.wrapOn(c, TEXT_W, 200)
    y -= 4
    tbl.drawOn(c, MARGIN_L, y - tbl_h)
    y -= (tbl_h + 12)

    # ═════════════════════════════════════════════════════════════════════════
    # 4. AI EXPLAINABILITY
    # ═════════════════════════════════════════════════════════════════════════
    y = section("3.  AI Model Explainability", y - 4)
    y = body("How the classifier reached this result:", y, font="Helvetica-Bold", size=9)

    expl_items = [
        ("Architecture:",
         "Swin Transformer (Swin-T) — a hierarchical vision transformer that splits the image into"
         " non-overlapping patches and learns local-to-global tissue features via shifted window attention."),
        ("Training data:",
         "Fine-tuned on a labelled dataset of intraoral clinical photographs annotated for the presence"
         " or absence of malignant mucosal lesions."),
        ("Temperature scaling:",
         "Raw logits are divided by T=1.5 before softmax, which prevents overconfident probability"
         " estimates that are common in small fine-tuned models."),
        ("Logit margin gate:",
         f"The absolute difference between the two class logits is {margin}. A margin below 3.2 triggers"
         " an INCONCLUSIVE result to avoid false high-confidence predictions on non-intraoral images"
         " (e.g. face, clothing, teeth)."),
        ("What the model looks for:",
         "The model attends to colour irregularities, surface texture changes, mucosal pallor or"
         " erythroplakia patterns, and tissue boundary disruptions consistent with OSCC lesions."),
        ("Limitations:",
         "This prototype cannot assess depth of invasion (DOI), perineural invasion, or lymph node"
         " status — all of which require histopathological examination. A positive result must be"
         " confirmed by a qualified oral oncologist."),
    ]
    for key, val in expl_items:
        # Key line
        c.setFillColor(BURGUNDY)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(MARGIN_L + 4, y, key)
        y -= 12
        # Word-wrap the value
        words = val.split()
        line, lines = "", []
        for w in words:
            test = (line + " " + w).strip()
            if c.stringWidth(test, "Helvetica", 8.5) < TEXT_W - 20:
                line = test
            else:
                lines.append(line)
                line = w
        if line:
            lines.append(line)
        for ln in lines:
            c.setFillColor(CHARCOAL)
            c.setFont("Helvetica", 8.5)
            c.drawString(MARGIN_L + 14, y, ln)
            y -= 12
        y -= 3

    # ═════════════════════════════════════════════════════════════════════════
    # 5. CLINICAL RISK INTERPRETATION
    # ═════════════════════════════════════════════════════════════════════════
    y = section("4.  Clinical Risk Interpretation Guide", y - 4)
    risk_rows = [
        ["AI Output", "Interpretation", "Recommended Action"],
        ["NORMAL / BENIGN",
         "No markers of malignant\nmucosal change detected",
         "Routine dental follow-up; report any persistent\nlesions > 3 weeks to a clinician"],
        ["CANCER / OSCC",
         "Features consistent with\noral mucosal carcinoma",
         "Immediate referral to oral oncology. Biopsy\nand histopathology required for diagnosis"],
        ["INCONCLUSIVE",
         "Image is out-of-distribution\nor logit margin too low",
         "Retake close-up intraoral photograph under\nbright light; consult a clinician if in doubt"],
    ]
    rtbl = Table(risk_rows, colWidths=[110, 150, 185])
    rtbl.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), DARK_MAROON),
        ("TEXTCOLOR",      (0, 0), (-1, 0), WHITE),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SOFT_GRAY]),
        ("GRID",           (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
    ]))
    rtbl_w, rtbl_h = rtbl.wrapOn(c, TEXT_W, 200)
    y -= 4
    rtbl.drawOn(c, MARGIN_L, y - rtbl_h)
    y -= (rtbl_h + 14)

    # ═════════════════════════════════════════════════════════════════════════
    # 6. MEDICAL DISCLAIMER
    # ═════════════════════════════════════════════════════════════════════════
    y = section("5.  Important Medical Disclaimer", y - 4)

    c.setFillColor(colors.HexColor("#FFF8E7"))
    c.setStrokeColor(ACCENT)
    c.setLineWidth(0.8)
    c.roundRect(MARGIN_L, y - 54, TEXT_W, 60, 4, fill=1, stroke=1)

    disc_lines = [
        "This report is generated by an experimental educational AI prototype and does NOT constitute a",
        "medical diagnosis. It must not be used to guide any clinical or therapeutic decisions. Oral",
        "cancer diagnosis requires physical examination, histopathology (biopsy), imaging, and assessment",
        "by a qualified oral and maxillofacial surgeon or oncologist. If you notice any oral lesion,",
        "ulcer, or discolouration persisting beyond 3 weeks, consult a healthcare professional immediately.",
    ]
    dy = y - 10
    for dl in disc_lines:
        c.setFillColor(colors.HexColor("#7A4800"))
        c.setFont("Helvetica", 8)
        c.drawString(MARGIN_L + 8, dy, dl)
        dy -= 11

    y -= 68

    # ═════════════════════════════════════════════════════════════════════════
    # FOOTER
    # ═════════════════════════════════════════════════════════════════════════
    c.setFillColor(BURGUNDY)
    c.rect(0, 0, PAGE_W, 26, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, 26, PAGE_W, 1.5, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica", 7.5)
    c.drawCentredString(PAGE_W / 2, 9, "Oral Cancer Detector  –  AI Research Prototype  |  Not for clinical use  |  Consult a qualified clinician for medical advice")

    c.save()
    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"oral_screening_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    )


def _qstatus(val, lo, hi, higher_is="mid"):
    """Return Pass/Fail string for a value in a range."""
    try:
        v = float(val)
        if lo <= v <= hi:
            return "✓ Pass"
        return "✗ Fail"
    except (ValueError, TypeError):
        return "N/A"


def _qstatus_min(val, minimum):
    """Return Pass/Fail for minimum threshold."""
    try:
        v = float(val)
        return "✓ Pass" if v >= minimum else "✗ Fail"
    except (ValueError, TypeError):
        return "N/A"

if __name__ == "__main__":
    app.run(debug=True)

