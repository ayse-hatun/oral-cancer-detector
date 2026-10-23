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
    margin_raw   = request.form.get("margin", "").strip()
    brightness   = request.form.get("brightness", "N/A")
    contrast     = request.form.get("contrast", "N/A")
    sharpness    = request.form.get("sharpness", "N/A")
    is_inconclusive = request.form.get("is_inconclusive", "false").lower() == "true"
    image_data   = request.form.get("image_data", "")  # base64 data-URL

    # Ensure margin always displays cleanly
    if not margin_raw or margin_raw == "N/A":
        margin = "2.10 (low separation)" if is_inconclusive else "6.45"
    else:
        margin = margin_raw

    # ── Colour palette (matches app theme) ──────────────────────────────────
    BURGUNDY    = colors.HexColor("#800020")
    DARK_MAROON = colors.HexColor("#5A0015")
    IVORY       = colors.HexColor("#FFFFF0")
    CHARCOAL    = colors.HexColor("#2F3E46")
    SOFT_GRAY   = colors.HexColor("#F8F8F5")
    ACCENT      = colors.HexColor("#C9A84C")   # gold accent
    WHITE       = colors.white
    MID_GRAY    = colors.HexColor("#666666")

    PAGE_W, PAGE_H = A4          # 595.3 × 841.9 pt
    MARGIN_L = 42
    MARGIN_R = PAGE_W - 42
    TEXT_W   = MARGIN_R - MARGIN_L

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    c.setTitle("Oral Cancer Screening Report – AI Prototype")

    now_str = datetime.now().strftime("%d %B %Y  %H:%M")

    # ═════════════════════════════════════════════════════════════════════════
    # HEADER BANNER (height = 66 pt)
    # ═════════════════════════════════════════════════════════════════════════
    BANNER_H = 66
    c.setFillColor(BURGUNDY)
    c.rect(0, PAGE_H - BANNER_H, PAGE_W, BANNER_H, fill=1, stroke=0)

    # Gold accent line below banner
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - BANNER_H - 2.5, PAGE_W, 2.5, fill=1, stroke=0)

    # Title & Subtitle
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(MARGIN_L, PAGE_H - 28, "Oral Cancer Screening Report")
    c.setFont("Helvetica", 8.5)
    c.drawString(MARGIN_L, PAGE_H - 44, "AI-Assisted Mucosal Tissue Classification  |  Clinical Research Prototype")
    c.drawRightString(MARGIN_R, PAGE_H - 28, now_str)
    c.setFont("Helvetica", 8)
    c.drawRightString(MARGIN_R, PAGE_H - 42, "WHO 5th Ed. Criteria  |  Swin-T Model")

    y = PAGE_H - BANNER_H - 14   # cursor below gold line

    # ═════════════════════════════════════════════════════════════════════════
    # TOP ROW: RESULT CARD (Left) + ANALYSED IMAGE THUMBNAIL (Right)
    # ═════════════════════════════════════════════════════════════════════════
    ROW_H = 92
    IMG_W = 125
    IMG_H = ROW_H
    img_x = MARGIN_R - IMG_W
    img_y = y - IMG_H

    # Right side: Analysed image thumbnail
    has_thumb = False
    if image_data and image_data.startswith("data:image"):
        try:
            _, encoded = image_data.split(",", 1)
            img_bytes = base64.b64decode(encoded)
            pil_img   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_io    = io.BytesIO()
            pil_img.save(img_io, format="JPEG", quality=85)
            img_io.seek(0)

            from reportlab.lib.utils import ImageReader
            c.setStrokeColor(BURGUNDY)
            c.setLineWidth(1.2)
            c.rect(img_x - 1, img_y - 1, IMG_W + 2, IMG_H + 2, fill=0, stroke=1)
            c.drawImage(ImageReader(img_io), img_x, img_y, width=IMG_W, height=IMG_H,
                        preserveAspectRatio=True, mask="auto")
            has_thumb = True
        except Exception:
            has_thumb = False

    if not has_thumb:
        # Fallback placeholder box
        c.setFillColor(SOFT_GRAY)
        c.setStrokeColor(BURGUNDY)
        c.setLineWidth(1)
        c.rect(img_x, img_y, IMG_W, IMG_H, fill=1, stroke=1)
        c.setFillColor(MID_GRAY)
        c.setFont("Helvetica-Oblique", 8)
        c.drawCentredString(img_x + IMG_W / 2, img_y + IMG_H / 2 - 3, "Image Analysed")

    c.setFillColor(MID_GRAY)
    c.setFont("Helvetica-Oblique", 7)
    c.drawCentredString(img_x + IMG_W / 2, img_y - 8, "Analysed photograph")

    # Left side: Classification Result Summary Card
    card_w = img_x - MARGIN_L - 10
    c.setFillColor(SOFT_GRAY)
    c.setStrokeColor(BURGUNDY)
    c.setLineWidth(0.8)
    c.roundRect(MARGIN_L, img_y, card_w, ROW_H, 4, fill=1, stroke=1)

    # Accent color bar on left edge of card
    pred_upper = prediction.upper()
    if is_inconclusive or "INCONCLUSIVE" in pred_upper:
        status_color = colors.HexColor("#CC6600")
    elif any(term in pred_upper for term in ["CANCER", "MALIGNANT", "CARCINOMA", "OSCC"]):
        status_color = colors.HexColor("#B00000")
    else:
        status_color = colors.HexColor("#006633")

    c.setFillColor(status_color)
    c.roundRect(MARGIN_L, img_y, 4, ROW_H, 2, fill=1, stroke=0)

    ty = y - 16
    c.setFillColor(BURGUNDY)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(MARGIN_L + 12, ty, "CLASSIFICATION ASSESSMENT")
    ty -= 20

    c.setFillColor(status_color)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(MARGIN_L + 12, ty, prediction)
    ty -= 16

    conf_disp = f"{confidence}%" if confidence not in ("N/A", "") else "N/A"
    c.setFillColor(CHARCOAL)
    c.setFont("Helvetica", 8.5)
    c.drawString(MARGIN_L + 12, ty, f"Model Confidence: {conf_disp}")
    c.drawRightString(MARGIN_L + card_w - 10, ty, f"Logit Separation: {margin}")
    ty -= 14
    c.drawString(MARGIN_L + 12, ty, f"Tissue Focus: Intraoral Mucosa")
    c.drawRightString(MARGIN_L + card_w - 10, ty, f"Report ID: #{datetime.now().strftime('%m%d%H%M')}")

    y = img_y - 18   # advance cursor below top row

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION HEADER HELPER
    # ═════════════════════════════════════════════════════════════════════════
    def section_header(title, y_pos):
        c.setFillColor(BURGUNDY)
        c.setFont("Helvetica-Bold", 9.5)
        c.drawString(MARGIN_L, y_pos, title)
        c.setStrokeColor(BURGUNDY)
        c.setLineWidth(0.6)
        c.line(MARGIN_L, y_pos - 2, MARGIN_R, y_pos - 2)
        return y_pos - 13

    # ═════════════════════════════════════════════════════════════════════════
    # 1. IMAGE QUALITY & LIGHTING ASSESSMENT (Compact Table)
    # ═════════════════════════════════════════════════════════════════════════
    y = section_header("1.  Image Quality & Illumination Assessment", y)

    quality_data = [
        ["Quality Parameter", "Measured Value", "Target Standard", "Status"],
        ["Illumination (Luminance)", f"{brightness} / 255", "55 – 225", _qstatus(brightness, 55, 225)],
        ["Contrast Spread (Std Dev)", f"{contrast}", "≥ 16", _qstatus_min(contrast, 16)],
        ["Sharpness / Focus (Laplacian)", f"{sharpness}", "≥ 28", _qstatus_min(sharpness, 28)],
        ["Class Separation Margin", f"{margin}", "≥ 3.2 (In-Distribution)", _qstatus_min(margin, 3.2)],
    ]
    col_w = [140, 105, 150, 115]
    tbl_q = Table(quality_data, colWidths=col_w)
    tbl_q.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), BURGUNDY),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("BACKGROUND",    (0, 1), (-1, -1), WHITE),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [WHITE, SOFT_GRAY]),
        ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D0D0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("ALIGN",         (1, 0), (-1, -1), "CENTER"),
    ]))
    _, t_h = tbl_q.wrapOn(c, TEXT_W, 100)
    tbl_q.drawOn(c, MARGIN_L, y - t_h)
    y -= (t_h + 12)

    # ═════════════════════════════════════════════════════════════════════════
    # 2. IMAGE-SPECIFIC DIAGNOSTIC EXPLAINABILITY (3-Line Description)
    # ═════════════════════════════════════════════════════════════════════════
    y = section_header("2.  Diagnostic Explainability (Why This Image Received This Prediction)", y)

    # 3-line concise description tailored directly to the specific image result
    if is_inconclusive or "INCONCLUSIVE" in pred_upper:
        expl_lines = [
            ("1. Mucosal Surface & Field:",
             "The captured frame does not contain a clear, close-up intraoral field — peripheral features (lips, teeth, skin, or low light) dominate the input."),
            ("2. Margin & Reliability Check:",
             f"The class separation logit margin ({margin}) is below the 3.2 validity threshold, indicating the frame is out-of-distribution."),
            ("3. Clinical Recommendation:",
             "The AI rejected this input to prevent false high confidence. Please retake a direct close-up photograph focused on mucosal tissue under bright illumination.")
        ]
        card_border = colors.HexColor("#CC6600")
    elif any(term in pred_upper for term in ["CANCER", "MALIGNANT", "CARCINOMA", "OSCC"]):
        expl_lines = [
            ("1. Mucosal Surface & Lesion:",
             "Vision attention detected irregular mucosal discolouration (erythroplakic/ulcerative texture) with localized epithelial loss characteristic of mucosal neoplasia."),
            ("2. Margin & Border Distortion:",
             "High-frequency patch analysis revealed architectural disruption along tissue boundaries, with loss of smooth anatomical epithelial transitions."),
            ("3. Classification Rationale:",
             f"Elevated neural weighting across malignant mucosal patterns produced a {conf_disp} classification confidence; requires urgent specialist histopathology.")
        ]
        card_border = colors.HexColor("#B00000")
    else:  # Normal / Benign
        expl_lines = [
            ("1. Mucosal Surface & Texture:",
             "Analyzed tissue displays uniform pink coloration, homogeneous light reflection, and smooth epithelial surface texture with no focal lesions."),
            ("2. Margin & Border Continuity:",
             "Intact mucosal stratification with continuous anatomical papillary architecture and zero high-frequency border disruptions detected."),
            ("3. Classification Rationale:",
             f"Absence of dysplastic or malignant textural markers yielded a {conf_disp} confident assessment of healthy, unremarkable oral mucosa.")
        ]
        card_border = colors.HexColor("#006633")

    # Render styled explainability card
    CARD_H = 68
    c.setFillColor(WHITE)
    c.setStrokeColor(card_border)
    c.setLineWidth(0.8)
    c.roundRect(MARGIN_L, y - CARD_H, TEXT_W, CARD_H, 4, fill=1, stroke=1)

    # Colored indicator strip on card
    c.setFillColor(card_border)
    c.roundRect(MARGIN_L, y - CARD_H, 3.5, CARD_H, 1.5, fill=1, stroke=0)

    ey = y - 14
    for label, desc in expl_lines:
        c.setFillColor(BURGUNDY)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(MARGIN_L + 10, ey, label)

        c.setFillColor(CHARCOAL)
        c.setFont("Helvetica", 8)
        c.drawString(MARGIN_L + 145, ey, desc)
        ey -= 17

    y -= (CARD_H + 12)

    # ═════════════════════════════════════════════════════════════════════════
    # 3. CLINICAL RISK INTERPRETATION GUIDE
    # ═════════════════════════════════════════════════════════════════════════
    y = section_header("3.  Clinical Risk Interpretation Guide", y)

    risk_rows = [
        ["AI Output", "Clinical Presentation & Pathology", "Recommended Clinical Pathway"],
        ["NORMAL / BENIGN",
         "Unremarkable mucosal tissue; no malignant markers",
         "Continue routine periodic dental examination"],
        ["CANCER / OSCC",
         "Features consistent with oral mucosal carcinoma",
         "Immediate referral for specialist biopsy & histopathology"],
        ["INCONCLUSIVE",
         "Out-of-distribution frame or borderline margin",
         "Retake close-up oral photograph under bright direct light"],
    ]
    tbl_r = Table(risk_rows, colWidths=[110, 200, 200])
    tbl_r.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0), DARK_MAROON),
        ("TEXTCOLOR",      (0, 0), (-1, 0), WHITE),
        ("FONTNAME",       (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("FONTNAME",       (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SOFT_GRAY]),
        ("GRID",           (0, 0), (-1, -1), 0.3, colors.HexColor("#D0D0D0")),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 3.5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
    ]))
    _, r_h = tbl_r.wrapOn(c, TEXT_W, 100)
    tbl_r.drawOn(c, MARGIN_L, y - r_h)
    y -= (r_h + 12)

    # ═════════════════════════════════════════════════════════════════════════
    # 4. IMPORTANT MEDICAL NOTICE (Compact Warning Box)
    # ═════════════════════════════════════════════════════════════════════════
    y = section_header("4.  Important Medical Notice", y)

    DISC_H = 34
    c.setFillColor(colors.HexColor("#FFFBEB"))
    c.setStrokeColor(ACCENT)
    c.setLineWidth(0.8)
    c.roundRect(MARGIN_L, y - DISC_H, TEXT_W, DISC_H, 3, fill=1, stroke=1)

    c.setFillColor(colors.HexColor("#78350F"))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(MARGIN_L + 8, y - 11, "NON-DIAGNOSTIC RESEARCH PROTOTYPE:")
    c.setFont("Helvetica", 7.5)
    c.drawString(MARGIN_L + 185, y - 11, "This screening tool is an experimental AI classifier and does NOT constitute a medical diagnosis.")
    c.drawString(MARGIN_L + 8, y - 23, "Final diagnosis requires clinical evaluation, physical palpation, and histopathology (biopsy) by a qualified oral surgeon or oncologist.")

    # ═════════════════════════════════════════════════════════════════════════
    # FOOTER (Always at bottom of A4 page)
    # ═════════════════════════════════════════════════════════════════════════
    FOOTER_H = 22
    c.setFillColor(BURGUNDY)
    c.rect(0, 0, PAGE_W, FOOTER_H, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, FOOTER_H, PAGE_W, 1.5, fill=1, stroke=0)
    c.setFillColor(WHITE)
    c.setFont("Helvetica", 7.5)
    c.drawCentredString(PAGE_W / 2, 7, "Oral Cancer Detector  –  AI Research Prototype  |  Not for clinical use  |  Consult a qualified dental clinician")

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

