"""Patch index.html to track lastCapturedDataUrl in setFile."""

HTML_PATH = r"templates/index.html"

with open(HTML_PATH, "r", encoding="utf-8") as f:
    content = f.read()

# ── find the exact setFile body and add lastCapturedDataUrl reset + FileReader ──
OLD = (
    "    function setFile(file) {\n"
    "      resultBox.replaceChildren();\n"
    "      clearReport();\n"
    "      if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }\n"
)
NEW = (
    "    function setFile(file) {\n"
    "      resultBox.replaceChildren();\n"
    "      clearReport();\n"
    "      lastCapturedDataUrl = null;\n"
    "      if (previewUrl) { URL.revokeObjectURL(previewUrl); previewUrl = null; }\n"
)

# Add FileReader near end of setFile (before closing brace after button.disabled)
OLD2 = (
    "      $(\"fileName\").textContent = file.name;\n"
    "      fileRow.classList.add(\"show\");\n"
    "      button.disabled = false;\n"
    "    }\n"
    "\n"
    "    input.addEventListener"
)
NEW2 = (
    "      $(\"fileName\").textContent = file.name;\n"
    "      fileRow.classList.add(\"show\");\n"
    "      button.disabled = false;\n"
    "      // Read as base64 for PDF thumbnail\n"
    "      const reader = new FileReader();\n"
    "      reader.onload = (e) => { lastCapturedDataUrl = e.target.result; };\n"
    "      reader.readAsDataURL(file);\n"
    "    }\n"
    "\n"
    "    input.addEventListener"
)

changed = 0
if OLD in content:
    content = content.replace(OLD, NEW, 1)
    changed += 1
    print("patch 1 (lastCapturedDataUrl reset): OK")
else:
    print("patch 1: NOT FOUND")

if OLD2 in content:
    content = content.replace(OLD2, NEW2, 1)
    changed += 1
    print("patch 2 (FileReader): OK")
else:
    print("patch 2: NOT FOUND")

with open(HTML_PATH, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Done — {changed}/2 patches applied.")
