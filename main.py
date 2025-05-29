import pdf2image
import pytesseract
from pytesseract import Output
import pandas as pd
import json
import cv2
import numpy as np
from pdf2image import convert_from_path
from pytesseract import image_to_string

pdf_path = "/mnt/d/THINK/Think-docs/cfh_data/2369BB39-B0FD-4EB5-B679-070762AC88E7.pdf"

# Convert PDF pages to images
pages = pdf2image.convert_from_path(pdf_path, dpi=400)

all_tables = []

for page_num, page in enumerate(pages):
    img = np.array(page)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    # Find horizontal and vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    mask = cv2.add(horizontal, vertical)
    # Find contours (table regions)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h > 100:  # filter out small boxes
            table_img = img[y:y+h, x:x+w]
            text = image_to_string(table_img)
            print(f"Table {i+1} on page {page_num+1}:")
            print(text)
            # Parse text into rows
            rows = [line.split() for line in text.split('\n') if line.strip()]
            # Pad rows to the same length
            if rows:
                max_cols = max(len(row) for row in rows)
                rows = [row + [''] * (max_cols - len(row)) for row in rows]
                df = pd.DataFrame(rows)
                all_tables.append(df)

# Save all tables to Excel (one sheet per table)
if all_tables:
    with pd.ExcelWriter("ocr_extracted_tables.xlsx", engine="openpyxl") as writer:
        for i, df in enumerate(all_tables):
            df.to_excel(writer, sheet_name=f"Page_{i+1}", index=False)
    print("OCR table extraction complete. Saved to ocr_extracted_tables.xlsx")
else:
    print("No tables were extracted from the PDF via OCR.")

tables_as_lists = [df.values.tolist() for df in all_tables]
with open("ocr_tables.json", "w", encoding="utf-8") as f:
    json.dump(tables_as_lists, f, ensure_ascii=False, indent=2)
print("Saved all tables to ocr_tables.json")

print(f"Total number of tables detected: {len(all_tables)}")

# Convert JSON tables to Excel
with open("ocr_tables.json", "r", encoding="utf-8") as file:
    data = json.load(file)

if data:
    with pd.ExcelWriter("converted_tables.xlsx") as writer:
        for idx, table in enumerate(data):
            df = pd.DataFrame(table)
            df.to_excel(writer, sheet_name=f"Table_{idx+1}", index=False)
    print("Converted ocr_tables.json to converted_tables.xlsx")
else:
    print("No tables found in ocr_tables.json, so no Excel file was created.")