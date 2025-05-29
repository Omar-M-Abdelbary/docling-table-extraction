import pdfplumber
import pandas as pd
import numpy as np
import json

pdf_path = "/mnt/d/THINK/Think-docs/cfh_data/2369BB39-B0FD-4EB5-B679-070762AC88E7.pdf"

# List of known section headers and total keywords
SECTION_HEADERS = [
    "Assets", "Financial investments", "Liabilities and equity", "Liabilities", "Equity"
]
TOTAL_KEYWORDS = ["Total", "TOTAL"]

# Helper functions
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip()
    # Remove multiple spaces
    text = ' '.join(text.split())
    return text

def is_section_header(cell):
    return cell in SECTION_HEADERS or (cell.isupper() and len(cell.split()) < 5)

def is_total_row(cell):
    return any(kw.lower() in cell.lower() for kw in TOTAL_KEYWORDS)

tables = []

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages):
        try:
            page_tables = page.extract_tables({
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines"
            })
            
            if page_tables:
                for table_num, table in enumerate(page_tables):
                    if table and len(table) > 1:
                        # Clean column names
                        headers = [clean_text(h) for h in table[0]]
                        # Remove any None values and replace with 'Column_X'
                        headers = [f'Column_{i}' if not h else h for i, h in enumerate(headers)]
                        # Make column names unique
                        seen = {}
                        for i, h in enumerate(headers):
                            if h in seen:
                                seen[h] += 1
                                headers[i] = f"{h}_{seen[h]}"
                            else:
                                seen[h] = 0
                        
                        # Clean data
                        cleaned_data = []
                        for row in table[1:]:
                            cleaned_row = [clean_text(cell) for cell in row]
                            cleaned_data.append(cleaned_row)
                        
                        df = pd.DataFrame(cleaned_data, columns=headers)
                        # Remove completely empty rows
                        df = df.replace('', np.nan)
                        df = df.dropna(how='all')
                        # Remove completely empty columns
                        df = df.dropna(axis=1, how='all')
                        
                        if not df.empty:
                            tables.append(df)
                            print(f"Extracted table {table_num + 1} from page {page_num + 1}")
                            print(f"Columns: {headers}")
                            print(f"Shape: {df.shape}")
                            print("-" * 40)
        except Exception as e:
            print(f"Error extracting tables from page {page_num + 1}: {str(e)}")

if tables:
    print(f"\nTotal number of tables extracted: {len(tables)}")
    for i, df in enumerate(tables):
        json_filename = f"table_{i+1}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(df.values.tolist(), f, ensure_ascii=False, indent=2)
        print(f"Saved table to {json_filename}")
else:
    print("No tables were extracted from the PDF")
