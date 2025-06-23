import os
import pytesseract
from pdf2image import convert_from_path
import difflib
import argparse
import re
from docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions,
)
from pathlib import Path
import openai
import pandas as pd
import logging
import time
import base64
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from mistralai import Mistral
from dotenv import load_dotenv
from markdown_it import MarkdownIt
from PyPDF2 import PdfReader
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem
import json
import io 



_log = logging.getLogger(__name__)



IMAGE_RESOLUTION_SCALE=4.0

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key="YOUR API KEY")  
pages = []


def preprocess_arabic_text(text):
    """
    Preprocess Arabic text by removing diacritics, normalizing, and other cleanups.
    This helps improve matching even whenfrom docling.document_converter import DocumentConverter, InputFormat, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from pathlib import Path
import logging OCR isn't perfect.
    """
    # Remove diacritics (harakat)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize alef formscle
    text = re.sub(r'[أإآا]', 'ا', text)
    
    # Normalize yaa and alif maqsura
    text = re.sub(r'[يى]', 'ي', text)
    
    # Normalize taa marbuta and haa
    text = re.sub(r'[ةه]', 'ة', text)
    
    # Remove tatweel (kashida)
    text = re.sub(r'\u0640', '', text)
    
    # Remove non-Arabic characters except spaces
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text




def find_titles_in_pdf(pdf_path, titles, similarity_threshold=0.99):
    """
    Find titles in a PDF file between specified pages using OCR.
    
    Args:
        pdf_path: Path to the PDF file
        titles: List of Arabic titles to search for
        start_page: First page to search (1-indexed)
        end_page: Last page to search (1-indexed)
        similarity_threshold: Threshold for similarity matching (0.0 to 1.0)
    
    Returns:
        Dictionary mapping each title to list of page numbers where it was found
    """
    start_page=1
    try:
        reader = PdfReader(pdf_path)
        end_page = len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return {}

    print(f"Processing PDF: {pdf_path}")
    print(f"Looking for {len(titles)} titles between pages {start_page} and {end_page}")





    
    try:
        images = convert_from_path(
            pdf_path, 
            first_page=1, 
            last_page=end_page
        )
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return {}
    
    print(f"Successfully converted {len(images)} pages to images")

    # Preprocess titles once
    processed_titles = [preprocess_arabic_text(title) for title in titles]
    
    results = {title: [] for title in titles}  # Initialize with empty lists
    pages_found=[]
    # OCR each page
    for i, img in enumerate(images):
        actual_page_num = i + start_page
        print(f"Processing page {actual_page_num}...")

        try:
            text = pytesseract.image_to_string(img, lang='ara')
            processed_text = preprocess_arabic_text(text)
            lines = [line.strip() for line in processed_text.split('\n') if line.strip()]

            for title, processed_title in zip(titles, processed_titles):
                best_ratio = 0
                best_line = ""

                for line in lines:
                    ratio = difflib.SequenceMatcher(None, processed_title, line).ratio()

                    # Sliding window for partial matches
                    if len(processed_title) > 5 and len(line) >= len(processed_title):
                        for j in range(0, len(line) - len(processed_title) + 1):
                            window = line[j:j+len(processed_title)]
                            window_ratio = difflib.SequenceMatcher(None, processed_title, window).ratio()
                            ratio = max(ratio, window_ratio)

                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_line = line

                if best_ratio >= similarity_threshold:
                    print(f"✅ Found '{title}' on page {actual_page_num} (match ratio: {best_ratio:.2f})")
                    print(f"Best matching line: {best_line}")
                    pages_found.append(actual_page_num)

                    
                    if actual_page_num not in results[title]:
                        results[title].append(actual_page_num)

        except Exception as e:
            print(f"Error processing page {actual_page_num}: {e}")

    # Clean up empty entries
    final_results = {title: pages for title, pages in results.items() if pages}

    for title in titles:
        if title not in final_results:
            print(f"Title not found: {title}")

    return final_results




def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None



def dataframe_to_safe_json_v2(df: pd.DataFrame) -> list[dict]:
    """
    Alternative approach using pandas' native JSON conversion with NaN handling.
    """
    # Clean headers
    df.columns = [None if pd.isna(col) else str(col).strip() for col in df.columns]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Replace all NaN with None using pandas' fillna
    df_cleaned = df.where(pd.notna(df), None)
    
    # Convert to JSON string first, then parse back
    # This ensures pandas handles NaN conversion properly
    json_str = df_cleaned.to_json(orient='records', date_format='iso')
    return json.loads(json_str)



def markdown_tables_to_json_list(markdown_text: str) -> list[dict]:
    md = MarkdownIt("gfm-like", {"linkify": False})
    html_output = md.render(markdown_text)
    all_tables_data_json = []
    print("html_output", type(html_output))
    print(html_output)
    try:
        list_of_dataframes = pd.read_html(io.StringIO(html_output))
        print("list_of_dataframes", list_of_dataframes)
        for df in list_of_dataframes:
            df_cleaned = dataframe_to_safe_json_v2(df)
            print()
            all_tables_data_json.append(df_cleaned)
    except ValueError as e:
        print(f"Error processing tables: {e}")
        return []
    
    return all_tables_data_json


def preprocess_ocr_table(ocr_markdown: str) -> str:
    """
    Attempts to convert a loosely formatted OCR table into a valid Markdown table.
    Assumes columns are separated by 2+ spaces.
    """
    # Split text into lines and filter out empty lines
    lines = [line.strip() for line in ocr_markdown.splitlines() if line.strip()]
    if not lines:
        return ocr_markdown

    # Assume the first row is the header
    header_line = lines[0]
    # Split header by multiple spaces
    headers = [col.strip() for col in header_line.split("  ") if col.strip()]
    if len(headers) < 2:
        # Not enough columns to be a table.
        return ocr_markdown

    # Construct a new header and separator row
    header = " | ".join(headers)
    separator = " | ".join(["---"] * len(headers))
    
    # Process the remaining rows similarly
    new_lines = [f"| {header} |", f"| {separator} |"]
    for line in lines[1:]:
        cols = [col.strip() for col in line.split("  ") if col.strip()]
        if len(cols) != len(headers):
            # If a row doesn't match header columns, try to join or skip it
            continue
        new_line = " | ".join(cols)
        new_lines.append(f"| {new_line} |")
    
    return "\n".join(new_lines)








def extract_table_images_from_pdf(pdf_path: Path, page_no: int) -> None:
    """
    Extracts tables from a specific page in a PDF using Docling and saves them as images.

    Args:
        pdf_path (Path): Path to the PDF file.
        page_no (int): Page number to extract tables from (1-based index).
    """
    output_dir = Path("scratch")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure Docling to generate images
    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # Convert only the specific page
    conv_res = doc_converter.convert(pdf_path, page_range=(page_no, page_no))
    doc_stem = conv_res.input.file.stem

    table_counter = 0
   # Save page images
    for page_no, page in conv_res.document.pages.items():
        page_no = page.page_no
        page_image_filename = output_dir / f"-{page_no}.png"
        with page_image_filename.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")




def main():
    pdf_path = "cib25.pdf"

    parser = argparse.ArgumentParser(description='Find Arabic titles in scanned PDF documents')
    parser.add_argument('--pdf_path',default=pdf_path,help='Path to the PDF file')
    parser.add_argument('--start-page', type=int, help='First page to search (default: 2)')
    parser.add_argument('--end-page', type=int, help='Last page to search (default: 15)')
    parser.add_argument('--threshold', type=float, default=0.99, help='Similarity threshold (0.0-1.0, default: 0.8)')
    args = parser.parse_args()
    
    # List of Arabic titles to search for - replace with your actual titles
    titles = [
    "قائمة الدخل الشامل المجمعة الدورية المختصرة",
    ]
    
    # Find titles in the PDF
    results = find_titles_in_pdf(
        args.pdf_path,  
        titles, 
        similarity_threshold=args.threshold
    )
    table_ocr_results=[]
    # Print results in a formatted way
    if results:
        print("\n=== RESULTS ===")
        print("Title".ljust(40) + "Page")
        print("-" * 50)
        for title, pages in results.items():
            print(f"{title[:37] + '...' if len(title) > 40 else title.ljust(40)} {pages}")
            print(pages)
            for page in pages:
                extract_table_images_from_pdf(pdf_path,page_no=page)
            

            scratch_dir = Path("scratch")
            for image_path in scratch_dir.glob("*.png"):
                print(f"Processing image: {image_path}")

                base64_image = encode_image(image_path)
                try:
                    ocr_response = client.ocr.process(
                    model="mistral-ocr-latest",
                    document={
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{base64_image}" 
                    },
                    include_image_base64=True
                )

                    for tick in ocr_response.pages:
                        table_ocr_results.append(tick.markdown)
                        print(tick.markdown)
                        print("-" * 40)
                        adjusted_markdown = preprocess_ocr_table(str(tick.markdown))
                        print("adjusted_markdown", adjusted_markdown )
                        print(type(adjusted_markdown))
                        structured_tables = markdown_tables_to_json_list(adjusted_markdown)
                        print(f"structrued tables {structured_tables}")

                    try:
                        if structured_tables:
                            # Safely access the first table if it exists
                            first_table_json = json.dumps(structured_tables[0], indent=2)
                            print(first_table_json)
                            for table in structured_tables:
                                with open("table.json", "w") as f:
                                    f.write(json.dumps(table, indent=2))

                    except Exception as e:
                        print("No tables were found in the markdown content.")
                        print(str(e))

                except Exception as e:
                    print(f"❌ Failed to OCR: {e}")



                




    else:
        print("\nNo titles found in the specified pages.")

if __name__ == "__main__":
    main()