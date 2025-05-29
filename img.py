import os
import pandas as pd
import json
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# Configure pipeline options for table extraction
pipeline_options = PdfPipelineOptions(
    do_table_structure=True,
    table_structure_options=TableStructureOptions(),
    device="cpu"  # <-- Force CPU instead of GPU
)


# ✅ Correctly configured DocumentConverter (used throughout)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=PyPdfiumDocumentBackend
        )
    }
)

def extract_pdf_table(pdf_path, output_format='json'):
    """
    Extract tables from a PDF using Docling's DocumentConverter.
    """
    # ✅ Use the global converter instead of redefining it
    result = converter.convert(pdf_path)
    doc = result.document

    # Extract tables
    tables = [item for item in doc.items if item.type == 'table']
    if not tables:
        print("⚠️ No tables found in the document.")
        return None

    table = tables[0]
    data = table.data

    if output_format == 'json':
        return {"table": data}
    elif output_format == 'dataframe':
        return pd.DataFrame(data)
    elif output_format == 'markdown':
        return pd.DataFrame(data).to_markdown(index=False)
    else:
        raise ValueError("Unsupported output format")

# Main execution
if __name__ == "__main__":
    PDF_PATH = "/mnt/d/THINK/Think-docs/cfh_data/2369BB39-B0FD-4EB5-B679-070762AC88E7.pdf"

    try:
        result = extract_pdf_table(PDF_PATH, output_format='json')
        if result:
            with open("structured_table_output.json", "w", encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            df = pd.DataFrame(result['table'])
            print("Extracted Table:")
            print(df)
            print("\n✅ Table saved to structured_table_output.json")
        else:
            print("⚠️ No table data found in the specified document.")

    except Exception as e:
        print(f"❌ Error processing PDF: {str(e)}")
