import logging
import time
from pathlib import Path
import pandas as pd
from docling.document_converter import DocumentConverter
import re
_log = logging.getLogger(__name__)



def is_arabic(text):
    return re.search(r'[\u0600-\u06FF]', text)

def reverse_arabic_phrase(text):
    if not isinstance(text, str):
        return text
    
    if is_arabic(text):
        words = text.split()
        return ' '.join(reversed(words))
    
    return text

def fix_dataframe_arabic(df):
    return df.applymap(reverse_arabic_phrase)



def main():
    logging.basicConfig(level=logging.INFO)

    PDF_PATH = "TMGH Q1 2024.pdf"
    output_dir = Path("scratch")

    doc_converter = DocumentConverter()

    start_time = time.time()

    conv_res = doc_converter.convert(PDF_PATH)

    output_dir.mkdir(parents=True, exist_ok=True)

    doc_filename = conv_res.input.file.stem

    for table_ix, table in enumerate(conv_res.document.tables):
        
        table_df: pd.DataFrame = table.export_to_dataframe()

        print(f"## Table {table_ix}")
        print(table_df.to_markdown)

        table_corrected= fix_dataframe_arabic(table_df)


        
        element_csv_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.csv"
        _log.info(f"Saving CSV table to {element_csv_filename}")
        table_corrected.to_csv(element_csv_filename)

        
        element_html_filename = output_dir / f"{doc_filename}-table-{table_ix + 1}.html"
        _log.info(f"Saving HTML table to {element_html_filename}")
        with element_html_filename.open("w") as fp:
            fp.write(table.export_to_html(doc=conv_res.document))

    end_time = time.time() - start_time

    _log.info(f"Document converted and tables exported in {end_time:.2f} seconds.")

if __name__ == "__main__":
    main()