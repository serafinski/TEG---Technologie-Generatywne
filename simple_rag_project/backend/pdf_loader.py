import os
import glob
from pypdf import PdfReader
from langchain.schema import Document

def get_documents_from_directory(directory_path, verbose=False):
    all_docs = []
    
    # Get all PDF files in the directory
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    
    for pdf_path in pdf_files:
        title = os.path.basename(pdf_path)
        
        try:
            with open(pdf_path, "rb") as file:
                pdf_reader = PdfReader(file)
                
                for num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    doc = Document(
                        page_content=page_text, 
                        metadata={'title': title, 'page': (num + 1), 'source': pdf_path}
                    )
                    all_docs.append(doc)
                    
            if verbose:
                    print(f"Processed: {title}")
        except Exception as e:
            print(f"Error processing {title}: {e}")
    
    n_pages = len(all_docs)

    if verbose:
        print(f"\nProcessed {n_pages} pages from {len(pdf_files)} files")

    return all_docs