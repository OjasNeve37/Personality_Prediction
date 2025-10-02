import os
import sys
import pdfplumber
import PyPDF2
from docx import Document

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using pdfplumber (primary) or PyPDF2 (fallback)"""
    text = ""
    
    try:
        # Try pdfplumber first (better for formatted PDFs)
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        if text.strip():
            return text
    except Exception as e:
        print(f"pdfplumber failed: {e}. Trying PyPDF2...")
    
    try:
        # Fallback to PyPDF2
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        return text
    except Exception as e:
        print(f"PyPDF2 also failed: {e}")
        return None


def extract_text_from_docx(docx_path):
    """Extract text from DOCX file"""
    try:
        doc = Document(docx_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return None


def parse_resume(file_path):
    """
    Parse resume and extract text
    Supports PDF and DOCX formats
    """
    try:
        print(f"[PARSER] Starting to parse resume: {file_path}")
        
        if not os.path.exists(file_path):
            error_msg = f"Error: File not found - {file_path}"
            print(f"[PARSER] {error_msg}")
            raise FileNotFoundError(error_msg)
        
        print(f"[PARSER] File exists, size: {os.path.getsize(file_path)} bytes")
        file_ext = os.path.splitext(file_path)[1].lower()
        print(f"[PARSER] File extension: {file_ext}")
        
        # Extract text based on file type
        if file_ext == '.pdf':
            print("[PARSER] Processing PDF file...")
            text = extract_text_from_pdf(file_path)
            if text is None:
                raise Exception("PDF extraction failed")
        elif file_ext in ['.docx', '.doc']:
            print("[PARSER] Processing DOCX/DOC file...")
            text = extract_text_from_docx(file_path)
            if text is None:
                raise Exception("DOCX extraction failed")
        else:
            error_msg = f"Unsupported file format: {file_ext}"
            print(f"[PARSER] {error_msg}")
            raise ValueError(error_msg)
        
        if not text or len(text.strip()) < 50:
            print("Warning: Extracted text is too short or empty")
            return None
        
        parsed_data = {
            'name': 'Unknown',  # Could add name extraction logic
            'email': 'N/A',
            'mobile': 'N/A',
            'skills': [],
            'experience': 0,
            'education': [],
            'companies': [],
            'designation': [],
            'raw_text': text
        }
        
        print(f"✓ Successfully extracted {len(text)} characters from resume")
        # Return just the raw text for the model
        return text
    
    except Exception as e:
        print(f"Error parsing resume: {e}")
        return None


if __name__ == "__main__":
    # Test with a sample resume
    test_resume = "data/raw/Ojas Resume.pdf"
    
    if os.path.exists(test_resume):
        result = parse_resume(test_resume)
        if result:
            print("\nExtracted text preview:")
            print(result['raw_text'][:500])  # Print first 500 characters
    else:
        print(f"Test resume not found: {test_resume}")
