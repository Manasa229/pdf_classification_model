# app.py
import io
from typing import List
from pdf2image import convert_from_bytes
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.params import File
from fastapi.responses import JSONResponse
import pdfplumber
import pytesseract
from classify import classify_pdf
import uvicorn
from PIL import Image

app = FastAPI()


async def extract_text(file_bytes):
    file_document_type = detect_file_document_type(file_bytes)
    extracted_text=None

    text_pages = []
    if file_document_type =="pdf":
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_pages.append(t)
        extracted_text = "\n\n".join(text_pages).strip()
        if not extracted_text:
            images = convert_from_bytes(file_bytes, dpi=300)
            ocr_texts = []
            for i, img in enumerate(images):
                try:
                    text = pytesseract.image_to_string(img)
                    if text.strip():
                        ocr_texts.append(text)
                except Exception as e:
                    print(f"Error OCR'ing page {i + 1}: {e}")

            extracted_text = "\n\n".join(ocr_texts).strip()
        
    elif file_document_type in ['jpg', 'jpeg', 'png']:
        extracted_text = extract_text_from_image(file_bytes)
    

    return extracted_text,file_document_type


def extract_text_from_image(file_bytes):
    try:

        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

def detect_file_document_type(file_bytes):
    if file_bytes.startswith(b'\xff\xd8\xff'):
        return 'jpg'
    elif file_bytes.startswith(b'\x89PNG'):
        return 'png'
    elif file_bytes.startswith(b'%PDF'):
        return 'pdf'
    else:
        return 'unknown'

@app.post("/classify/")
async def classify_pdf_endpoint(file: UploadFile):

    try:
        file_bytes = await file.read()

        extracted_text,file_document_type= await extract_text(file_bytes)

        if not extracted_text:
            raise HTTPException(status_code=422, detail="Could not extract text from the PDF")

        document_type,confidence = await classify_pdf(extracted_text,file_bytes,file_document_type)
        return JSONResponse(content={"document_type": document_type,"confidence": confidence, "filename": file.filename})

    except HTTPException:
        raise


@app.post("/classify/bulk/")
async def classify_bulk(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        file_bytes = await file.read()
        extracted_text,file_document_type = await extract_text(file_bytes)
        document_type,confidence = await classify_pdf(extracted_text,file_bytes,file_document_type)
        results.append({"filename": file.filename, "document_document_type": document_type, "confidence": confidence})

    return JSONResponse(content={"results": results, "count": len(results)})


if __name__ == "__main__":

    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=True)