from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import io

# Import your custom ML functions
from doc_processor import read_pdf_content_from_bytes, classify_document, extract_information

app = FastAPI(
    title="Document Classifier & Extractor",
    description="Classify financial/legal PDFs and extract structured information",
    version="1.0.0"
)

# Optional: Allow CORS (helpful for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze/")
async def analyze_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await file.read()
        pdf_io = io.BytesIO(file_bytes)

        # Text extraction
        text = read_pdf_content_from_bytes(pdf_io)
        if not text:
            return JSONResponse(status_code=400, content={"error": "Could not extract text from PDF."})

        # Document classification
        doc_type = classify_document(text)

        # Information extraction
        extracted_info = extract_information(text, doc_type)

        return {
            "filename": file.filename,
            "document_type": doc_type,
            "extracted_information": extracted_info,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# Run the API using:
# uvicorn doc_processor_api:app --reload
if __name__ == "__main__":
    uvicorn.run("doc_processor_api:app", host="0.0.0.0", port=8010, reload=True)
