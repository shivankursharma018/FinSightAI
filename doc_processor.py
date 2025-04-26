# -*- coding: utf-8 -*-
import io
import re
import os # Keep os for path joining in test block
import PyPDF2
import json # For pretty-printing in test block

# --- Configuration ---
# DEBUG_PRINT_TEXT = False # Keep for debugging REGEX if needed

# --- Classification Keywords ---
CLASSIFICATION_KEYWORDS = {
    "GST Invoice": ["gst invoice", "tax invoice", "invoice no", "invoice number", "hsn/sac", "cgst", "sgst", "gstin"],
    "ITR": ["income tax return", "acknowledgement", "assessment year", "pan", "e-filing", "income tax department"],
    "Loan Agreement": ["loan agreement", "lender", "borrower", "principal sum", "interest rate", "loan amount", "promissory note"],
    "Consultant Agreement": ["consulting agreement", "consultancy agreement", "consultant", "client", "services", "scope of work"],
    "NDA": ["non-disclosure agreement", "nda", "confidentiality agreement", "disclosing party", "receiving party", "confidential information"],
}

# --- Reusable Base Patterns ---
BASE_DATE_PATTERN = r"(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{4})"
BASE_AMOUNT_PATTERN = r"((?:[₹$]|INR|USD)\s*)?([\d,]+\.?\d{2})" # Group 2 is the number

# --- Regular Expressions for Information Extraction ---
REGEX_PATTERNS = {
    "date": BASE_DATE_PATTERN,
    "amount": BASE_AMOUNT_PATTERN,
    "gst_no": r"\b(\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d[Z][A-Z\d])\b",
    "pan_no": r"\b([A-Z]{5}\d{4}[A-Z]{1})\b",
    "percentage": r"(\d{1,2}(?:\.\d{1,2})?%?)",
    "party1": r"(?:Lender|Client|Disclosing Party|From|Seller|Service Provider)\s*[:\-]?\s*(.*?)(?:\n|GSTIN:|PAN:|Address:|Registered Office:)",
    "party2": r"(?:Borrower|Consultant|Receiving Party|To|Buyer|Billed To)\s*[:\-]?\s*(.*?)(?:\n|GSTIN:|PAN:|Address:|Registered Office:)",
    "agreement_date": r"(?:Agreement Date|Effective Date|Invoice Date|Date of Filing|Date)\s*[:\-]?\s*" + BASE_DATE_PATTERN,
    "total_amount": r"(?:Total Amount Due|Total Invoice Value|Grand Total|Loan Amount|Compensation|fixed fee of)\s*[:\-]?\s*" + BASE_AMOUNT_PATTERN,
    "taxable_amount": r"(?:Total Taxable Amount|Taxable Value)\s*[:\-]?\s*" + BASE_AMOUNT_PATTERN,
    "tax_rate": r"(?:CGST|SGST|IGST)\s*@\s*(\d{1,2}(?:\.\d{1,2})?%)",
    "itr_ack_no": r"Acknowledgement Number\s*[:\-]?\s*(\d+)",
    "itr_ay": r"Assessment Year\s*[:\-]?\s*(\d{4}-\d{2,4})",
    "itr_total_income": r"Total Income\s*[:\-]?\s*" + BASE_AMOUNT_PATTERN,
    "itr_tax_payable": r"Total Tax Payable\s*[:\-]?\s*" + BASE_AMOUNT_PATTERN,
    "itr_name": r"\nName\s*[:\-]?\s*(.*?)\n",
    "invoice_no": r"Invoice\s*(?:No|Number)\s*[:\-]?\s*(\S+)",
    "term": r"(?:Term of Agreement|Term|Duration|Repayment Term)\s*[:\-]?\s*(.*?)(?:\n|\.|\()"
}


# --- Helper Functions ---

def read_pdf_content_from_bytes(pdf_bytes_io):
    """Extracts text from a PDF provided as an in-memory byte stream."""
    text_content = ""
    try:
        reader = PyPDF2.PdfReader(pdf_bytes_io, strict=False)
        if reader.is_encrypted:
            try:
                reader.decrypt('')
            except Exception:
                print(f"Warning: Could not decrypt PDF. Extraction might fail.")
                pass

        num_pages = len(reader.pages)
        for i in range(num_pages):
            try:
                page = reader.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
            except Exception as page_ex:
                print(f"Warning: Error extracting text from page {i+1}: {page_ex}")
                pass

        if not text_content.strip():
            return None # Indicate no text found

        text_content = re.sub(r'\s{2,}', ' ', text_content)
        text_content = re.sub(r'\n+', '\n', text_content)
        return text_content.strip()

    except Exception as e:
        print(f"Error processing PDF stream: {e}")
        raise ValueError(f"Failed to read PDF content: {e}")


def classify_document(text):
    """Classifies the document based on keywords found in the text."""
    if not text:
        return "Unknown (No Text)"
    text_lower = text.lower()
    scores = {doc_type: 0 for doc_type in CLASSIFICATION_KEYWORDS}
    max_score = 0
    best_match = "Unknown"

    # --- Keyword Scoring (Same as before) ---
    for doc_type, keywords in CLASSIFICATION_KEYWORDS.items():
        score = 0
        for keyword in keywords:
            # Count occurrences of each keyword
            score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower))
        scores[doc_type] = score
        # Track the document type with the highest score
        if score > max_score:
            max_score = score
            best_match = doc_type
        # Simple tie-breaking: If scores are equal, potentially prefer more specific types if needed (optional enhancement)
        # elif score == max_score and score > 0:
            # Add logic here if needed, e.g., prefer Invoice over a generic Agreement if scores tie.

    # --- Thresholding (Same as before) ---
    # Determine the minimum score needed based on the potential best match
    threshold = 2 # Default minimum score
    if best_match in ["GST Invoice", "Loan Agreement", "Consultant Agreement", "NDA"]:
        threshold = 3 # Require a higher score for these more complex types

    # --- Decision ---
    # Check if the highest score meets the required threshold
    if max_score >= threshold:
        return best_match # Return the best match if score is sufficient
    else:
        # If the best score doesn't meet the threshold, classify as Unknown
        # Optionally print debug scores here if needed
        # print(f"Debug Classification Scores ({best_match}, Score: {max_score}, Threshold: {threshold}): {scores}")
        return "Unknown"


def extract_information(text, doc_type):
    """Extracts structured information from the text based on the classified document type."""
    if not text:
         return {} # Return empty dict if no text

    summary = {} # Start with an empty summary

    # Helper to find first match, prioritizing specific capture groups
    def find_first(pattern_key_or_raw, text_content, is_raw_pattern=False):
        pattern = REGEX_PATTERNS.get(pattern_key_or_raw) if not is_raw_pattern else pattern_key_or_raw
        if not pattern: return None # Return None if pattern not found/defined

        match = re.search(pattern, text_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if not match: return None # Return None if no match

        value = None
        if match.lastindex and match.lastindex > 0:
            raw_value = match.group(match.lastindex).strip()
            if not is_raw_pattern and pattern_key_or_raw in ["amount", "total_amount", "taxable_amount", "itr_total_income", "itr_tax_payable"]:
                 # Try extracting numeric part specifically (group 2)
                 if match.lastindex >= 2 and match.group(2):
                     value = match.group(2).replace(',', '').strip()
                 else: # Fallback to last group if group 2 failed
                     value = raw_value.replace(',', '').strip() # Still clean commas
            else:
                value = raw_value.replace('\n', ' ').strip()
        else:
            value = match.group(0).replace('\n', ' ').strip()

        return value if value else None # Return None if value is empty string

    # Helper to find all matches, cleaning results
    def find_all(pattern_key, text_content):
        pattern = REGEX_PATTERNS.get(pattern_key)
        if not pattern: return [] # Return empty list

        matches = re.findall(pattern, text_content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if not matches: return [] # Return empty list

        results = []
        for match in matches:
            val = None
            if isinstance(match, tuple):
                val = next((g for g in reversed(match) if isinstance(g, str) and g.strip()), None)
            elif isinstance(match, str) and match.strip():
                val = match

            if val:
                cleaned_val = val.replace('\n', ' ').strip()
                # Specific cleaning for amounts
                if pattern_key == "amount":
                    amount_num_pattern = r"([\d,]+\.?\d{2})"
                    num_match = re.search(amount_num_pattern, cleaned_val)
                    if num_match:
                        results.append(num_match.group(1).replace(',', '').strip())
                else:
                     results.append(cleaned_val)

        # Remove duplicates for certain fields
        if pattern_key in ["gst_no", "pan_no", "tax_rate", "date"]:
             results = sorted(list(set(results)))

        return results # Return list (might be empty)

    # --- Extraction logic based on document type ---
    # Use .get on summary to avoid KeyError if a field wasn't found
    # Assign directly to summary dict, helper functions return None if not found

    summary["All Identified Dates"] = find_all("date", text)
    summary["Agreement/Filing Date"] = find_first("agreement_date", text)

    if doc_type == "GST Invoice":
        summary["Invoice No"] = find_first("invoice_no", text)
        summary["GSTINs"] = find_all("gst_no", text)
        summary["PANs"] = find_all("pan_no", text)
        summary["Taxable Amount"] = find_first("taxable_amount", text)
        summary["Tax Rates (%)"] = find_all("tax_rate", text)
        cgst_match = re.search(r"CGST\s*(?:@.*?)?[:\-]?\s*" + BASE_AMOUNT_PATTERN, text, re.IGNORECASE | re.DOTALL)
        summary["CGST Amount"] = cgst_match.group(2).replace(',', '').strip() if cgst_match and cgst_match.group(2) else None
        sgst_match = re.search(r"SGST\s*(?:@.*?)?[:\-]?\s*" + BASE_AMOUNT_PATTERN, text, re.IGNORECASE | re.DOTALL)
        summary["SGST Amount"] = sgst_match.group(2).replace(',', '').strip() if sgst_match and sgst_match.group(2) else None
        igst_match = re.search(r"IGST\s*(?:@.*?)?[:\-]?\s*" + BASE_AMOUNT_PATTERN, text, re.IGNORECASE | re.DOTALL)
        summary["IGST Amount"] = igst_match.group(2).replace(',', '').strip() if igst_match and igst_match.group(2) else None
        summary["Total Amount"] = find_first("total_amount", text)
        summary["Seller/Provider Name"] = find_first("party1", text)
        summary["Buyer/Billed To Name"] = find_first("party2", text)

    elif doc_type == "ITR":
        summary["Assessment Year"] = find_first("itr_ay", text)
        summary["PAN"] = find_first("pan_no", text)
        summary["Acknowledgement No"] = find_first("itr_ack_no", text)
        summary["Total Income"] = find_first("itr_total_income", text)
        summary["Tax Payable"] = find_first("itr_tax_payable", text)
        summary["Assessee Name"] = find_first("itr_name", text)

    elif doc_type == "Loan Agreement":
        summary["Loan Amount"] = find_first("total_amount", text)
        interest_match = re.search(r"(?:Interest Rate|rate of)\s*[:\-]?\s*(\d{1,2}(?:\.\d{1,2})?%?)", text, re.IGNORECASE | re.DOTALL)
        summary["Interest Rate"] = interest_match.group(1).strip() if interest_match else find_first("percentage", text)
        summary["Term"] = find_first("term", text)
        summary["Lender"] = find_first("party1", text)
        summary["Borrower"] = find_first("party2", text)

    elif doc_type == "Consultant Agreement":
        summary["Compensation/Fee"] = find_first("total_amount", text)
        summary["Term"] = find_first("term", text)
        summary["Client"] = find_first("party1", text)
        summary["Consultant"] = find_first("party2", text)

    elif doc_type == "NDA":
        term_match = re.search(r"(?:period of|term of|duration of)\s*([\w\s\(\)]+?)\s*(?:year|month|commencing|from)", text, re.IGNORECASE | re.DOTALL)
        summary["Obligation Period"] = term_match.group(1).strip() if term_match else find_first("term", text)
        purpose_match = re.search(r"\nPurpose\s*[:\-]?\s*(.*?)\.?\n", text, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        summary["Purpose"] = purpose_match.group(1).strip() if purpose_match else None
        summary["Disclosing Party"] = find_first("party1", text)
        summary["Receiving Party"] = find_first("party2", text)

    # Attempt to fill missing Agreement Date if specific patterns failed and dates were found
    if not summary.get("Agreement/Filing Date") and summary.get("All Identified Dates"):
         summary["Agreement/Filing Date (Guessed)"] = summary["All Identified Dates"][0]

    # Final cleanup: Remove keys where the value is None or an empty list
    final_summary = {k: v for k, v in summary.items() if v is not None and v != []}
    return final_summary


# ==============================================================================
# --- MAIN INTERFACE FUNCTION FOR BACKEND ---
# ==============================================================================

def process_document_upload(uploaded_file_bytes):
    """
    Processes an uploaded document file bytes. Expected to be PDF.

    Args:
        uploaded_file_bytes (bytes): The raw byte content of the PDF file.

    Returns:
        dict: Contains classification and summary, or an error message.
              Example success: {'classification': 'GST Invoice', 'summary': {...}}
              Example error: {'error': 'Failed to read PDF content.'}
    """
    try:
        pdf_stream = io.BytesIO(uploaded_file_bytes)
        extracted_text = read_pdf_content_from_bytes(pdf_stream) # Can raise ValueError

        if not extracted_text:
            return {"error": "No text could be extracted from the PDF. It might be image-based or corrupted."}

        doc_type = classify_document(extracted_text)

        # Always return classification, even if unknown
        result = {"classification": doc_type, "summary": None} # Default summary to None

        if doc_type in ["Unknown", "Unknown (No Text)"]:
             result["error"] = "Could not reliably classify the document type."
             # Keep summary as None for unknown types
        else:
            # Proceed with extraction only if classification is known
            summary_data = extract_information(extracted_text, doc_type)
            result["summary"] = summary_data # Add the extracted summary

        return result # Return the structured result

    except ValueError as pdf_err:
        print(f"PDF Reading Error: {pdf_err}") # Log error
        return {"error": f"Failed to process PDF: {pdf_err}", "classification": None, "summary": None}
    except Exception as e:
        print(f"Unexpected error during document processing: {e}") # Log error
        return {"error": "An unexpected error occurred while processing the document.", "classification": None, "summary": None}

# ==============================================================================
# --- STANDALONE TEST BLOCK ---
# (This part only runs when you execute the script directly, e.g., python your_script_name.py)
# ==============================================================================
if __name__ == '__main__':
    print("--- Running Document Parser Standalone Test ---")

    # Define the path to a single PDF file for testing
    # *** IMPORTANT: Make sure this path points to a real PDF file in a 'data' subdirectory ***
    # ***          relative to where you run the script from.                  ***
    test_file_relative_path = os.path.join('data', 'Loan Agreement.pdf') # Example file

    # Get the directory where the script is located
    script_directory = os.path.dirname(__file__)
    # Create the absolute path to the test file
    test_file_absolute_path = os.path.join(script_directory, test_file_relative_path)

    print(f"Attempting to read test file: {test_file_absolute_path}")

    try:
        # Check if file exists before opening
        if not os.path.exists(test_file_absolute_path):
             raise FileNotFoundError(f"Test file not found at the specified path: {test_file_absolute_path}")

        with open(test_file_absolute_path, 'rb') as f:
            test_pdf_bytes = f.read()

        print(f"Read {len(test_pdf_bytes)} bytes. Calling process_document_upload...")
        # Call the main interface function
        result_dict = process_document_upload(test_pdf_bytes)

        print("\nProcessing Result (Dictionary):")
        # Pretty-print the dictionary result using json module
        print(json.dumps(result_dict, indent=2))

    except FileNotFoundError as fnf_err:
        print(f"\nERROR: {fnf_err}")
        print("Please ensure the 'data' folder exists in the same directory as the script,")
        print(f"and the file '{os.path.basename(test_file_relative_path)}' is inside the 'data' folder.")
    except Exception as e:
        print(f"\nERROR during standalone test: {e}")
        # Print traceback for detailed debugging if needed
        import traceback
        traceback.print_exc()

    print("\n--- Standalone Test Complete ---")