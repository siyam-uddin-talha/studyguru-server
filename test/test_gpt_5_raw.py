#!/usr/bin/env python3
"""
Single-file script: analyze a mixed S3 URL (image or text/pdf/docx/txt) with GPT-5 via LangChain
Usage:
    1. Install dependencies:
       pip install langchain langchain-openai requests PyPDF2 python-docx
    2. Set OPENAI_API_KEY env var (or the script will prompt)
    3. Run:
       python app.py
"""

import os
import sys
import json
import mimetypes
import tempfile
from typing import Optional
from urllib.parse import urlparse

import requests

# Optional extraction libs
try:
    from PyPDF2 import PdfReader  # pip install PyPDF2
except Exception:
    PdfReader = None

try:
    import docx  # pip install python-docx
except Exception:
    docx = None

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# -------------------------
# The exact DOCUMENT_ANALYSIS system prompt (unchanged)
# -------------------------
SYSTEM_PROMPT = """
You are StudyGuru AI analyzing educational content. Analyze the given image/document and provide a structured response:

CONTENT TYPE DETECTION:
- "mcq": If you see ANY of these patterns:
  * Questions with lettered sub-parts (a), b), c), d), etc.)
  * Numbered exercises with multiple parts
  * Multiple choice questions with A, B, C, D options
  * Practice problems with sub-questions
  * Exercises that can be broken into separate solvable problems
  * Exercise sheets with multiple problems to solve
  * Math worksheets with numbered questions (1, 2, 3, etc.)
  * Quiz papers with multiple questions
  * Any document with multiple separate questions that can be answered individually
  
- "written": Only if it's pure explanatory text, essays, or single concept explanations
- "other": For mixed content or unclear format

CRITICAL: If you see a document with multiple numbered questions (like 1, 2, 3, 4, 5, 6, 7, 8, 9), it MUST be classified as "mcq" type, even if the questions don't have traditional A/B/C/D options. Each numbered question should be treated as a separate question to extract.

INSTRUCTIONS:
1. First, detect the language of the content
2. Carefully identify the content type using the criteria above
3. Provide a short, descriptive title for the page/content  
4. Provide a summary title that describes what you will help the user with
5. Based on the question type:
   - If MCQ: Extract each question/exercise part as a separate question
     * Look for numbered questions (1, 2, 3, 4, 5, 6, 7, 8, 9, etc.) and extract each one
     * If the document contains actual multiple choice options (like: A, B, C, D or a, b, c, d or 1, 2, 3, 4), include them in the "options" field
     * If the document does NOT contain multiple choice options, omit the "options" field entirely
     * For the "answer" field: provide the correct option letter if options exist, or provide the actual solution/answer if no options are given
     * IMPORTANT: Extract ALL numbered questions from the document, not just a summary
   - If written: Provide organized explanatory content

RESPONSE FORMAT:
Respond with valid JSON only. No additional text or formatting.

For MCQ content:
{
    "type": "mcq",
    "language": "detected language",
    "title": "short descriptive title for the content",
    "summary_title": "summary of how you will help the user",
    "_result": {
        "questions": [
            {
                "question": "question text (e.g., 'Domain of 1/x is …………….')",
                "options": {
                    "a": "option1",
                    "b": "option2", 
                    "c": "option3",
                    "d": "option4"
                },
                "answer": "correct option letter (like: 'c')",
                "explanation": "step-by-step solution or brief explanation"
            }
        ]
    }
}

For written content:
{
    "type": "written",
    "language": "detected language", 
    "title": "short descriptive title for the content",
    "summary_title": "summary of how you will help the user",
    "_result": {
        "content": "organized explanatory text as you would provide in a chat response"
    }
}

CRITICAL REQUIREMENTS:
- If the document has multiple choice options, ALWAYS include the "options" field with a, b, c, d keys
- If the document does NOT have multiple choice options, omit the "options" field completely
- The "answer" field should contain the correct option letter (a, b, c, d) if options exist
- The "explanation" field should provide clear, educational explanations
- Extract ALL questions from the document - don't summarize or skip any
- Use proper mathematical notation and clear language
- CRITICAL: Respond with valid JSON only. No markdown, no explanations outside JSON.
"""

# -------------------------
# Helper functions
# -------------------------


def ensure_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    # prompt user
    key = input("Enter your OPENAI_API_KEY: ").strip()
    if not key:
        print("OpenAI API key required. Exiting.")
        sys.exit(1)
    os.environ["OPENAI_API_KEY"] = key
    return key


def is_probably_image(content_type: Optional[str], url: str) -> bool:
    if content_type:
        return content_type.startswith("image/")
    # fallback by extension
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    return ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff")


def fetch_url_head(url: str) -> Optional[str]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        if r.status_code == 200:
            return r.headers.get("content-type")
    except Exception:
        return None
    return None


def fetch_raw(url: str) -> bytes:
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    return r.content


def extract_text_from_pdf_bytes(b: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError(
            "PyPDF2 is required to extract PDF text. Install with 'pip install PyPDF2'"
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(b)
        tmp.flush()
        path = tmp.name
    reader = PdfReader(path)
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)


def extract_text_from_docx_bytes(b: bytes) -> str:
    if docx is None:
        raise RuntimeError(
            "python-docx is required to extract .docx text. Install with 'pip install python-docx'"
        )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp.write(b)
        tmp.flush()
        path = tmp.name
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


# -------------------------
# Main logic
# -------------------------


def analyze_file_with_gpt5(file_url: str, model_name: str = "gpt-5"):
    """
    Detects file type, extracts text when applicable, and calls GPT-5 via LangChain ChatOpenAI.
    It will pass the SYSTEM_PROMPT as system message and either:
      - for images: include a human message that contains a JSON-like image_url payload (so the model sees the image URL)
      - for text: include the extracted text directly in the human message
    The function prints the raw JSON response (model output) to stdout.
    """

    ensure_api_key()

    # Determine content-type (HEAD preferred)
    content_type = fetch_url_head(file_url)

    if is_probably_image(content_type, file_url):
        mode = "image"
    else:
        mode = "text"

    # Prepare model
    llm = ChatOpenAI(
        model="gpt-5",  # GPT-5 with enhanced vision capabilities
        temperature=0.3,
        max_tokens=5000,
        request_timeout=90,  # Increased timeout for GPT-5 vision processing
        streaming=True,
    )

    # Construct messages depending on mode
    if mode == "image":
        # For images: we pass a human message that mirrors the structure the SYSTEM_PROMPT expects.
        # The DOCUMENT_ANALYSIS prompt (user's original) included an "image_url" object.
        # We'll create a human message containing the textual instruction and a structured JSON snippet with the image URL.
        human_payload = {"type": "image_url", "image_url": {"url": file_url}}
        human_content = "Please analyze this document/image:\n" + json.dumps(
            human_payload
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        # Call the model
        result = llm(messages)
        # llm(messages) returns an LLMResult-like object in many langchain versions; handle gracefully:
        # If it's an object with .content or .generations, try to extract text accordingly.
        output_text = _extract_text_from_llm_result(result)

        # Print the model's response (expected to be JSON only per prompt)
        print(output_text)

    else:
        # Text mode: download & extract text if necessary
        raw_bytes = fetch_raw(file_url)
        text_content = ""

        # Decide by content-type and extension
        ext = os.path.splitext(urlparse(file_url).path)[1].lower()
        content_type_lower = (content_type or "").lower()

        try:
            if content_type_lower == "application/pdf" or ext == ".pdf":
                text_content = extract_text_from_pdf_bytes(raw_bytes)
            elif ext == ".docx" or "word" in (content_type_lower or ""):
                text_content = extract_text_from_docx_bytes(raw_bytes)
            else:
                # treat as plain text if possible
                try:
                    text_content = raw_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    text_content = raw_bytes.decode("latin-1", errors="ignore")
        except Exception as e:
            print(f"Error extracting text from file: {e}", file=sys.stderr)
            # fallback: attempt naive decode
            try:
                text_content = raw_bytes.decode("utf-8", errors="ignore")
            except Exception:
                text_content = ""

        # Compose the human message containing the file text (keeping it concise if massive)
        MAX_CHARS = 20000
        if len(text_content) > MAX_CHARS:
            snippet = text_content[:MAX_CHARS] + "\n\n...[truncated]..."
        else:
            snippet = text_content

        human_content = "Please analyze this document/image:\n" + snippet
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_content),
        ]

        result = llm(messages)
        output_text = _extract_text_from_llm_result(result)
        print(output_text)


def _extract_text_from_llm_result(result) -> str:
    """
    Helper to extract usable text from the LangChain ChatOpenAI call result.
    This supports different LangChain versions.
    """
    # If result is already a string, return it
    if isinstance(result, str):
        return result

    # Some versions return an object with 'content' attribute on the first message
    try:
        if hasattr(result, "content"):
            return result.content
    except Exception:
        pass

    # Some versions return an LLMResult-like object with .generations
    try:
        gens = getattr(result, "generations", None)
        if gens:
            # gens is a list of lists; take first generation's text/content
            if isinstance(gens, list) and len(gens) > 0:
                first = gens[0]
                if isinstance(first, list) and len(first) > 0:
                    g0 = first[0]
                else:
                    g0 = first
                # try common attributes
                for attr in ("text", "message", "content"):
                    if hasattr(g0, attr):
                        val = getattr(g0, attr)
                        if val:
                            return val
                # if it's a dict-like
                if isinstance(g0, dict):
                    for key in ("text", "content"):
                        if key in g0:
                            return g0[key]
    except Exception:
        pass

    # Some versions return an object with .generations[0][0].text
    try:
        return result.generations[0][0].text
    except Exception:
        pass

    # As a last resort, stringify
    try:
        return str(result)
    except Exception:
        return ""


# -------------------------
# CLI
# -------------------------
def main():
    if len(sys.argv) >= 2:
        url = sys.argv[1]
    else:
        url = input("Enter S3 file URL (image or text/pdf/docx/txt): ").strip()
    if not url:
        print("No URL provided. Exiting.")
        sys.exit(1)

    model_choice = "gpt-5"
    print(f"Using model: {model_choice}")
    analyze_file_with_gpt5(url, model_choice)


if __name__ == "__main__":
    main()
