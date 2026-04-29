import os
import json
from openai import OpenAI

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Please install PyMuPDF and python-dotenv to run this script:")
    print("uv pip install pymupdf python-dotenv openai")
    exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()  # Load API keys from .env if present
except ImportError:
    pass

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from the given PDF file."""
    print(f"Extracting text from: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        exit(1)

def chunk_text(text: str, words_per_chunk: int = 800):
    """Splits text into manageable chunks for the LLM."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunks.append(" ".join(words[i:i+words_per_chunk]))
    return chunks

def generate_synthetic_data(client: OpenAI, chunk: str):
    """Calls GPT-4o to generate synthetic training pairs from a text chunk."""
    prompt = f"""
    Analyze the following text snippet from a report on "Young Graduates' Career Moves & Strategies".
    Generate 3 to 5 high-quality training examples (instruction, input, output) for an AI career advisor.
    
    Guidelines:
    - `instruction`: A realistic question a young graduate might ask about career strategy, job hunting, or market trends.
    - `input`: Any additional context (can be an empty string if the instruction is self-contained).
    - `output`: A helpful, detailed, and insightful response based ONLY on the provided text snippet.
    
    Respond strictly in the following JSON format:
    {{
        "examples": [
            {{
                "instruction": "...",
                "input": "",
                "output": "..."
            }}
        ]
    }}
    
    Text snippet:
    {chunk}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert data generator creating high-quality instruction-tuning datasets."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        return data.get("examples", [])
        
    except Exception as e:
        print(f"Error calling OpenAI API or parsing JSON: {e}")
        return []

def main():
    pdf_path = r"c:\Users\momo-\OneDrive\Desktop\FinNavigator\Young Graduates' Career Moves & Strategies.pdf"
    output_path = "synthetic_career_data.jsonl"
    
    # Verify OpenAI API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not found.")
        print("Please set it in your .env file or environment.")
        exit(1)
        
    client = OpenAI()
    
    # 1. Read PDF
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print("No text extracted from PDF. Is the file empty or an image?")
        exit(1)
        
    # 2. Chunk text to fit context windows efficiently
    chunks = chunk_text(text, words_per_chunk=800)
    print(f"Extracted {len(text)} characters. Split into {len(chunks)} chunks.")
    
    # 3. Generate Data
    all_examples = []
    
    # Clear existing output file if it exists
    if os.path.exists(output_path):
        os.remove(output_path)
        
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)} via GPT-4o...")
        examples = generate_synthetic_data(client, chunk)
        
        if examples:
            all_examples.extend(examples)
            # Append progressively to file in case of crash/interrupt
            with open(output_path, "a", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                    
    print(f"\n✅ Success! Generated {len(all_examples)} synthetic examples.")
    print(f"📁 Data saved to: {output_path}")

if __name__ == "__main__":
    main()
