# C:\Users\y sai krishna\OneDrive\Desktop\Geetham_e-Exam\data_processing\process_books.py

import os
import fitz  # PyMuPDF
import re
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- CONFIGURATION ---
# The root path of your project.
# Using a raw string (r"...") is important for Windows paths with spaces.
ROOT_PATH = r"C:\Users\y sai krishna\OneDrive\Desktop\Geetham_e-Exam"
PDF_DIRECTORY = os.path.join(ROOT_PATH, "data_processing", "source_pdfs")
DB_PATH = os.path.join(ROOT_PATH, "server", "db")
MODEL_NAME = 'all-MiniLM-L6-v2'  # A powerful and efficient model for embeddings

def extract_content_and_solutions(pdf_path):
    """
    Extracts main book content and solutions from a PDF. It intelligently
    separates the two by looking for common solution-section headers.
    Returns two separate strings: one for content, one for solutions.
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  [Error] Could not open PDF {os.path.basename(pdf_path)}: {e}")
        return "", ""

    book_content = ""
    solutions_content = ""
    is_solution_section = False
    
    # Regex to find headers like "Solutions", "Answer Key", etc., on a new line
    solution_keywords = re.compile(r'^\s*(solutions|answer key|hints and solutions|answers)\s*$', re.IGNORECASE | re.MULTILINE)

    for page in doc:
        text = page.get_text("text")
        
        if not is_solution_section and solution_keywords.search(text):
            is_solution_section = True

        if is_solution_section:
            solutions_content += text + "\n"
        else:
            book_content += text + "\n"
            
    doc.close()
    return book_content, solutions_content

# --- INITIALIZATION ---
print("Initializing models and database... This may take a moment.")
client = chromadb.PersistentClient(path=DB_PATH)
embedding_model = SentenceTransformer(MODEL_NAME)
collection = client.get_or_create_collection(name="jee_books")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# --- MAIN PROCESSING LOOP ---
processed_books = set()
print(f"Starting to process PDFs from: {PDF_DIRECTORY}\n")

for filename in os.listdir(PDF_DIRECTORY):
    # Process only main textbook files, not solution files directly
    if filename.endswith(".pdf") and "_Solutions" not in filename:
        
        base_name = filename.replace(".pdf", "")
        if base_name in processed_books:
            continue

        print(f"Processing Book: {base_name}")
        
        # 1. Extract content from the main textbook PDF
        book_path = os.path.join(PDF_DIRECTORY, filename)
        book_content, book_solutions = extract_content_and_solutions(book_path)
        
        # 2. Check for a corresponding separate solutions book
        solution_filename = f"{base_name}_Solutions.pdf"
        solution_path = os.path.join(PDF_DIRECTORY, solution_filename)
        external_solutions = ""
        if os.path.exists(solution_path):
            print(f"  Found separate solutions file: {solution_filename}")
            external_solutions, _ = extract_content_and_solutions(solution_path)

        # 3. Combine all solutions and chunk all text content
        all_solutions_text = book_solutions + "\n" + external_solutions
        
        content_chunks = text_splitter.split_text(book_content)
        solution_chunks = text_splitter.split_text(all_solutions_text)

        # 4. Helper function to add chunks to the database
        def add_to_db(chunks, content_type):
            if not chunks: 
                print(f"  No text found for {content_type}.")
                return
            
            ids = [f"{base_name}_{content_type}_{i}" for i in range(len(chunks))]
            metadatas = [{"source_book": base_name, "content_type": content_type} for _ in chunks]
            
            try:
                embeddings = embedding_model.encode(chunks, show_progress_bar=False).tolist()
                collection.add(embeddings=embeddings, documents=chunks, metadatas=metadatas, ids=ids)
                print(f"  Successfully added {len(chunks)} chunks for '{content_type}'")
            except Exception as e:
                print(f"  [Error] Failed to add chunks to DB for {base_name}: {e}")

        # 5. Add content and solutions to the database
        add_to_db(content_chunks, "book_content")
        add_to_db(solution_chunks, "solution")
        
        processed_books.add(base_name)
        print("-" * 20)

print(f"\n--- Processing complete! ---")
print(f"Database has been built at: {DB_PATH}")
print(f"Total items in collection: {collection.count()}")