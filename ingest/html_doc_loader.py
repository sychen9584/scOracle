import os
from bs4 import BeautifulSoup
from llama_index.core.schema import Document

def extract_text_from_html(file_path) -> str:
    """
    Extract text from an HTML file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
        
        # Remove navigation, scripts, and footers
        for tag in soup(['nav', 'footer', 'script', 'style', 'head']):
            tag.decompose()
            
        # Target the main content area
        main = soup.find("div", {"role": "main"}) or soup.body
        return main.get_text(separator="\n", strip=True) if main else ""
    
def load_html_documents(directory: str) -> list[Document]:
    """
    Load HTML documents from a directory and extract text.
    """
    documents = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".html"):
                path = os.path.join(root, file)
                try:
                    text = extract_text_from_html(path)
                    if text:
                        rel_path = os.path.relpath(path, directory)
                        documents.append(Document(
                            text=text,
                            metadata={
                                "source": rel_path,
                                "type": "html_doc",
                            }
                        ))
                except Exception as e:
                    print(f"[WARN] Failed to process {path}: {e}")
                    
    print(f"[INFO] Loaded {len(documents)} HTML documents from {directory}")
    return documents