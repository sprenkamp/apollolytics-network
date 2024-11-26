import sys
import argparse
import pandas as pd
import concurrent.futures
from tqdm import tqdm
from typing import List
from langchain.embeddings import OpenAIEmbeddings

class EventEmbeddingSystem:
    """
    A system to generate embeddings for text data using the OpenAI Embeddings API.
    """
    def __init__(
        self,
        embedding_model_name: str = "text-embedding-3-large",
    ):
        """
        Initializes the EventEmbeddingSystem with necessary components.

        Args:
            embedding_model_name (str): Name of the OpenAI embedding model.
        """
        self.embedding_model_name = embedding_model_name
        self.embedding_model = self._initialize_embedding_model()

    def _initialize_embedding_model(self) -> OpenAIEmbeddings:
        """
        Initializes the OpenAI embeddings model.

        Returns:
            OpenAIEmbeddings: Initialized embeddings model.
        """
        try:
            embedding_model = OpenAIEmbeddings(model=self.embedding_model_name)
            print(f"Initialized embedding model '{self.embedding_model_name}'.")
            return embedding_model
        except Exception as e:
            sys.exit(f"Failed to initialize embedding model: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates an embedding for a single text using the OpenAI embedding model.

        Args:
            text (str): The text to embed.

        Returns:
            List[float]: Embedding vector.
        """
        try:
            embedding = self.embedding_model.embed_documents([text])[0]
            return embedding
        except Exception as e:
            print(f"Failed to generate embedding for text '{text}': {e}")
            return []

def process_text(idx_text):
    idx, text = idx_text
    embedding = embedding_system.get_embedding(text)
    return idx, embedding

if __name__ == "__main__":
    # Use argparse to get the input CSV file path
    parser = argparse.ArgumentParser(description='Process a CSV file to generate embeddings.')
    parser.add_argument('--input', type=str, help='Path to the input CSV file.')
    args = parser.parse_args()
    MAX_WORKERS = 10
    # Initialize the embedding system
    embedding_system = EventEmbeddingSystem()

    # Read the CSV file
    df = pd.read_csv(args.input)

    # Assuming the text to embed is in a column named 'text'
    if 'messageText' not in df.columns:
        sys.exit("The CSV file must contain a 'text' column.")

    # Check if 'embedding' column already exists
    if 'embedding' in df.columns:
        print("The CSV file already contains an 'embedding' column. No processing needed.")
        sys.exit()

    # Initialize the 'embedding' column with None
    df['embedding'] = None

    # Get the texts to process
    texts_to_process = df['messageText'].tolist()
    # Process texts with progress bar
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_text, enumerate(texts_to_process)), total=len(texts_to_process), desc="generating embeddings"))

    # Assign embeddings back to DataFrame
    for idx, embedding in results:
        df.at[idx, 'embedding'] = embedding

    # Save the DataFrame to the output CSV
    df.to_csv(args.input, index=False)
    print("Embeddings generated and saved to the input CSV file.")
