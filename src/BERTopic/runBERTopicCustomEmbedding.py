import os
import argparse
from bertopic import BERTopic
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP 
from hdbscan import HDBSCAN
import pandas as pd
import openai
from bertopic.representation import OpenAI
import tiktoken 
import numpy as np
from bertopic.backend import BaseEmbedder
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time 


class CustomEmbedder(BaseEmbedder):
    def __init__(self, df, input_file_name, embedding_model="text-embedding-3-large"):
        super().__init__()
        self.embedding_model = embedding_model
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.df = df
        self.embedding_exists = self.check_if_embedding_exists()
        self.input_file_name = input_file_name

    def check_if_embedding_exists(self):
        if "embedding" in self.df.columns:
            return True

    def _embed_single_document(self, document, max_retries=3, delay=30):
        """Helper function to embed a single document with retry mechanism."""
        for attempt in range(max_retries):
            try:
                # Attempt to create the embedding
                result = self.client.embeddings.create(input=document, model=self.embedding_model).data[0].embedding
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error embedding document, retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    # Raise the exception if out of retries
                    raise RuntimeError(f"Failed to embed document after {max_retries} attempts: {e}")
    
    def embed(self, documents, verbose=False):
        """
        Embed documents from a DataFrame, checking if embeddings already exist.

        Args:
            documents (list): List of documents to embed.
        Returns:
            np.ndarray: Array of embeddings.
        """
        # Check if the embedding column exists
        if self.embedding_exists:
            # Always print this message
            print(f"Embedding column exists. Loading embeddings from the column.")
            # Extract embeddings from the column
            embeddings_list = self.df["embedding"].tolist()
            # If embeddings are stored as strings (e.g., after saving/loading CSV), convert them back to arrays
            if isinstance(embeddings_list[0], str):
                import ast 
                embeddings_list = [ast.literal_eval(embedding) for embedding in embeddings_list]
            # Convert list of embeddings to numpy array
            embeddings_array = np.vstack(embeddings_list)
            return embeddings_array
        else:
            embeddings = []
        with ThreadPoolExecutor() as executor:
            # Submit all tasks to the executor
            futures = {executor.submit(self._embed_single_document, doc): doc for doc in documents}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating embeddings", unit="doc"):
                try:
                    # Collect results as they complete
                    embeddings.append(future.result())
                except Exception as e:
                    print(f"Failed to embed document: {futures[future]} due to error: {e}")
        # print(embeddings)
        self.df["embedding"] = embeddings
        self.df.to_csv(self.input_file_name, index=False)
        print(f"Embeddings saved to {self.input_file_name}")
        # Concatenate the list of embeddings into a single NumPy array
        embeddings_array = np.vstack(embeddings)
        return embeddings_array

# Define stopwords
stopWords = stopwords.words('english') 
stopWords.extend(stopwords.words('german'))
stopWords.extend(stopwords.words('russian'))
with open("data/stopwords/stopwords_ua.txt") as file:  # Add Ukrainian stopwords loaded from .txt file
    ukrstopWords = [line.rstrip() for line in file]
stopWords.extend(ukrstopWords)

vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopWords)  # Define vectorizer model with stopwords

def validate_file(f):  # Function to check if file exists
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError(f"{f} does not exist")
    return f

class BERTopicAnalysis:
    """
    Class to train a BERTopic model on a given input file and save the model and visualizations.
    If an existing model is specified, it is loaded for inference.

    Parameters:
        input_file (str): Path to the input file.
        output_folder (str): Path to the output folder.
        k_cluster (int or str): Number of clusters to be used for the model.
        use_existing_model (bool): Whether to use an existing model for inference.
    """

    def __init__(self, input_file, output_folder, k_cluster, use_existing_model, embedding_model="text-embedding-3-large"):
        self.input_file = input_file
        self.output_folder = output_folder
        self.k_cluster = k_cluster 
        self.use_existing_model = use_existing_model
        self.read_data()
        self.custom_embedder = CustomEmbedder(self.df, input_file)
        self.model_folder = self.output_folder  # Use output_folder as model_folder

    def read_data(self):
        self.df = pd.read_csv(self.input_file)
        self.df = self.df[self.df['messageText'].map(type) == str]
        self.df["messageText"] = self.df['messageText'].str.split().str.join(' ')
        self.text_to_analyse_list = self.df['messageText'].values
        print(f"Number of texts to analyse: {len(self.text_to_analyse_list)}")

    def read_model(self):
        model_path = os.path.join(self.model_folder, "BERTopicmodel")
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            self.model = BERTopic.load(model_path, embedding_model=self.custom_embedder)
        else:
            raise FileNotFoundError(f"No model found in {model_path}. Please check the path.")

    def k_cluster_type(self):
        if str(self.k_cluster).isnumeric():
            self.k_cluster = int(self.k_cluster)

    def fit_BERTopic(self):
        umap_model = UMAP(n_neighbors=10, n_components=10, metric='cosine', low_memory=False, random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', prediction_data=True)
        from sklearn.cluster import KMeans
        # kmeans = KMeans(n_clusters=25)
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        openai_model = "gpt-4o-mini"
        custom_prompt = """
Please note that the following messages are likely related to the Ukraine-Russia conflict.

And the following representative documents:
[DOCUMENTS]

Based on this information, please provide a concise and highly specific label that best represents this topic, focusing on the details pertinent to the Ukraine-Russia conflict. Don't create any labels that are too broad or general, like "Russia-Ukraine conflict coverage and commentary" or "Conflict and Military Operations in Ukraine and Russia". Good labels are specific and detailed for each topic, like "Wagner Group Putsch", "Azov Battalion in Mariupol", "Rumours about Zelensky". Still the label needs to be representative of the topic as a whole.
"""
        tokenizer = tiktoken.encoding_for_model(openai_model)
        representation_model = OpenAI(
            client,
            model=openai_model, 
            prompt=custom_prompt,
            chat=True,
            nr_docs=50,
        )
        self.model = BERTopic(
            verbose=True,
            embedding_model=self.custom_embedder,
            vectorizer_model=vectorizer_model,
            representation_model=representation_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
        )
        topics, probs = self.model.fit_transform(self.text_to_analyse_list)

    def save_results(self):
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        fig = self.model.visualize_topics()
        fig.write_html(f"{self.model_folder}/bert_topic_model_distance_model.html")
        fig = self.model.visualize_hierarchy()
        fig.write_html(f"{self.model_folder}/bert_topic_model_hierarchical_clustering.html")
        fig = self.model.visualize_barchart(top_n_topics=30)
        fig.write_html(f"{self.model_folder}/bert_topic_model_word_scores.html")
        fig = self.model.visualize_heatmap()
        fig.write_html(f"{self.model_folder}/bert_topic_model_word_heatmap.html")
        self.model.save(f"{self.model_folder}/BERTopicmodel", serialization="pytorch")
    
    def write_multi_sheet_excel(self):
        writer = pd.ExcelWriter(f"{self.model_folder}/representative_docs.xlsx", engine='xlsxwriter')
        for i in self.model.get_representative_docs().keys():
            df = pd.DataFrame(self.model.get_representative_docs()[i], columns=['message'])
            sheet_name = ''.join([c for c in self.model.get_topic_info()[self.model.get_topic_info()['Topic']==i]['Name'].values[0][:31] if c not in '[]:*?/\\'])
            df.to_excel(writer, sheet_name=sheet_name)
        writer._save()
        self.model.get_topic_info().to_csv(f"{self.model_folder}/topic_info.csv")

    def inference(self):
        print(f"Performing inference on the input data for {len(self.df)} instances")
        pred, prob = self.model.transform(self.df['messageText'].values)
        self.df['cluster'] = pred
        input_filename = os.path.basename(self.input_file)
        output_file = os.path.join(self.model_folder, f"inference_{input_filename}")
        self.df.to_csv(output_file, index=False)
        print(f"Inference results saved to {output_file}")

    def run_all(self):
        if self.use_existing_model:
            self.read_model()
            self.inference()
        else:
            self.k_cluster_type()
            # Create a timestamped subfolder within output_folder
            timestamp = time.strftime("%d%m%Y_%H%M%S")
            self.model_folder = os.path.join(
                self.output_folder, 
                f"{os.path.basename(self.input_file).split('.')[0]}_{self.k_cluster}_{timestamp}"
            )
            self.output_folder = self.model_folder  # Update output_folder to the timestamped folder
            self.fit_BERTopic()
            self.save_results()
            self.write_multi_sheet_excel()
            # After training and saving, perform inference on the input data
            self.inference()

def main():
    # Define parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help="Specify the input file", type=validate_file, required=True)
    parser.add_argument('-o', '--output_folder', help="Specify folder for results or model", required=True)
    parser.add_argument('-k', '--k_cluster', help="Number of topic clusters", required=False, default="auto")
    parser.add_argument('-uem', '--use_existing_model', help="Use existing model in output folder for inference", action='store_true')
    args = parser.parse_args()
    
    # Initialize class
    BERTopic_Analysis = BERTopicAnalysis(
        args.input_file,
        args.output_folder,
        args.k_cluster,
        args.use_existing_model
    )
    # Run all functions
    BERTopic_Analysis.run_all()

if __name__ == '__main__':
    main()
