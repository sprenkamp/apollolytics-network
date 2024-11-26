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
    def __init__(self, embedding_model, df, input_file_name):
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
                return np.array(result, dtype="float32")
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
            df (pd.DataFrame): DataFrame containing the documents.
            input_file_name (str): The file name to save the DataFrame with embeddings.
            text_column (str): Name of the column containing the text documents.
            embedding_column (str): Name of the column to store/retrieve embeddings.
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
                embeddings_list = [np.fromstring(embedding.strip("[]"), sep=',') for embedding in embeddings_list]
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
        self.df["embedding"] = embeddings
        self.df.to_csv(self.input_file_name, index=False)
        print(f"Embeddings saved to {self.input_file_name}")
        # Concatenate the list of embeddings into a single NumPy array
        embeddings_array = np.vstack(embeddings)
        return embeddings_array

#define stopwords
stopWords = stopwords.words('english') 
for word in stopwords.words('german'):
    stopWords.append(word)
for word in stopwords.words('russian'):
    stopWords.append(word)
with open("data/stopwords/stopwords_ua.txt") as file: #add ukrainian stopwords loaded from .txt file
    ukrstopWords = [line.rstrip() for line in file]
for stopwords in ukrstopWords:
    stopWords.append(stopwords)

vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words=stopWords) #define vectorizer model with stopwords

def validate_file(f): #function to check if file exists
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f

class BERTopicAnalysis:
    """
    The following Class trains a BERTopic model on a given input file 
    and saves the model and visualizations in the given output folder.
    If the model already exists, meaning the output_folder already contains a BERTopic model,
    it can be loaded and solely used for inference.

    Parameters for the class:
    input_file: path to the input file
    output_folder: path to the output folder
    k_cluster: number of clusters to be used for the model
    do_inference: boolean to indicate if the model should also be used for inference, 
                predicting the class of each text line in the input file
    """

    # initialize class
    def __init__(self, input_file, output_folder, k_cluster, do_inference, embedding_model="text-embedding-3-large"):
        self.input_file = input_file
        # add time to output folder to avoid overwriting existing folders in format DDMMYYYY_HHMMSS
        self.output_folder = output_folder + f"{os.path.basename(input_file).split('.')[0]}_{k_cluster}_" + time.strftime("%d%m%Y_%H%M%S")
        self.k_cluster = k_cluster 
        self.do_inference = do_inference
        self.embedding_model = embedding_model

    # read input file and prepare data for BERTopic
    def read_data(self):
        self.df = pd.read_csv(self.input_file)
        self.df = self.df[self.df['messageText'].map(type) == str]
        self.df["messageText"] = self.df['messageText'].str.split().str.join(' ')
        # lines = self.df[self.df['messageText'].str.len() >= 100].messageText.values
        # self.text_to_analyse_list = [line.rstrip() for line in lines]
        self.text_to_analyse_list = self.df['messageText'].values
        print(f"Number of texts to analyse: {len(self.text_to_analyse_list)}")

    # load potentially existing model
    def read_model(self):
        print("loading model")
        self.model=BERTopic.load(f"{self.output_folder}/BERTopicmodel")


    # check if k_cluster is numeric and convert to int if so.
    # this is necessary for BERTopic if the cluster number is given as a string, 
    # as BERTopic can be set to automatically determine the number of clusters using the string "auto".
    def k_cluster_type(self):
        if self.k_cluster.isnumeric():
            self.k_cluster = int(self.k_cluster)

    # train BERTopic model we use basic parameters for the model, 
    # using basic umap_model and hdbscan_model,
    # as defined in the BERTopic documentation
    def fit_BERTopic(self):
        custom_embedder = CustomEmbedder(embedding_model=self.embedding_model,
                                        df=self.df,
                                        input_file_name=self.input_file
                                        )
        umap_model = UMAP(n_neighbors=100, n_components=10, metric='cosine', low_memory=False, random_state=42)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=25)
        # hdbscan_model = HDBSCAN(min_cluster_size=50, cluster_selection_method='leaf', prediction_data=True)
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        openai_model = "gpt-4o-mini"
        custom_prompt = """
Please note that the following messages are likely related to the Ukraine-Russia conflict.

And the following representative documents:
[DOCUMENTS]

Based on this information, please provide a concise and highly specific label that best represents this topic, focusing on the details pertinent to the Ukraine-Russia conflict. Don't create any labels that are too broad or general, like "Russia-Ukraine conflict coverage and commentary" or "Conflict and Military Operations in Ukraine and Russia". Good labels are specific and detailed for each topic, like "Wagner Group Putsch", "Azov Battalion in Mariupol", "Rumours about Zelensky". Still the label needs to be representative of the topic as a whole.
"""
#Good labels are specific and detailed, like "Wagner Group Putsch", "Azov Battalion in Mariupol", "Rumours about Zelensky".

        tokenizer= tiktoken.encoding_for_model(openai_model)
        representation_model = OpenAI(
            client,
            model=openai_model, 
            prompt=custom_prompt,
            # delay_in_seconds=2, 
            chat=True,
            nr_docs=50,
            # doc_length=100,
            # tokenizer=tokenizer
        )
        from bertopic.backend import OpenAIBackend
        self.model = BERTopic(verbose=True,
                            # language="multilingual",
                            # nr_topics=self.k_cluster, 
                            embedding_model = custom_embedder,
                            vectorizer_model=vectorizer_model,
                            representation_model=representation_model,
                            umap_model=umap_model,
                            hdbscan_model=kmeans,
                            )
        topics, probs = self.model.fit_transform(self.text_to_analyse_list)

    # save model and visualizations
    def save_results(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        fig = self.model.visualize_topics()
        fig.write_html(f"{self.output_folder}/bert_topic_model_distance_model.html")
        fig = self.model.visualize_hierarchy()
        fig.write_html(f"{self.output_folder}/bert_topic_model_hierarchical_clustering.html")
        fig = self.model.visualize_barchart(top_n_topics=30)
        fig.write_html(f"{self.output_folder}/bert_topic_model_word_scores.html")
        fig = self.model.visualize_heatmap()
        fig.write_html(f"{self.output_folder}/bert_topic_model_word_heatmap.html")
        self.model.save(f"{self.output_folder}/BERTopicmodel", serialization="pytorch")
    
    # save representative documents for each topic
    def write_multi_sheet_excel(self):
        writer = pd.ExcelWriter(f"{self.output_folder}/representative_docs.xlsx", engine='xlsxwriter')
        for i in self.model.get_representative_docs().keys():
            df = pd.DataFrame(self.model.get_representative_docs()[i], columns=['message'])
            sheet_name = ''.join([c for c in self.model.get_topic_info()[self.model.get_topic_info()['Topic']==i]['Name'].values[0][:31] if c not in '[]:*?/\\'])
            df.to_excel(writer, sheet_name=sheet_name)
        writer._save()
        self.model.get_topic_info().to_csv(f"{self.output_folder}/topic_info.csv")

    # predict the class of each text line in the input file
    def inference(self):
        pred, prob = self.model.transform(self.df['messageText'].values)
        self.df['cluster'] = pred
        self.df.to_csv(f"{self.output_folder}/df_model.csv", index=False)

    # run all functions
    def run_all(self):
        self.read_data()
        if os.path.exists(f"{self.output_folder}/BERTopicmodel"):
            self.read_model()
        else:
            self.k_cluster_type()
            self.fit_BERTopic()
            self.save_results()
            self.write_multi_sheet_excel()
        if self.do_inference:
            self.inference()

def main():
    # define parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file', help="Specify the input file", type=validate_file, required=True) #TODO change to argparse.FileType('r')
    parser.add_argument('-o', '--output_folder', help="Specify folder for results", required=True)
    parser.add_argument('-k', '--k_cluster', help="number of topic cluster", required=False, default="auto")
    parser.add_argument('-di', '--do_inference', help="does inference on data", action='store_true')
    args = parser.parse_args()
    # initialize class
    BERTopic_Analysis = BERTopicAnalysis(args.input_file,
                                         args.output_folder,
                                         args.k_cluster,
                                         args.do_inference
                                         )
    # run all functions
    BERTopic_Analysis.run_all()

if __name__ == '__main__':
    main()