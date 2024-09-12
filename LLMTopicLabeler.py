import json
import pickle
import ollama
import pandas as pd 
import numpy as np

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import roc_auc_score


class LLMTopicLabeler:
    """
    A class to handle the iterative process of topic classification using embeddings, 
    regression modeling, and refinement with a chat agent.
    
    Attributes:
        model (Ridge): The trained Ridge regression model.
        optimal_cutoff (float): The cutoff probability value for classifying a paragraph.
        refined_indices_bad_list (list): A list of indices for paragraphs that were refined and marked as not belonging to the topic.
    """

    def __init__(self):
        self.model = None
        self.optimal_cutoff = None
        self.refined_indices_bad_list = []

    def embed_topic_text(self, topic_text: str) -> np.array:
        """
        Embed the topic text using Ollama's embedding model.

        Args:
            topic_text (str): The topic text to be embedded.

        Returns:
            np.array: The embedding of the topic text.
        """
        response = ollama.embeddings(model="mxbai-embed-large", prompt=topic_text)
        temp_embedding = np.array(response["embedding"]).reshape(1, -1)
        return temp_embedding

    def build_initial_model(self, X_train, y_train):
        """
        Build and train a Ridge regression model.

        Args:
            X_train (np.array): The training data.
            y_train (np.array): The labels for training data.

        Returns:
            Ridge: The trained Ridge regression model.
        """
        model = Ridge(max_iter=1000)
        model.fit(X_train, y_train)
        self.model = model

    def refine_predictions_with_chat_agent(self, paragraphs, top_indices, topic_text):
        """
        Refine predictions using a chat agent.

        Args:
            paragraphs (list): The paragraphs to refine.
            top_indices (list): The indices of the top paragraphs.
            topic_text (str): The topic text for comparison.
            threshold (float): The confidence threshold for refinement.

        Returns:
            list: Refined indices that match the topic.
            list: Refined indices that do not match the topic.
        """
        refined_indices = []
        refined_indices_bad = []
        for idx in top_indices:
            paragraph = paragraphs[idx]
            prompt = f"""The topic is: '{topic_text}'. 
            Does the given text contain or relate to the specified topic? Respond with 'Yes' or 'No' and provide a confidence score between 0 and 1. Mentioning the topic in passing should be considered a 'Yes'.
            
            Return only your decision and reasoning in the form of a json with keys: 'decision', 'confidence', 'reasoning'.
            Text: 
            {paragraph}"""

            response = ollama.chat(
                model="llama3.1:latest",
                messages=[{"role": "user", "content": prompt}]
            )
            try:
                print(paragraph)
                response_dict = json.loads(response["message"]["content"])
                print(response["message"]["content"])
                if 'Yes' in response_dict['decision']:
                    refined_indices.append(idx)
                else:
                    refined_indices_bad.append(idx)
            except:
                pass

        return refined_indices, refined_indices_bad

    def find_optimal_cutoff(self, paragraphs, probabilities, topic_text, quantiles, target_percentage=0.6, num_obs_to_confirm = 30):
        """
        Find the optimal cutoff based on the specified quantiles.

        Args:
            paragraphs (list): The paragraphs to classify.
            probabilities (np.array): The predicted probabilities.
            topic_text (str): The topic text for refinement.
            quantiles (list): The list of quantiles to evaluate.
            target_percentage (float): The target percentage of correct classifications.

        Returns:
            float: The optimal cutoff value.
            float: The actual percentage of correct classifications.
        """
        for i, quantile in enumerate(quantiles):
            cutoff = np.quantile(probabilities, quantile)
            temp_proba = probabilities[np.where(probabilities > cutoff)[0]]
            top_indices = np.argsort(temp_proba)[:num_obs_to_confirm]
            refined_indices, _ = self.refine_predictions_with_chat_agent(paragraphs[np.where(probabilities > cutoff)[0]], top_indices, topic_text)

            if len(refined_indices) / len(top_indices) < target_percentage:
                self.optimal_cutoff = np.quantile(probabilities, quantiles[i-1])
                return np.quantile(probabilities, quantiles[i-1]), len(refined_indices) / len(top_indices)

        return None, None

    def iterative_topic_classification(self, topic_text, embeddings_df, y_iterations=50, num_obs_stopping_criteria = 350, var_names = ['embedding_' + str(x) for x in range(1024)],
                                       quantiles_cutoff = [.9999,.9995,.999,.995,.99,.9], text_col = 'paragraph',
                                       target_percentage=0.3, num_obs_to_confirm = 20):
        """
        Perform iterative topic classification.

        Args:
            topic_text (str): The topic text for classification.
            embeddings_df (pd.DataFrame): The DataFrame containing embeddings and paragraphs.
            y_iterations (int): The number of iterations for refinement.

        Returns:
            Ridge: The trained model.
            np.array: The in-sample predictions.
            np.array: The final labels.
            float: The optimal cutoff value.
        """
        self.var_names = var_names
        self.topic_text = topic_text
        topic_embedding = self.embed_topic_text(topic_text)
        temp_y = np.concatenate([np.zeros(embeddings_df.shape[0]), [1]])
        temp_x = np.concatenate([embeddings_df[var_names].values, topic_embedding], axis=0)

        self.build_initial_model(temp_x, temp_y)
        predictions = self.model.predict(embeddings_df[var_names])

        for iteration in range(y_iterations):
            top_x = 25 
            top_indices = np.argsort(predictions)
            top_indices = [i for i in top_indices if i not in np.where(temp_y == 1)[0]]
            top_indices = [i for i in top_indices if i not in self.refined_indices_bad_list]
            top_indices = top_indices[-top_x:]
            refined_indices, refined_indices_bad = self.refine_predictions_with_chat_agent(embeddings_df[text_col], top_indices, topic_text)
            self.refined_indices_bad_list.extend(refined_indices_bad)
            temp_y[refined_indices] = 1
            model = RidgeCV(alphas=[.1, 1.0, 5, 10.0, 50 , 100, 500, 1000] )
            model.fit(temp_x, temp_y)
            self.model = model

            predictions = self.model.predict(embeddings_df[self.var_names])
            print(temp_y.sum())
            if temp_y.sum() > num_obs_stopping_criteria: 
                print('Finished Finding Examples')
                break
        print('Beginning to final cutoff')
        self.optimal_cutoff, _ = self.find_optimal_cutoff(np.array(embeddings_df[text_col]), predictions, topic_text, quantiles=quantiles_cutoff, target_percentage = target_percentage, num_obs_to_confirm=num_obs_to_confirm )
        self.labels = temp_y

    def predict(self, embeddings):
        """
        Predict using the trained model and the optimal cutoff.

        Args:
            embeddings (np.array): The embeddings for prediction.

        Returns:
            np.array: The predicted labels based on the optimal cutoff.
        """
        if self.model is None or self.optimal_cutoff is None:
            raise ValueError("Model is not trained or optimal cutoff is not set.")
        
        predictions = self.model.predict(embeddings[self.var_names])
        return (predictions >= self.optimal_cutoff).astype(int)

    def save_model(self, filename):
        """
        Save the trained model and the optimal cutoff.

        Args:
            filename (str): The file path to save the model.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename):
        """
        Load the model from a file.

        Args:
            filename (str): The file path to load the model from.

        Returns:
            TopicClassifier: The loaded model.
        """
        with open(filename, 'rb') as file:
            return pickle.load(file)
