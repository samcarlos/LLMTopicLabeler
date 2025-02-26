import json
import pickle
import ollama
import pandas as pd 
import numpy as np
import os

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
import re

class LLMTopicLabeler:
    """
    A class to handle the iterative process of topic classification using embeddings, 
    regression modeling, and refinement with a chat agent.
    
    Attributes:
        model (Ridge): The trained Ridge regression model.
        optimal_cutoff (float): The cutoff probability value for classifying a paragraph.
        refined_indices_bad_list (list): A list of indices for paragraphs that were refined and marked as not belonging to the topic.
        prompt_template (str): The default prompt template for the chat agent.
    """

    def __init__(self):
        self.model = None
        self.optimal_cutoff = None
        self.refined_indices_bad_list = []
        # Add default prompt template
        self.prompt_template = """I am trying to classify whether given texts contain or relate to a specific topic in conversations between patients and their care providers for maternity care. The topic is: '{topic_text}'.

For each text, determine if it contains or relates to the specified topic. Respond with 'Yes' or 'No' for each text and provide a confidence score between 0 and 1. Mentioning the topic in passing should be considered a 'Yes'.

Return your decisions in the form of a JSON array with each element containing keys: 'text_number', 'decision', 'confidence', 'reasoning'.

If the topic is not in English return 'No'.

{texts_formatted}"""

    def set_prompt_template(self, prompt_template):
        """
        Set a custom prompt template for the chat agent.
        
        The template must include {topic_text} and {texts_formatted} placeholders.
        
        Args:
            prompt_template (str): The prompt template to use
        """
        if "{topic_text}" not in prompt_template or "{texts_formatted}" not in prompt_template:
            raise ValueError("Prompt template must include {topic_text} and {texts_formatted} placeholders")
        self.prompt_template = prompt_template

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

    def refine_predictions_with_chat_agent(self, paragraphs, top_indices, topic_text, batch_size=1, verbose=True):
        """
        Refine predictions using a chat agent, processing multiple paragraphs at once.

        Args:
            paragraphs (list): The paragraphs to refine.
            top_indices (list): The indices of the top paragraphs.
            topic_text (str): The topic text for comparison.
            batch_size (int): Number of paragraphs to process in a single prompt.
            verbose (bool): Whether to print detailed progress information.

        Returns:
            list: Refined indices that match the topic.
            list: Refined indices that do not match the topic.
        """
        refined_indices = []
        refined_indices_bad = []
        
        # Process paragraphs in batches
        for i in range(0, len(top_indices), batch_size):
            batch_indices = top_indices[i:i + batch_size]
            batch_paragraphs = [paragraphs[idx] for idx in batch_indices]
            
            if verbose:
                print(f"\nProcessing batch {i//batch_size + 1}/{(len(top_indices) + batch_size - 1)//batch_size}")
            
            # Create numbered list of paragraphs
            texts_formatted = "\n\n".join([f"Text {j+1}:\n{para}" for j, para in enumerate(batch_paragraphs)])
            
            # Use the prompt template
            prompt = self.prompt_template.format(
                topic_text=topic_text,
                texts_formatted=texts_formatted
            )

            response = ollama.chat(
                model="deepseek-r1:14b",
                messages=[{"role": "user", "content": prompt}]
            )
            
            try:
                content = response["message"]["content"]
                match = re.search(r'```json\s*(\[[\s\S]*?\])\s*```', content, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    responses = json.loads(json_str)
                    
                    # Process each response in the batch
                    for j, resp in enumerate(responses):
                        idx = batch_indices[j]
                        is_relevant = 'Yes' in resp['decision']
                        
                        if verbose:
                            # Print a shortened version of the paragraph (first 100 chars)
                            short_text = paragraphs[idx][:100] + "..." if len(paragraphs[idx]) > 100 else paragraphs[idx]
                            print(f"\nText {j+1}:")
                            print(f"Content: {short_text}")
                            print(f"Decision: {resp['decision']} (confidence: {resp['confidence']:.2f})")
                            print(f"Reasoning: {resp['reasoning']}")
                        
                        if is_relevant:
                            refined_indices.append(idx)
                            if verbose:
                                print("✓ Included")
                        else:
                            refined_indices_bad.append(idx)
                            if verbose:
                                print("✗ Excluded")
                else:
                    if verbose:
                        print("No valid JSON found in response")
            except Exception as e:
                if verbose:
                    print(f"Error processing batch: {e}")
                continue

        if verbose:
            print(f"\nBatch processing complete. Found {len(refined_indices)} relevant and {len(refined_indices_bad)} non-relevant texts.")
        
        return refined_indices, refined_indices_bad

    def find_optimal_cutoff(self, paragraphs, probabilities, topic_text, target_percentage=0.6, num_obs_to_confirm=30, batch_size=30):
        """
        Find the optimal cutoff using binary search.
        
        Args:
            paragraphs (list): The paragraphs to classify.
            probabilities (np.array): The predicted probabilities.
            topic_text (str): The topic text for refinement.
            target_percentage (float): The target percentage of correct classifications.
            num_obs_to_confirm (int): Number of observations to check at each cutoff.
            batch_size (int): Number of paragraphs to process in a single prompt.
        
        Returns:
            float: The optimal cutoff value.
        """
        # Start with high confidence range (0.9 to 1.0)
        left = 0.9
        right = 1.0
        optimal_cutoff = None
        
        while right - left > 0.01:  # Stopping condition for precision
            mid = (left + right) / 2
            print(f"\nTesting cutoff: {mid:.3f}")
            
            # Get indices just above the cutoff
            above_cutoff_indices = np.where(probabilities > mid)[0]
            
            if len(above_cutoff_indices) == 0:
                # If no examples above cutoff, try lower range
                right = left
                left = max(0.0, left - 0.1)
                continue
            
            # Get probabilities for texts above cutoff
            temp_proba = probabilities[above_cutoff_indices]
            
            # Take texts just above the cutoff to test the boundary
            sorted_indices = np.argsort(temp_proba)
            boundary_indices = above_cutoff_indices[sorted_indices[:num_obs_to_confirm]]
            
            # Process in batches
            all_refined_indices = []
            for j in range(0, len(boundary_indices), batch_size):
                batch_indices = boundary_indices[j:j + batch_size]
                refined_indices, _ = self.refine_predictions_with_chat_agent(
                    paragraphs, 
                    batch_indices, 
                    topic_text,
                    batch_size=batch_size
                )
                all_refined_indices.extend(refined_indices)
            
            accuracy = len(all_refined_indices) / len(boundary_indices)
            print(f"Accuracy at cutoff {mid:.3f}: {accuracy:.2%}")
            
            if accuracy >= target_percentage:
                # If accuracy is good, try a lower cutoff
                optimal_cutoff = mid
                right = mid
            else:
                # If accuracy is poor, try a higher cutoff
                left = mid
        
        return optimal_cutoff or right

    def select_diverse_and_predicted_indices(self, embeddings, predictions, temp_y, top_predictions=5):
        """
        Select both diverse texts based on embeddings and top predicted texts.
        
        Args:
            embeddings (np.array): Embeddings of the texts
            predictions (np.array): Model predictions for each text
            temp_y (np.array): Current labels during iteration
            top_predictions (int): Number of top predicted texts to include
            
        Returns:
            list: Combined selected indices (top_predictions + 9 diverse texts)
        """
        # Add debug prints
        print("\nDebug info:")
        print(f"Total labels: {len(temp_y)}")
        print(f"Positive labels: {np.sum(temp_y[:-1] == 1)}")
        print(f"Available texts: {np.sum((temp_y[:-1] == 0) & ~np.isin(np.arange(len(temp_y)-1), self.refined_indices_bad_list))}")
        
        # Get all positive embeddings from the full dataset
        all_embeddings = self.temp_x[:-1]  # Exclude topic embedding
        positive_mean = np.mean(all_embeddings[temp_y[:-1] == 1], axis=0) if np.sum(temp_y[:-1] == 1) > 0 else None
        
        print("\nSelecting diverse indices:")
        
        if positive_mean is None:
            print("No positive examples yet, selecting top predictions only")
            selected = list(np.argsort(predictions)[-top_predictions:])
            print(f"Selected {len(selected)} indices from predictions")
            return selected
        
        # Get mask for available indices using temp_y
        available_mask = np.ones(len(predictions), dtype=bool)  # Work with candidate predictions length
        
        if len(available_mask) == 0:
            print("No available indices to select from")
            return []
        
        # Get available embeddings
        available_embeddings = embeddings[available_mask]
        
        # Get top prediction indices
        available_predictions = predictions[available_mask]
        top_pred_relative_indices = np.argsort(available_predictions)[-top_predictions:]
        top_pred_indices = np.where(available_mask)[0][top_pred_relative_indices]
        
        # Get diverse indices using embedding similarity
        # Normalize the query vector
        query = positive_mean / np.linalg.norm(positive_mean)
        
        # Initialize for diverse selection
        selected_indices = []
        remaining_mask = np.ones(len(available_mask), dtype=bool)
        current_query = query.copy()
        
        # Remove top prediction indices from consideration for diversity selection
        for idx in top_pred_relative_indices:
            remaining_mask[idx] = False
        
        # Select diverse indices
        for _ in range(3):  # We'll do this 3 times
            if not np.any(remaining_mask):
                break
            
            # Calculate cosine similarity with current query
            similarities = np.dot(available_embeddings[remaining_mask], current_query)
            
            # Get top 3 most similar texts (or fewer if not enough available)
            num_remaining = np.sum(remaining_mask)
            n_to_select = min(3, num_remaining)
            top_3_relative = np.argsort(similarities)[-n_to_select:]
            top_3_actual = np.where(remaining_mask)[0][top_3_relative]
            
            # Calculate mean embedding of these 3 texts
            selected_vector = np.mean(available_embeddings[top_3_actual], axis=0)
            selected_vector_norm = selected_vector / np.linalg.norm(selected_vector)
            
            # Add all 3 indices
            for idx in top_3_actual:
                original_idx = np.where(available_mask)[0][idx]
                selected_indices.append(original_idx)
                remaining_mask[idx] = False
            
            # Update query by subtracting projection of mean vector
            projection = np.dot(current_query, selected_vector_norm) * selected_vector_norm
            current_query = current_query - projection
            current_query = current_query / (np.linalg.norm(current_query) + 1e-10)
        
        # Combine and return both sets of indices
        combined_indices = list(top_pred_indices) + selected_indices
        
        # Move print statements here before return
        print("\nSelection summary:")
        print(f"- From predictions ({len(top_pred_indices)}): {top_pred_indices}")
        print(f"- From diversity ({len(selected_indices)}): {selected_indices}")
        print(f"Total indices selected: {len(combined_indices)}")
        
        return combined_indices

    def iterative_topic_classification(self, topic_text, embeddings_df, y_iterations=50, 
                                     num_obs_stopping_criteria=350, 
                                     var_names=['embedding_' + str(x) for x in range(1024)],
                                     text_col='paragraph',
                                     target_percentage=0.3, 
                                     num_obs_to_confirm=20,
                                     batch_size=1,
                                     top_predictions=5):
        """
        Perform iterative topic classification.

        Args:
            topic_text (str): The topic text for classification.
            embeddings_df (pd.DataFrame): The DataFrame containing embeddings and paragraphs.
            y_iterations (int): The number of iterations for refinement.
            num_obs_stopping_criteria (int): Number of positive examples to stop at
            var_names (list): Names of embedding columns
            text_col (str): Name of text column
            target_percentage (float): Target accuracy for cutoff
            num_obs_to_confirm (int): Number of observations to check at each cutoff
            batch_size (int): Batch size for processing
            top_predictions (int): Number of top predictions to include in diversity selection
        """
        self.var_names = var_names
        self.topic_text = topic_text
        topic_embedding = self.embed_topic_text(topic_text)
        temp_y = np.concatenate([np.zeros(embeddings_df.shape[0]), [1]])
        temp_x = np.concatenate([embeddings_df[var_names].values, topic_embedding], axis=0)
        self.temp_x = temp_x  # Store temp_x as instance variable

        self.build_initial_model(temp_x, temp_y)
        predictions = self.model.predict(embeddings_df[var_names])

        for iteration in range(y_iterations):
            
            embeddings = embeddings_df[var_names].values
            candidate_indices = np.where((temp_y == 0) & 
                                       ~np.isin(np.arange(len(temp_y)), self.refined_indices_bad_list))[0]
            
            if len(candidate_indices) > 0:
                candidate_predictions = predictions[candidate_indices]
                candidate_embeddings = embeddings[candidate_indices]
                
                print(f"\nIteration {iteration + 1}:")
                print(f"Total positive examples so far: {np.sum(temp_y == 1)}")
                print(f"Candidates available: {len(candidate_indices)}")
                
                # Pass full temp_y instead of just the candidate subset
                combined_indices = self.select_diverse_and_predicted_indices(
                    candidate_embeddings,
                    candidate_predictions,
                    temp_y,  # Pass full temp_y instead of temp_y[candidate_indices]
                    top_predictions=top_predictions
                )
                
                if combined_indices:
                    # Convert relative indices back to original indices
                    top_indices = [candidate_indices[i] for i in combined_indices]
                    #print the top indices
                    print('Top Indices:')
                    print(len(top_indices))
                    refined_indices, refined_indices_bad = self.refine_predictions_with_chat_agent(
                        list(embeddings_df[text_col]), 
                        top_indices, 
                        topic_text,
                        batch_size=len(top_indices)
                    )
                    
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
            else:
                print('No candidate texts found for refinement')
                break

        print('Beginning to final cutoff')
        self.optimal_cutoff = self.find_optimal_cutoff(np.array(embeddings_df[text_col]), predictions, topic_text, target_percentage=target_percentage, num_obs_to_confirm=num_obs_to_confirm)
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

    def read_all_pickled_dataframes(self, directory: str, twilio_message_ids = None, subset = True):
        """
        Reads all pickled DataFrames in a given directory and returns a list of DataFrames.
        
        Parameters:
        directory (str): The directory containing the pickled files.
        
        Returns:
        List[pd.DataFrame]: A list of DataFrames loaded from the pickled files.
        """
        # List to store individual DataFrames
        dataframes = []

        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.pkl') or filename.endswith('.pickle'):
                filepath = os.path.join(directory, filename)
                if os.path.getsize(filepath) > 0:
                    with open(filepath, 'rb') as file:
                        df = pickle.load(file)

                        if df.shape[0] > 10000 and subset:  # Changed & to and
                            print(f"Sampling from DataFrame with {df.shape[0]} rows")
                            df = df.sample(10000)
                        if twilio_message_ids:
                            df = df.loc[df['twilio_message_id'].isin(twilio_message_ids)]
                        if isinstance(df, pd.DataFrame):
                            dataframes.append(df)
        dataframes = pd.concat(dataframes, ignore_index=True)
        return dataframes

tone_prompt = """You are an expert in analyzing communication tone and emotional resonance in healthcare conversations.

Analyze the following text(s) for the presence of a {topic_text} tone. Focus on HOW the message is conveyed and received, not the specific content.

Consider these aspects:
- Language choices and phrasing
- Emotional undertones
- How the message might make the recipient feel
- Non-verbal cues evident in the text
- Overall communication style

{texts_formatted}

For each text, determine if it demonstrates a {topic_text} tone. Respond with:
- 'Yes' if the tone is clearly present
- 'No' if the tone is absent or different

Format your response as JSON:
[
  {{
    "text_number": N,
    "decision": "Yes/No",
    "confidence": 0.XX,
    "reasoning": "Analysis of tone and emotional impact..."
  }}
]"""

# Usage example:
labeler = LLMTopicLabeler()
labeler.set_prompt_template(tone_prompt)
