import numpy as np
import pandas as pd
import ollama 

from sklearn.datasets import fetch_20newsgroups
from LLMTopicLabeler import LLMTopicLabeler

#load data
newsgroups_train = fetch_20newsgroups(subset='train')

def embed_topic_text(topic_text: str) -> np.array:
    # Use ollama to embed the topic text
    response = ollama.embeddings(model="mxbai-embed-large", prompt=topic_text)
    temp_embedding = np.array(response["embedding"]).reshape(1,-1)
    return temp_embedding

#embed and append raw text to df
embeddings = [embed_topic_text(x) for x in newsgroups_train.data]
embeddings_pd = pd.DataFrame(np.array(embeddings)[:,0,:])
embeddings_pd.columns = ['embedding_' + str(x) for x in range(1024)]
embeddings_pd['paragraph'] = newsgroups_train.data

#build model
auto_classifier = LLMTopicLabeler()
model, predictions, labels, optimal_cutoff = auto_classifier.iterative_topic_classification('taxes', embeddings_pd, 
                                                                                            y_iterations = 5, quantiles_cutoff = [.999, .995,.99,.985])
#predict
predictions = auto_classifier.predict(embeddings_pd)

#print the top five predicted texts of the class
for x in np.where(predictions == 1)[0][:5]:
    print(newsgroups_train.data[x])
    print('\n')
    print('_______')