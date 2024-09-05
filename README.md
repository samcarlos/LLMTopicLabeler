# LLMTopicLabeler

LLMs are incredibly good at labeling text without prior data. However, they have limitations, particularly when dealing with multiple topics. For example, asking an LLM to distinguish between 20 different topics in a single prompt can lead to confusion and inconsistent results. Moreover, it can be very time consuming or costly to have seperate prompts / queries for each individual topic.

The Solution: I propose using LLMs iteratively to label your dataset and build a supervised model. 

    1. Model Building:
        Embed all texts into some numeric representation (here I suggest using embeddings from an LLM but could be something else like TF-IDF)
        Embed the topic and create an initial supervised model with the topic as the positive class (1) and all other texts as the negative class (0).
        
    2. Iterative Refinement:
        Predict topic relevance for all texts.
        Send the top K predictions to an LLM for validation. If validated, adjust the labels to correspond that these are of the positive class.
        Retrain the model.
        
    3. Optimal Cutoff:
        Once the training set is complete, determine the optimal cutoff by analyzing prediction percentiles and the error rate. If that percentile’s error rate is above a certain threshold (here I use 50%) then stop and use the previous threshold.


Below is an example on how to use it this code in python.

```import numpy as np
import pandas as pd
import ollama 

from sklearn.datasets import fetch_20newsgroups
from LLMTopicLabeler import LLMTopicLabeler

#load data
newsgroups_train = fetch_20newsgroups(subset='train')

def embed_topic_text(text):
    # Use ollama to embed the topic text
    response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    temp_embedding = np.array(response["embedding"]).reshape(1,-1)
    return temp_embedding

#embed and append raw text to df
embeddings = [embed_topic_text(x) for x in newsgroups_train.data]
embeddings_pd = pd.DataFrame(np.array(embeddings)[:,0,:])
embeddings_pd.columns = ['embedding_' + str(x) for x in range(1024)]
embeddings_pd['paragraph'] = newsgroups_train.data

#build model
auto_classifier = LLMTopicLabeler()
auto_classifier.iterative_topic_classification('taxes', embeddings_pd, 
                                                                                            y_iterations = 5, quantiles_cutoff = [.999, .995,.99,.985])
#predict
predictions = auto_classifier.predict(embeddings_pd)

#print the top five predicted texts of the class
for x in np.where(predictions == 1)[0][:5]:
    print(newsgroups_train.data[x])
    print('\n')
    print('_______')
```
