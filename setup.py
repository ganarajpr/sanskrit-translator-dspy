import dspy
import json
import os
from dspy import Example
from dotenv import load_dotenv
import numpy as np
import ollama

load_dotenv()


def init_ollama_model():
    ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434/v1')
    mistral_model = dspy.OpenAI(api_base=ollama_url.rstrip('/') + '/', api_key='ollama', 
                               model='mistral:7b-instruct-v0.2-q6_K', stop='\n\n', model_type='chat', temperature=0.423)
    dspy.settings.configure(lm=mistral_model, trace=[])
    return mistral_model

def init_openai_model():
    openai_model = dspy.OpenAI(model='gpt-4o', temperature=0.2)
    dspy.settings.configure(lm=openai_model, trace=[])
    return openai_model

def get_mistral_config():
    ollama_url = os.getenv('OLLAMA_URL', 'http://localhost:11434/v1')
    mistral_model = dspy.OpenAI(api_base=ollama_url.rstrip('/') + '/', api_key='ollama', 
                               model='mistral:7b-instruct-v0.2-q6_K', stop='\n\n', model_type='chat')
    return mistral_model

def get_openai_config():
    api_key = os.getenv('OPENAI_API_KEY')
    return dspy.OpenAI(model='gpt-4o', api_base='https://api.openai.com/v1/', api_key=api_key)

def import_json(filename):
    print(filename)
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def create_examples(data, inputs):
    output = []
    for example in data:
        output.append(Example(base=example).with_inputs(*inputs))
    return output
    
def cosine_similarity(embedding1, embedding2):
    """
    Compute the cosine similarity between two embeddings.

    Parameters:
    embedding1 (numpy array): The first embedding vector.
    embedding2 (numpy array): The second embedding vector.

    Returns:
    float: The cosine similarity between the two embeddings.
    """
    # Ensure the inputs are numpy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Compute the dot product between the embeddings
    dot_product = np.dot(embedding1, embedding2)
    
    # Compute the norm (magnitude) of each embedding
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Compute the cosine similarity
    cosine_sim = dot_product / (norm1 * norm2)
    
    return cosine_sim
    
def get_cosine_similarity(answer, prediction):
    answer_embed = ollama.embeddings(model='mistral:7b-instruct-v0.2-q6_K', prompt=str(answer))
    pred_embed = ollama.embeddings(model='mistral:7b-instruct-v0.2-q6_K', prompt=str(prediction))
    cos_sim = cosine_similarity(answer_embed['embedding'], pred_embed['embedding'])
    return cos_sim
