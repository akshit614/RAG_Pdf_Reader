from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2_model = GPT2LMHeadModel.from_pretrained('distilgpt2')
gpt2_tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_index(text):
    sentences = text.split('. ')
    embeddings = model.encode(sentences)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, sentences

def retrieve_documents(query, index, sentences, top_k=5):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k)
    return [sentences[i] for i in I[0]]

def generate_answer(context, query):
    input_text = f"{context}\nUser: {query}\nBot:"
    inputs = gpt2_tokenizer.encode(input_text, return_tensors='pt')
    outputs = gpt2_model.generate(inputs, max_length=1000, do_sample=True)
    return gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)
