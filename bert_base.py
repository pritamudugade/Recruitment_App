from flask import Flask, request, render_template
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# Load candidate data
df = pd.read_csv('candidates.csv')

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Calculate BERT embeddings for job descriptions
job_descriptions = df['Designation'] + ' ' + df['Expertise']
job_descriptions = job_descriptions.tolist()
job_description_embeddings = []

for jd in job_descriptions:
    inputs = tokenizer(jd, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    job_description_embeddings.append(embeddings)

job_description_embeddings = np.vstack(job_description_embeddings)

@app.route('/')
def index():
    return render_template('index_advanced.html')

@app.route('/match', methods=['POST'])
def match():
    job_description = request.form.get('job_description')
    experience = int(request.form.get('experience'))

    # Calculate BERT embedding for the input job description
    inputs = tokenizer(job_description, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    input_embedding = outputs.last_hidden_state.mean(dim=1).numpy()

    # Calculate cosine similarities between input and candidate job descriptions
    similarities = cosine_similarity(input_embedding, job_description_embeddings).flatten()

    # Create a DataFrame to store candidate data along with similarity scores
    candidates_with_similarity = df.copy()
    candidates_with_similarity['Similarity Score'] = similarities

    # Filter candidates based on experience and similarity score
    filtered_candidates = candidates_with_similarity[
        (candidates_with_similarity['Experience (years)'] >= experience) &
        (candidates_with_similarity['Similarity Score'] > 0.6)  # Adjust the threshold as needed
    ]

    return render_template('index_advanced.html', candidates=filtered_candidates.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
