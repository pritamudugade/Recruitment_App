from flask import Flask, request, render_template
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load candidate data
df = pd.read_csv('candidates.csv')

# Load BERT model and tokenizer
model_name = "bert-base-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Define the weights for different fields
field_weights = {
    'Name': 0.1,
    'Designation': 0.2,
    'Domain': 0.3,
    'Experience (years)': 0.2,
    'Expertise': 0.1,
    'Languages': 0.05,
    'Certifications': 0.05,
}

# Create a function to calculate the relevance score
def calculate_relevance(job_description, candidate_data):
    total_similarity = 0
    total_weight = 0

    # Tokenize the job description
    job_desc_tokens = tokenizer.tokenize(job_description)

    for field, weight in field_weights.items():
        if field in candidate_data:
            # Tokenize and calculate BERT embeddings for the candidate field
            candidate_text = str(candidate_data[field])
            candidate_tokens = tokenizer.tokenize(candidate_text)
            with torch.no_grad():
                job_desc_inputs = tokenizer(job_description, return_tensors="pt", padding=True, truncation=True)
                candidate_inputs = tokenizer(candidate_text, return_tensors="pt", padding=True, truncation=True)
                job_desc_outputs = model(**job_desc_inputs)
                candidate_outputs = model(**candidate_inputs)
            # Calculate cosine similarity between the two embeddings
            similarity = cosine_similarity(job_desc_outputs.last_hidden_state[0], candidate_outputs.last_hidden_state[0])
            # Update total similarity and total weight
            total_similarity += weight * similarity[0][0]
            total_weight += weight

    # Calculate the weighted average similarity
    if total_weight > 0:
        weighted_similarity = total_similarity / total_weight
    else:
        weighted_similarity = 0

    return weighted_similarity

@app.route('/')
def index():
    return render_template('index_advanced.html')

@app.route('/match', methods=['POST'])
def match():
    job_description = request.form.get('job_description')
    experience = int(request.form.get('experience'))

    # Calculate relevance scores for all candidates
    df['Relevance Score'] = df.apply(lambda row: calculate_relevance(job_description, row), axis=1)

    # Filter candidates based on experience and relevance score
    filtered_candidates = df[
        (df['Experience (years)'] >= experience) &
        (df['Relevance Score'] > 0.6)  # Adjust the threshold as needed
    ]

    return render_template('index_advanced.html', candidates=filtered_candidates.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
