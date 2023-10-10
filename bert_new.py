from flask import Flask, request, render_template
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load candidate data
df = pd.read_csv('candidates.csv')

# Load BERT model and tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

# Define the list of columns to consider for relevance calculation
relevant_columns = ['Name', 'Designation', 'Domain', 'Experience (years)', 'Expertise', 'Languages', 'Certifications', 'Work History']

# Create a function to calculate the relevance score
def calculate_relevance(job_description, candidate_data):
    # Combine all relevant fields for the candidate into a single string
    candidate_text = ' '.join([str(candidate_data[column]) for column in relevant_columns])

    # Tokenize the job description and candidate text
    inputs = tokenizer(job_description, candidate_text, return_tensors="pt", padding=True, truncation=True)
    
    # Calculate BERT embeddings for job description and candidate text
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Print the shape of outputs.last_hidden_state to debug
    print("Shape of outputs.last_hidden_state:", outputs.last_hidden_state.shape)

    # Calculate cosine similarity between the two embeddings
    similarity = cosine_similarity(outputs.last_hidden_state[0], outputs.last_hidden_state[0])  # Use [0] for the single pair of inputs
    
    # Print the calculated similarity for debugging
    print("Calculated Similarity:", similarity)

    return similarity[0][0]

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
