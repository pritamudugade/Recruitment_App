from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

# Load candidate data
df = pd.read_csv('candidates.csv')

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Calculate TF-IDF matrix for candidate designations and expertise
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Designation'] + ' ' + df['Expertise'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    job_description = request.form.get('job_description')
    experience = int(request.form.get('experience'))

    # Calculate TF-IDF for the input job description
    job_description_tfidf = tfidf_vectorizer.transform([job_description])

    # Calculate cosine similarity between the input and candidate designations/expertise
    cosine_similarities = linear_kernel(job_description_tfidf, tfidf_matrix).flatten()

    # Sort candidates by similarity score
    candidates_ranked = df.copy()
    candidates_ranked['Similarity Score'] = cosine_similarities
    candidates_ranked.sort_values(by='Similarity Score', ascending=False, inplace=True)

    # Filter candidates by experience
    filtered_candidates = candidates_ranked[candidates_ranked['Experience (years)'] >= experience]

    # Filter candidates with a similarity score greater than 60%
    filtered_candidates = filtered_candidates[filtered_candidates['Similarity Score'] > 0.6]

    if not filtered_candidates.empty:
        # Create a list of candidate information including name, designation, and similarity score
        candidate_records = []
        for index, row in filtered_candidates.iterrows():
            candidate_info = {
                'Name': row['Name'],
                'Designation': row['Designation'],
                'Similarity Score': row['Similarity Score']
            }
            candidate_records.append(candidate_info)

        return render_template('index.html', candidates=candidate_records)
    else:
        return render_template('index.html', no_candidates=True)

if __name__ == '__main__':
    app.run(debug=True)
