import pickle
import pandas as pd
from fuzzywuzzy import process
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Load pre-trained models and dataset
similarity_matrix = pickle.load(open('similarity_matrix.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
data = pd.read_csv('Coursera.csv', encoding='utf-8')

# Step 1: Remove duplicates based on specific columns
data = data.drop_duplicates(subset=['Course Name', 'University', 'Difficulty Level', 'Course Rating',
                                    'Course URL', 'Course Description'])

# Preprocessing setup (same as in the notebook)
lemmatizer = WordNetLemmatizer()

def clean_name(text):
    """Clean text by removing special characters and lemmatizing words"""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()  # Convert text to lowercase
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])  # Lemmatize words
    return text

def clean_description(text):
    """Clean text by removing special characters and lemmatizing words"""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text

# Preprocess the course names (as they are used for matching and recommendations)
data['Cleaned Course Name'] = data['Course Name'].apply(clean_name)
data['Course Description'] = data['Course Description'].apply(clean_description)

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Home page

@app.route('/recommend', methods=['POST'])
def recommend():
    course_name = request.form['course_name']  # Get course name from user input
    
    # Preprocess user input (same as the dataset preprocessing)
    cleaned_course_name = clean_name(course_name)
    
    # Get recommendations based on the cleaned course name
    recommendations = get_recommendations(cleaned_course_name)
    
    return render_template('recommendations.html', recommendations=recommendations)

def get_recommendations(course_name, top_n=3, threshold=90, rating_weight=0.05):
    """
    Function to get course recommendations by matching course names and adjusting for course ratings.
    
    Parameters:
    - course_name (str): The course name entered by the user
    - top_n (int): The number of top recommendations to return
    - threshold (int): The minimum similarity score required for recommendations
    - rating_weight (float): Weight factor to adjust the impact of course ratings
    
    Returns:
    - recommendations (list): A list of course recommendations
    """
    match = process.extractOne(course_name, data['Course Name'])
    
    if match:
        matched_courses = data[data['Course Name'] == match[0]]
        
        if matched_courses.empty:  # If no match is found in the DataFrame
            return []
        
        course_idx = matched_courses.index[0]  # Access the first index
        
        score = match[1]
        
        # If the similarity score is below the threshold, return an empty list
        if score < threshold:
            return []
        
        # Get similarity scores for the course
        similarity_scores = list(enumerate(similarity_matrix[course_idx]))
        
        # Filter out the matched course itself and sort by similarity
        course_list = [item for item in similarity_scores if item[0] != course_idx]
        course_list = sorted(course_list, key=lambda x: x[1], reverse=True)[:top_n]
        
        # Prepare the recommendations as a list of dictionaries (course name, image, link, similarity)
        recommendations = []
        for idx, similarity_score in course_list:
            course_url = data.iloc[idx].get('Course URL', '')  # Assuming 'Course URL' column in CSV
            course_name = data.iloc[idx]['Course Name']
            rating = data.iloc[idx].get('Course Rating', '0')  # Default rating to '0' if not available
            institution = data.iloc[idx].get('University', 'Unknown Institution')  # Get the institution name
            description = data.iloc[idx].get('Course Description', 'No description available')  # Get course description
            
            try:
                # Convert the rating to a float
                rating = float(rating)
            except ValueError:
                rating = 0  # If rating is not numeric, set it to 0
            
            # Normalize the rating to a scale of 0 to 1 (assuming rating is between 0 and 5)
            normalized_rating = (rating - 0) / (5 - 0)
            
            # Final score is a weighted average of similarity and normalized rating
            final_score = (similarity_score * (1 - rating_weight)) + (normalized_rating * rating_weight)
            
            # Add to the recommendations list
            recommendations.append({
                "course_name": course_name,
                "rating": rating,
                "institution": institution,
                "description": description,
                "course_url": course_url,
                "similarity": similarity_score,
                "final_score": final_score  # This is the adjusted score
            })
        
        # Sort the recommendations based on final score (descending order)
        recommendations = sorted(recommendations, key=lambda x: x['final_score'], reverse=True)
        
        return recommendations

    # Return an empty list if no match is found
    return []

if __name__ == '__main__':
    app.run(debug=True)
