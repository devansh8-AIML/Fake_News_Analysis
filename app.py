from flask import Flask, request, render_template
import joblib
import pandas as pd
import re
import string

app = Flask(__name__)

# Load models
hybrid_model = joblib.load('hybrid_fake_news_model.pkl')
vectorization = joblib.load('tfidf_vectorizer.pkl')

# Text cleaning function
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news_text']
        
        testing_news = {"text": [news]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        
        new_x_test = new_def_test["text"]
        new_xv_test = vectorization.transform(new_x_test)
        prediction = hybrid_model.predict(new_xv_test)
        
        # Updated to pure English
        if prediction[0] == 1:
            result = "Real News ✅"
        else:
            result = "Fake News 🚨"
            
        return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == '__main__':
    app.run(debug=True)