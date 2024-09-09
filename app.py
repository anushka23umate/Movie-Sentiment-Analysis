from flask import Flask, request, jsonify, render_template
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load your model and tokenizer
model = load_model("E:\\ANU Projects\\imdb_movie_review_sentiment-main\\models\\senti-model.h5")
tokenizer = joblib.load("E:\\ANU Projects\\imdb_movie_review_sentiment-main\\models\\tokenizer.pkl")

def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.4 else "negative"
    return sentiment

@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    review = data['review']
    sentiment = predictive_system(review)
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
