from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your model and vectorizer at the start of your app
model = joblib.load('/home/sunbeam/Desktop/Sentiment_analysis/Sentiment_Analysis/model/model.pkl')  # Adjust the path as necessary
vectorizer = joblib.load('/home/sunbeam/Desktop/Sentiment_analysis/Sentiment_Analysis/model/vectorizer.pkl')  # Adjust the path as necessary


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # Transform the input message
    message_vectorized = vectorizer.transform([message])
    print("Vectorized message shape:", message_vectorized.shape)  # Debugging line

    # Predict sentiment
    prediction = model.predict(message_vectorized)[0]  # Get the predicted class
    print("Prediction:", prediction)  # Debugging line

    return render_template('index.html', prediction=prediction, message=message)


if __name__ == '__main__':
    app.run(debug=True)