from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask app and VADER sentiment analyzer
app = Flask(__name__)
analyzer = SentimentIntensityAnalyzer()


# Home route to display the input form
@app.route('/')
def home():
    return render_template('index.html')


# Route to handle sentiment analysis when form is submitted
@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        # Get user input from the form
        user_input = request.form['user_input']

        # Perform sentiment analysis using VADER
        scores = analyzer.polarity_scores(user_input)
        compound_score = scores['compound']

        # Classify sentiment based on the compound score
        if compound_score >= 0.05:
            sentiment = 'Positive'
        elif compound_score <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'

        # Return the result to the webpage
        return render_template('index.html', user_input=user_input, sentiment=sentiment)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
