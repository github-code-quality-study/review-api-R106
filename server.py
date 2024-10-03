import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        # This method is a placeholder for future initialization logic
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict, start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """
        # Initialize valid locations
        valid_locations = [
            "Albuquerque, New Mexico",
            "Carlsbad, California",
            "Chula Vista, California",
            "Colorado Springs, Colorado",
            "Denver, Colorado",
            "El Cajon, California",
            "El Paso, Texas",
            "Escondido, California",
            "Fresno, California",
            "La Mesa, California",
            "Las Vegas, Nevada",
            "Los Angeles, California",
            "Oceanside, California",
            "Phoenix, Arizona",
            "Sacramento, California",
            "Salt Lake City, Utah",
            "San Diego, California",
            "Tucson, Arizona"
        ]

        if environ["REQUEST_METHOD"] == "GET":
            query_params = parse_qs(environ.get('QUERY_STRING', ''))
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            # Filter reviews
            filtered_reviews = reviews.copy()

            # Filter by location
            if location:
                if location in valid_locations:
                    filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]
                else:
                    # Return empty list if location is invalid
                    filtered_reviews = []

            # Filter by start_date
            if start_date:
                try:
                    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    filtered_reviews = [
                        review for review in filtered_reviews
                        if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date_dt
                    ]
                except ValueError:
                    pass

            if end_date:
                try:
                    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    filtered_reviews = [
                        review for review in filtered_reviews
                        if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date_dt
                    ]
                except ValueError:
                    # Invalid date format
                    pass

            # Add sentiment analysis to each review if not already present
            for review in filtered_reviews:
                if 'sentiment' not in review:
                    review_body = review['ReviewBody']
                    sentiment = self.analyze_sentiment(review_body)
                    review['sentiment'] = sentiment

            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

        elif environ["REQUEST_METHOD"] == "POST":
            try:
                # Read the POST data
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
                post_params = parse_qs(post_data)
                location = post_params.get('Location', [None])[0]
                review_body = post_params.get('ReviewBody', [None])[0]

                # Validate input
                if not location or not review_body:
                    response_body = json.dumps({'error': 'Location and ReviewBody are required'}).encode('utf-8')
                    start_response('400 Bad Request', [
                        ('Content-Type', 'application/json'),
                        ('Content-Length', str(len(response_body)))
                    ])
                    return [response_body]

                if location not in valid_locations:
                    response_body = json.dumps({'error': 'Invalid Location'}).encode('utf-8')
                    start_response('400 Bad Request', [
                        ('Content-Type', 'application/json'),
                        ('Content-Length', str(len(response_body)))
                    ])
                    return [response_body]

                # Create new review
                new_review = {
                    'ReviewId': str(uuid.uuid4()),
                    'Location': location,
                    'ReviewBody': review_body,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'sentiment': self.analyze_sentiment(review_body)
                }

                # Append the new review to the in-memory list
                reviews.append(new_review)

                response_body = json.dumps(new_review).encode('utf-8')
                start_response('201 Created', [
                    ('Content-Type', 'application/json'),
                    ('Content-Length', str(len(response_body)))
                ])
                return [response_body]

            except Exception as e:
                # Handle unexpected errors
                response_body = json.dumps({'error': str(e)}).encode('utf-8')
                start_response('500 Internal Server Error', [
                    ('Content-Type', 'application/json'),
                    ('Content-Length', str(len(response_body)))
                ])
                return [response_body]

        else:
            # Method not allowed
            response_body = json.dumps({'error': 'Method Not Allowed'}).encode('utf-8')
            start_response('405 Method Not Allowed', [
                ('Content-Type', 'application/json'),
                ('Content-Length', str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
