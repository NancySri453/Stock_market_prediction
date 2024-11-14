from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import pickle
import numpy as np

# Load the trained model
with open('stock_predictor_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Create Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to predict stock prices
@app.route('/predict', methods=['POST'])
def predict():
    # Get the stock symbol from the user
    stock_symbol = request.form['symbol']
    
    # Fetch stock data from Yahoo Finance
    stock_data = yf.download(stock_symbol, period="5y", interval="1d")
    
    # Calculate technical indicators (SMA50 and SMA200)
    stock_data['SMA50'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['SMA200'] = stock_data['Close'].rolling(window=200).mean()
    
    # Use the last row of the data to make predictions
    last_data = stock_data[['SMA50', 'SMA200']].dropna().iloc[-1:]
    
    # Make a prediction
    prediction = model.predict(last_data)
    
    return render_template('result.html', prediction=prediction[0], symbol=stock_symbol)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)