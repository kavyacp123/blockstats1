from flask import Flask, jsonify, render_template, request, redirect, url_for
import requests
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    crypto = request.form['crypto']
    # Generate the graph
    return redirect(url_for('index', crypto=crypto))

@app.route('/index/<crypto>')
def index(crypto):
    graph_filename = f'{crypto}_price_prediction_with_future.jpg'
    return render_template('index.html', crypto=crypto, graph_filename=graph_filename)

@app.route('/current_price/<crypto>')
def current_price(crypto):
    try:
        symbol_map = {
            'bitcoin': 'BTCUSDT',
            'ethereum': 'ETHUSDT',
            'litecoin': 'LTCUSDT',
            'polygon': 'MATICUSDT',
            'tether': 'USDTUSDT',
            'stellar': 'XLMUSDT',
            'binancecoin': 'BNBUSDT',
            'chainlink': 'LINKUSDT'
        }
        symbol = symbol_map.get(crypto.lower(), 'BTCUSDT')
        url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
        response = requests.get(url)
        data = response.json()
        price = data.get('price')
        if price:
            return jsonify(price=float(price))
        else:
            return jsonify(error="Failed to fetch price"), 500
    except requests.exceptions.SSLError:
        return jsonify(error="SSL certificate verification failed"), 500


if __name__ == '__main__':
    app.run(debug=True)