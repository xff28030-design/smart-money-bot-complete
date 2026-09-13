from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from binance.client import Client
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY'),
    'api_secret': os.getenv('BINANCE_SECRET_KEY'),
    'initial_capital': float(os.getenv('INITIAL_CAPITAL', 1000)),
    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 0.02)),
    'risk_reward_ratio': float(os.getenv('RISK_REWARD_RATIO', 3)),
    'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', 0.05)),
}

# Initialize Binance Client
try:
    client = Client(CONFIG['api_key'], CONFIG['api_secret'])
    client.API_URL = 'https://testnet.binance.vision/api'
except:
    client = None

# Trading State
trading_state = {
    'balance': CONFIG['initial_capital'],
    'is_trading': False,
    'total_trades': 0,
    'winning_trades': 0,
    'total_profit': 0,
    'open_positions': [],
    'closed_trades': [],
    'daily_loss': 0,
}

# Smart Money Indicators
def calculate_moving_averages(prices):
    """Calculate 20, 50, 200 day moving averages"""
    if len(prices) < 200:
        return None, None, None
    
    ma20 = np.mean(prices[-20:])
    ma50 = np.mean(prices[-50:])
    ma200 = np.mean(prices[-200:])
    
    return ma20, ma50, ma200

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    if len(prices) < period:
        return 50
    
    deltas = np.diff(prices[-period-1:])
    seed = deltas[:period]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    
    rs = up / down if down != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(prices, period=20):
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    ma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    
    upper = ma + (std * 2)
    lower = ma - (std * 2)
    
    return upper, ma, lower

def detect_order_block(prices, volumes):
    """Detect Smart Money Order Blocks"""
    if len(prices) < 5:
        return False
    
    # Check for large volume candle followed by smaller candles
    recent_volume = volumes[-1]
    avg_volume = np.mean(volumes[-20:-1])
    
    if recent_volume > avg_volume * 1.5:
        price_change = abs(prices[-1] - prices[-2]) / prices[-2]
        if price_change > 0.005:  # 0.5% change
            return True
    
    return False

def analyze_symbol(symbol='BTCUSDT'):
    """Full technical analysis of a symbol"""
    try:
        if not client:
            return None
        
        # Get historical data
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, "100 hours ago UTC")
        
        prices = np.array([float(kline[4]) for kline in klines])  # Close prices
        volumes = np.array([float(kline[7]) for kline in klines])  # Volumes
        
        # Calculate indicators
        ma20, ma50, ma200 = calculate_moving_averages(prices)
        rsi = calculate_rsi(prices)
        upper, mid, lower = calculate_bollinger_bands(prices)
        order_block = detect_order_block(prices, volumes)
        
        current_price = prices[-1]
        
        # Signal generation
        signal = 'NEUTRAL'
        if ma20 and ma50 and ma200:
            if ma20 > ma50 > ma200 and rsi > 50 and current_price > mid:
                signal = 'BUY'
            elif ma20 < ma50 < ma200 and rsi < 50 and current_price < mid:
                signal = 'SELL'
        
        return {
            'symbol': symbol,
            'current_price': float(current_price),
            'ma20': float(ma20) if ma20 else None,
            'ma50': float(ma50) if ma50 else None,
            'ma200': float(ma200) if ma200 else None,
            'rsi': float(rsi),
            'upper_band': float(upper) if upper else None,
            'middle_band': float(mid) if mid else None,
            'lower_band': float(lower) if lower else None,
            'order_block': order_block,
            'signal': signal,
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    trading_state['is_trading'] = True
    return jsonify({'status': 'Trading started', 'is_trading': True})

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    trading_state['is_trading'] = False
    return jsonify({'status': 'Trading stopped', 'is_trading': False})

@app.route('/api/stats', methods=['GET'])
def get_stats():
    win_rate = (trading_state['winning_trades'] / max(trading_state['total_trades'], 1)) * 100
    
    return jsonify({
        'balance': trading_state['balance'],
        'total_trades': trading_state['total_trades'],
        'winning_trades': trading_state['winning_trades'],
        'win_rate': win_rate,
        'total_profit': trading_state['total_profit'],
        'daily_loss': trading_state['daily_loss'],
        'is_trading': trading_state['is_trading'],
        'open_positions': len(trading_state['open_positions']),
    })

@app.route('/api/positions', methods=['GET'])
def get_positions():
    return jsonify({
        'open_positions': trading_state['open_positions'][:5],
        'closed_trades': trading_state['closed_trades'][:10],
    })

@app.route('/api/analyze/<symbol>', methods=['GET'])
def analyze(symbol):
    result = analyze_symbol(symbol)
    return jsonify(result or {'error': 'Failed to analyze'})

# To-Do List API
@app.route('/api/todos', methods=['GET'])
def get_todos():
    todos_file = 'todos.json'
    if os.path.exists(todos_file):
        with open(todos_file, 'r') as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/api/todos', methods=['POST'])
def add_todo():
    todos_file = 'todos.json'
    data = request.json
    
    todos = []
    if os.path.exists(todos_file):
        with open(todos_file, 'r') as f:
            todos = json.load(f)
    
    new_todo = {
        'id': len(todos) + 1,
        'title': data.get('title'),
        'completed': False,
        'created_at': datetime.now().isoformat()
    }
    
    todos.append(new_todo)
    
    with open(todos_file, 'w') as f:
        json.dump(todos, f)
    
    return jsonify(new_todo), 201

@app.route('/api/todos/<int:todo_id>', methods=['PUT'])
def update_todo(todo_id):
    todos_file = 'todos.json'
    data = request.json
    
    todos = []
    if os.path.exists(todos_file):
        with open(todos_file, 'r') as f:
            todos = json.load(f)
    
    for todo in todos:
        if todo['id'] == todo_id:
            todo['completed'] = data.get('completed', todo['completed'])
            todo['title'] = data.get('title', todo['title'])
    
    with open(todos_file, 'w') as f:
        json.dump(todos, f)
    
    return jsonify({'status': 'Updated'})

@app.route('/api/todos/<int:todo_id>', methods=['DELETE'])
def delete_todo(todo_id):
    todos_file = 'todos.json'
    
    todos = []
    if os.path.exists(todos_file):
        with open(todos_file, 'r') as f:
            todos = json.load(f)
    
    todos = [t for t in todos if t['id'] != todo_id]
    
    with open(todos_file, 'w') as f:
        json.dump(todos, f)
    
    return jsonify({'status': 'Deleted'})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
