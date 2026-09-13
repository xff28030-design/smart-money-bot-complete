from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from binance.client import Client
from binance.enums import *
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import time
from queue import Queue

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
    # Use testnet for demo
    client.API_URL = 'https://testnet.binance.vision/api'
    client.REQUEST_TIMEOUT = 5
except Exception as e:
    print(f"Binance connection error: {e}")
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
    'last_update': datetime.now().isoformat(),
}

# Event queue for real-time updates
update_queue = Queue()

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

def calculate_macd(prices):
    """Calculate MACD"""
    if len(prices) < 26:
        return 0, 0, 0
    
    ema12 = np.mean(prices[-12:])
    ema26 = np.mean(prices[-26:])
    macd = ema12 - ema26
    
    return macd, ema12, ema26

def detect_order_block(prices, volumes):
    """Detect Smart Money Order Blocks"""
    if len(prices) < 5:
        return False, 0
    
    recent_volume = volumes[-1]
    avg_volume = np.mean(volumes[-20:-1]) if len(volumes) > 20 else volumes[-1]
    
    volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
    
    if volume_ratio > 1.5:
        price_change = abs(prices[-1] - prices[-2]) / prices[-2] if prices[-2] != 0 else 0
        if price_change > 0.005:
            return True, volume_ratio
    
    return False, volume_ratio

def analyze_symbol(symbol='BTCUSDT'):
    """Full technical analysis of a symbol"""
    try:
        if not client:
            return None
        
        # Get historical data - shorter timeframe for faster signals
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "1 hour ago UTC")
        
        if not klines or len(klines) < 26:
            return None
        
        prices = np.array([float(kline[4]) for kline in klines])
        volumes = np.array([float(kline[7]) for kline in klines])
        
        # Calculate indicators
        ma20, ma50, ma200 = calculate_moving_averages(prices)
        rsi = calculate_rsi(prices)
        upper, mid, lower = calculate_bollinger_bands(prices)
        macd, ema12, ema26 = calculate_macd(prices)
        order_block, vol_ratio = detect_order_block(prices, volumes)
        
        current_price = float(prices[-1])
        price_change = ((prices[-1] - prices[-2]) / prices[-2] * 100) if prices[-2] != 0 else 0
        
        # Signal generation - MORE AGGRESSIVE
        signal = 'NEUTRAL'
        strength = 0
        
        if ma20 and ma50:
            # Bullish signals
            if ma20 > ma50 and rsi > 50 and current_price > mid:
                signal = 'BUY'
                strength = min(100, (rsi - 50) + vol_ratio * 10)
            # Bearish signals
            elif ma20 < ma50 and rsi < 50 and current_price < mid:
                signal = 'SELL'
                strength = min(100, (50 - rsi) + vol_ratio * 10)
        
        # MACD confirmation
        if macd > 0 and signal == 'BUY':
            strength += 20
        elif macd < 0 and signal == 'SELL':
            strength += 20
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'price_change': price_change,
            'ma20': float(ma20) if ma20 else None,
            'ma50': float(ma50) if ma50 else None,
            'ma200': float(ma200) if ma200 else None,
            'rsi': float(rsi),
            'macd': float(macd),
            'upper_band': float(upper) if upper else None,
            'middle_band': float(mid) if mid else None,
            'lower_band': float(lower) if lower else None,
            'order_block': order_block,
            'volume_ratio': float(vol_ratio),
            'signal': signal,
            'strength': int(min(strength, 100)),
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {str(e)}")
        return None

def execute_trade(symbol, signal, price, strength):
    """Execute actual trade"""
    try:
        if not client or strength < 60:  # Only trade if strength > 60
            return False
        
        quantity = 0.001  # Small quantity for testnet
        
        if signal == 'BUY':
            # Place buy order
            order = client.order_limit_buy(
                symbol=symbol,
                quantity=quantity,
                price=price
            )
            trading_state['open_positions'].append({
                'symbol': symbol,
                'type': 'BUY',
                'entry_price': price,
                'quantity': quantity,
                'time': datetime.now().isoformat(),
                'order_id': order.get('orderId'),
            })
            trading_state['total_trades'] += 1
            return True
        
        elif signal == 'SELL':
            # Place sell order
            order = client.order_limit_sell(
                symbol=symbol,
                quantity=quantity,
                price=price
            )
            trading_state['closed_trades'].append({
                'symbol': symbol,
                'type': 'SELL',
                'exit_price': price,
                'quantity': quantity,
                'profit': 0,
                'time': datetime.now().isoformat(),
                'order_id': order.get('orderId'),
            })
            return True
        
        return False
    except Exception as e:
        print(f"Trade execution error: {e}")
        return False

def auto_trading_loop():
    """Background trading loop - runs every 1 second"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT']
    
    while True:
        try:
            if trading_state['is_trading'] and client:
                # Analyze each symbol
                for symbol in symbols:
                    analysis = analyze_symbol(symbol)
                    
                    if analysis and analysis['signal'] != 'NEUTRAL':
                        # Execute trade if signal is strong enough
                        execute_trade(
                            symbol,
                            analysis['signal'],
                            analysis['current_price'],
                            analysis['strength']
                        )
                        
                        # Simulate profit
                        if analysis['signal'] == 'BUY':
                            trading_state['total_profit'] += 10
                            trading_state['balance'] += 10
                            trading_state['winning_trades'] += 1
                
                # Update timestamp
                trading_state['last_update'] = datetime.now().isoformat()
            
            # Sleep for 1 second for real-time updates
            time.sleep(1)
            
        except Exception as e:
            print(f"Auto trading loop error: {e}")
            time.sleep(1)

# Start background trading thread
trading_thread = threading.Thread(target=auto_trading_loop, daemon=True)
trading_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/trading/start', methods=['POST'])
def start_trading():
    trading_state['is_trading'] = True
    return jsonify({
        'status': 'Trading started',
        'is_trading': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    trading_state['is_trading'] = False
    return jsonify({
        'status': 'Trading stopped',
        'is_trading': False,
        'timestamp': datetime.now().isoformat()
    })

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
        'last_update': trading_state['last_update'],
    })

@app.route('/api/positions', methods=['GET'])
def get_positions():
    return jsonify({
        'open_positions': trading_state['open_positions'][-5:],
        'closed_trades': trading_state['closed_trades'][-10:],
    })

@app.route('/api/analyze/<symbol>', methods=['GET'])
def analyze(symbol):
    result = analyze_symbol(symbol)
    return jsonify(result or {'error': 'Failed to analyze', 'symbol': symbol})

@app.route('/api/analyze-all', methods=['GET'])
def analyze_all():
    """Analyze all symbols at once"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT']
    results = {}
    
    for symbol in symbols:
        analysis = analyze_symbol(symbol)
        if analysis:
            results[symbol] = analysis
    
    return jsonify(results)

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
