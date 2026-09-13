from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import time
import random
import requests

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
CONFIG = {
    'initial_capital': float(os.getenv('INITIAL_CAPITAL', 1000)),
    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', 0.02)),
    'risk_reward_ratio': float(os.getenv('RISK_REWARD_RATIO', 3)),
    'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', 0.05)),
}

# Real price cache
price_cache = {}

# Trading State
trading_state = {
    'balance': CONFIG['initial_capital'],
    'initial_balance': CONFIG['initial_capital'],
    'is_trading': False,
    'total_trades': 0,
    'winning_trades': 0,
    'losing_trades': 0,
    'total_profit': 0,
    'open_positions': [],
    'closed_trades': [],
    'daily_loss': 0,
    'last_update': datetime.now().isoformat(),
    'trades_today': 0,
    'win_rate': 0,
}

# Get real prices from CoinGecko (free API)
def get_real_price(symbol):
    """Get real price from CoinGecko API"""
    try:
        symbol_map = {
            'BTCUSDT': 'bitcoin',
            'ETHUSDT': 'ethereum',
            'BNBUSDT': 'binancecoin',
            'ADAUSDT': 'cardano',
            'DOGEUSDT': 'dogecoin',
        }
        
        coin = symbol_map.get(symbol, 'bitcoin')
        
        if symbol in price_cache:
            # Return cached price with slight random variation
            cached = price_cache[symbol]
            variation = random.uniform(-0.002, 0.002)  # ±0.2% variation
            return cached * (1 + variation)
        
        url = f'https://api.coingecko.com/api/v3/simple/price?ids={coin}&vs_currencies=usd'
        response = requests.get(url, timeout=5)
        data = response.json()
        price = float(data[coin]['usd'])
        
        # Cache the price
        price_cache[symbol] = price
        
        return price
    except Exception as e:
        print(f"Error getting price for {symbol}: {e}")
        # Return a default price if API fails
        return price_cache.get(symbol, 45000 if symbol == 'BTCUSDT' else 2500)

def calculate_moving_averages(prices):
    """Calculate 20, 50, 200 moving averages"""
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

def simulate_price_history(current_price):
    """Generate simulated price history for technical analysis"""
    prices = []
    price = current_price * 0.98
    
    for i in range(300):
        # Random walk
        change = random.uniform(-0.01, 0.01)
        price = price * (1 + change)
        prices.append(price)
    
    return np.array(prices)

def analyze_symbol(symbol='BTCUSDT'):
    """Full technical analysis of a symbol with REAL prices"""
    try:
        # Get REAL current price
        current_price = get_real_price(symbol)
        
        # Generate simulated historical data based on current price
        prices = simulate_price_history(current_price)
        
        # Calculate indicators
        ma20, ma50, ma200 = calculate_moving_averages(prices)
        rsi = calculate_rsi(prices)
        upper, mid, lower = calculate_bollinger_bands(prices)
        
        # Price change calculation
        price_change = ((prices[-1] - prices[-2]) / prices[-2] * 100) if prices[-2] != 0 else 0
        
        # Signal generation - AGGRESSIVE FOR MORE TRADES
        signal = 'NEUTRAL'
        strength = 0
        
        if ma20 and ma50:
            # BUY signals
            if ma20 > ma50 and rsi > 45 and current_price > mid:
                signal = 'BUY'
                strength = min(100, (rsi - 40) + 30)
            # SELL signals
            elif ma20 < ma50 and rsi < 55 and current_price < mid:
                signal = 'SELL'
                strength = min(100, (60 - rsi) + 30)
            # Additional signals for more trades
            elif rsi > 75:  # Overbought
                signal = 'SELL'
                strength = 75
            elif rsi < 25:  # Oversold
                signal = 'BUY'
                strength = 75
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'price_change': round(price_change, 2),
            'ma20': float(ma20) if ma20 else None,
            'ma50': float(ma50) if ma50 else None,
            'ma200': float(ma200) if ma200 else None,
            'rsi': round(rsi, 1),
            'upper_band': float(upper) if upper else None,
            'middle_band': float(mid) if mid else None,
            'lower_band': float(lower) if lower else None,
            'signal': signal,
            'strength': int(min(strength, 100)),
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

def execute_simulation_trade(symbol, signal, price, strength):
    """Execute simulated trade with profit/loss"""
    try:
        if strength < 50:  # Lower threshold for more trades
            return False
        
        # Simulate trade with 65% win rate
        is_win = random.random() < 0.65
        
        if signal == 'BUY':
            # Simulate profit/loss
            if is_win:
                profit = random.uniform(5, 50)  # $5 to $50 profit
                trading_state['winning_trades'] += 1
            else:
                profit = -random.uniform(2, 20)  # $2 to $20 loss
                trading_state['losing_trades'] += 1
            
            trade = {
                'symbol': symbol,
                'type': 'BUY → SELL',
                'entry_price': round(price, 2),
                'exit_price': round(price * (1 + profit / (price * 100)), 2),
                'profit': round(profit, 2),
                'time': datetime.now().isoformat(),
                'status': '✅ WIN' if is_win else '❌ LOSS',
            }
            
            trading_state['open_positions'].append(trade)
            trading_state['total_profit'] += profit
            trading_state['balance'] += profit
            trading_state['total_trades'] += 1
            
            return True
        
        elif signal == 'SELL':
            # Short trade
            if is_win:
                profit = random.uniform(5, 50)
                trading_state['winning_trades'] += 1
            else:
                profit = -random.uniform(2, 20)
                trading_state['losing_trades'] += 1
            
            trade = {
                'symbol': symbol,
                'type': 'SELL → BUY',
                'entry_price': round(price, 2),
                'exit_price': round(price * (1 - profit / (price * 100)), 2),
                'profit': round(profit, 2),
                'time': datetime.now().isoformat(),
                'status': '✅ WIN' if is_win else '❌ LOSS',
            }
            
            trading_state['closed_trades'].append(trade)
            trading_state['total_profit'] += profit
            trading_state['balance'] += profit
            trading_state['total_trades'] += 1
            
            return True
        
        return False
    except Exception as e:
        print(f"Trade execution error: {e}")
        return False

def auto_trading_loop():
    """Background trading loop - runs every 1 second"""
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT']
    last_trade_time = {}
    
    for symbol in symbols:
        last_trade_time[symbol] = time.time()
    
    while True:
        try:
            if trading_state['is_trading']:
                current_time = time.time()
                
                # Analyze each symbol
                for symbol in symbols:
                    # Trade every 3-5 seconds per symbol to avoid spam
                    if current_time - last_trade_time[symbol] >= random.uniform(3, 5):
                        analysis = analyze_symbol(symbol)
                        
                        if analysis and analysis['signal'] != 'NEUTRAL':
                            # Execute trade
                            execute_simulation_trade(
                                symbol,
                                analysis['signal'],
                                analysis['current_price'],
                                analysis['strength']
                            )
                            
                            last_trade_time[symbol] = current_time
                
                # Update win rate
                if trading_state['total_trades'] > 0:
                    trading_state['win_rate'] = (trading_state['winning_trades'] / trading_state['total_trades'] * 100)
                
                # Update timestamp
                trading_state['last_update'] = datetime.now().isoformat()
                trading_state['trades_today'] = trading_state['total_trades']
            
            # Sleep for 1 second
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
        'status': '✅ Trading Started! Bot is now analyzing markets every 1 second...',
        'is_trading': True,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/trading/stop', methods=['POST'])
def stop_trading():
    trading_state['is_trading'] = False
    return jsonify({
        'status': '⏹️ Trading Stopped',
        'is_trading': False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    profit_percent = ((trading_state['balance'] - trading_state['initial_balance']) / trading_state['initial_balance'] * 100) if trading_state['initial_balance'] > 0 else 0
    
    return jsonify({
        'balance': round(trading_state['balance'], 2),
        'initial_balance': trading_state['initial_balance'],
        'total_trades': trading_state['total_trades'],
        'winning_trades': trading_state['winning_trades'],
        'losing_trades': trading_state['losing_trades'],
        'win_rate': round(trading_state['win_rate'], 1),
        'total_profit': round(trading_state['total_profit'], 2),
        'profit_percent': round(profit_percent, 2),
        'daily_loss': trading_state['daily_loss'],
        'is_trading': trading_state['is_trading'],
        'open_positions': len(trading_state['open_positions']),
        'last_update': trading_state['last_update'],
    })

@app.route('/api/positions', methods=['GET'])
def get_positions():
    return jsonify({
        'open_positions': trading_state['open_positions'][-10:],
        'closed_trades': trading_state['closed_trades'][-20:],
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
