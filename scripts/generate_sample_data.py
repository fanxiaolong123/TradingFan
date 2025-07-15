#!/usr/bin/env python3
"""
生成模拟历史数据用于回测演示
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def generate_price_data(symbol: str, timeframe: str, start_date: datetime, 
                       end_date: datetime, base_price: float, volatility: float):
    """生成模拟价格数据"""
    
    # 计算时间间隔
    timeframe_minutes = {
        '15m': 15,
        '1h': 60,
        '4h': 240,
        '1d': 1440
    }
    
    minutes = timeframe_minutes.get(timeframe, 60)
    
    # 生成时间序列
    timestamps = []
    current = start_date
    while current <= end_date:
        timestamps.append(current)
        current += timedelta(minutes=minutes)
    
    # 生成价格数据
    n = len(timestamps)
    returns = np.random.normal(0.0002, volatility, n)  # 微小的正向漂移
    price_multipliers = np.exp(np.cumsum(returns))
    
    # 生成OHLCV数据
    data = []
    for i, ts in enumerate(timestamps):
        close = base_price * price_multipliers[i]
        
        # 生成开高低
        daily_vol = volatility * 0.5
        open_price = close * (1 + np.random.normal(0, daily_vol))
        high = max(open_price, close) * (1 + abs(np.random.normal(0, daily_vol)))
        low = min(open_price, close) * (1 - abs(np.random.normal(0, daily_vol)))
        
        # 生成成交量
        volume = np.random.lognormal(15, 1.5)  # 对数正态分布的成交量
        
        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    """生成所有币种的模拟数据"""
    
    output_dir = Path("binance_historical_data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 币种配置
    symbols_config = {
        'BTCUSDT': {'base_price': 45000, 'volatility': 0.02},
        'ETHUSDT': {'base_price': 2500, 'volatility': 0.025},
        'BNBUSDT': {'base_price': 300, 'volatility': 0.03},
        'SOLUSDT': {'base_price': 100, 'volatility': 0.04},
        'DOGEUSDT': {'base_price': 0.15, 'volatility': 0.05},
        'PEPEUSDT': {'base_price': 0.000001, 'volatility': 0.06}
    }
    
    timeframes = ['15m', '1h', '4h', '1d']
    
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 7, 15)
    
    print("🚀 生成模拟历史数据")
    print("=" * 60)
    
    for symbol, config in symbols_config.items():
        for timeframe in timeframes:
            print(f"📊 生成 {symbol} {timeframe} 数据...")
            
            # 生成数据
            df = generate_price_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                base_price=config['base_price'],
                volatility=config['volatility']
            )
            
            # 保存数据
            filename = f"{symbol}_{timeframe}_combined.csv"
            filepath = output_dir / filename
            df.to_csv(filepath, index=False)
            
            print(f"   ✅ 已保存: {filepath} ({len(df)} 条记录)")
    
    print("\n✅ 数据生成完成！")

if __name__ == "__main__":
    main()