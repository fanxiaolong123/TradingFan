#!/usr/bin/env python3
"""
下载Binance历史数据
"""

import os
import sys
import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BinanceDataDownloader:
    """Binance历史数据下载器"""
    
    def __init__(self, output_dir: str = "binance_historical_data/processed"):
        """初始化下载器"""
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_ohlcv_data(self, symbol: str, timeframe: str, 
                           start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """下载OHLCV数据"""
        logger.info(f"📥 下载 {symbol} {timeframe} 数据...")
        
        all_data = []
        
        # 将时间转换为毫秒时间戳
        since = int(start_date.timestamp() * 1000)
        end_timestamp = int(end_date.timestamp() * 1000)
        
        # 计算每次请求的时间间隔
        timeframe_minutes = {
            '15m': 15,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        limit = 1000  # 每次请求的最大数据条数
        
        while since < end_timestamp:
            try:
                # 获取数据
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # 更新时间戳
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + (minutes * 60 * 1000)
                
                # 显示进度
                progress_date = datetime.fromtimestamp(last_timestamp / 1000)
                logger.info(f"   进度: {progress_date.strftime('%Y-%m-%d %H:%M')}")
                
                # 限速
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ 下载失败: {e}")
                time.sleep(5)
                continue
        
        # 转换为DataFrame
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 去重并排序
            df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            
            logger.info(f"✅ 下载完成: {len(df)} 条记录")
            return df
        else:
            logger.warning(f"⚠️ 没有获取到数据")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """保存数据到文件"""
        if df.empty:
            return
        
        # 将斜杠替换为下划线
        clean_symbol = symbol.replace('/', '')
        filename = f"{clean_symbol}_{timeframe}_combined.csv"
        filepath = self.output_dir / filename
        
        df.to_csv(filepath, index=False)
        logger.info(f"💾 数据已保存: {filepath}")
    
    def download_all_data(self, symbols: list, timeframes: list, 
                         start_date: datetime, end_date: datetime):
        """下载所有币种和时间框架的数据"""
        total = len(symbols) * len(timeframes)
        count = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                count += 1
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 处理 ({count}/{total}): {symbol} {timeframe}")
                logger.info(f"{'='*60}")
                
                try:
                    # 下载数据
                    df = self.download_ohlcv_data(symbol, timeframe, start_date, end_date)
                    
                    # 保存数据
                    if not df.empty:
                        self.save_data(df, symbol, timeframe)
                    
                except Exception as e:
                    logger.error(f"❌ 处理失败 {symbol} {timeframe}: {e}")
                    continue
                
                # 休息一下，避免触发限速
                time.sleep(2)
        
        logger.info(f"\n✅ 全部下载完成！")

def main():
    """主函数"""
    print("🚀 Binance历史数据下载器")
    print("=" * 80)
    
    # 创建下载器
    downloader = BinanceDataDownloader()
    
    # 定义要下载的币种和时间框架
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'DOGE/USDT', 'PEPE/USDT']
    timeframes = ['15m', '1h', '4h', '1d']
    
    # 定义时间范围
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 7, 15)
    
    logger.info(f"📅 下载时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"🪙 币种: {', '.join(symbols)}")
    logger.info(f"⏱️ 时间框架: {', '.join(timeframes)}")
    
    # 开始下载
    downloader.download_all_data(symbols, timeframes, start_date, end_date)

if __name__ == "__main__":
    main()