#!/usr/bin/env python3
"""
Binance历史数据下载器

从 https://data.binance.vision/ 批量下载历史K线数据
支持多币种、多时间周期、多月份的数据获取
"""

import os
import sys
import requests
import zipfile
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import logging
from pathlib import Path
import hashlib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binance_download.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceHistoricalDownloader:
    """Binance历史数据下载器"""
    
    def __init__(self, data_dir: str = "binance_historical_data"):
        """
        初始化下载器
        
        Args:
            data_dir: 数据保存目录
        """
        self.base_url = "https://data.binance.vision/data/spot/monthly/klines"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.downloads_dir = self.data_dir / "downloads"
        self.processed_dir = self.data_dir / "processed"
        self.downloads_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # 请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Binance Historical Data Downloader)'
        })
        
        # 数据列名
        self.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades_count', 
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
        ]
    
    def generate_download_urls(self, symbol: str, interval: str, 
                             start_year: int, end_year: int) -> List[Dict]:
        """
        生成下载URL列表
        
        Args:
            symbol: 交易对符号 (如 BTCUSDT)
            interval: 时间间隔 (如 1h, 1d)
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            List[Dict]: 包含URL和文件信息的列表
        """
        urls = []
        
        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # 格式化月份
                month_str = f"{year}-{month:02d}"
                
                # 构建URL
                filename = f"{symbol}-{interval}-{month_str}.zip"
                url = f"{self.base_url}/{symbol}/{interval}/{filename}"
                
                # 本地文件路径
                local_path = self.downloads_dir / filename
                
                urls.append({
                    'url': url,
                    'filename': filename,
                    'local_path': local_path,
                    'symbol': symbol,
                    'interval': interval,
                    'year': year,
                    'month': month
                })
        
        return urls
    
    def download_file(self, url: str, local_path: Path, 
                     timeout: int = 30) -> bool:
        """
        下载单个文件
        
        Args:
            url: 下载URL
            local_path: 本地保存路径
            timeout: 超时时间
            
        Returns:
            bool: 下载是否成功
        """
        try:
            # 检查文件是否已存在
            if local_path.exists():
                logger.info(f"文件已存在，跳过: {local_path.name}")
                return True
            
            logger.info(f"开始下载: {url}")
            
            # 发送请求
            response = self.session.get(url, timeout=timeout, stream=True)
            
            if response.status_code == 200:
                # 保存文件
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"✅ 下载成功: {local_path.name} ({local_path.stat().st_size / 1024:.2f} KB)")
                return True
                
            elif response.status_code == 404:
                logger.warning(f"⚠️ 文件不存在: {url}")
                return False
                
            else:
                logger.error(f"❌ 下载失败: {url} (状态码: {response.status_code})")
                return False
                
        except Exception as e:
            logger.error(f"❌ 下载异常: {url} - {str(e)}")
            return False
    
    def extract_and_process_zip(self, zip_path: Path) -> Optional[pd.DataFrame]:
        """
        解压并处理ZIP文件
        
        Args:
            zip_path: ZIP文件路径
            
        Returns:
            Optional[pd.DataFrame]: 处理后的数据
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 获取CSV文件名
                csv_files = [f for f in zip_ref.namelist() if f.endswith('.csv')]
                
                if not csv_files:
                    logger.warning(f"ZIP文件中没有找到CSV文件: {zip_path}")
                    return None
                
                # 读取第一个CSV文件
                csv_file = csv_files[0]
                
                with zip_ref.open(csv_file) as f:
                    # 读取数据
                    df = pd.read_csv(f, names=self.columns)
                    
                    # 转换时间戳
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
                    
                    # 重命名列以匹配我们的格式
                    df = df.rename(columns={
                        'open_time': 'timestamp',
                        'quote_volume': 'quote_volume',
                        'trades_count': 'trades_count'
                    })
                    
                    # 选择需要的列
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trades_count']]
                    
                    # 添加symbol列
                    symbol = zip_path.stem.split('-')[0]
                    df['symbol'] = symbol
                    
                    logger.info(f"✅ 处理完成: {zip_path.name} ({len(df)} 条记录)")
                    return df
                    
        except Exception as e:
            logger.error(f"❌ 处理ZIP文件失败: {zip_path} - {str(e)}")
            return None
    
    def download_symbol_data(self, symbol: str, interval: str, 
                           start_year: int = 2020, end_year: int = 2024,
                           delay: float = 0.1) -> bool:
        """
        下载指定交易对的数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_year: 开始年份
            end_year: 结束年份
            delay: 请求间隔
            
        Returns:
            bool: 下载是否成功
        """
        logger.info(f"🚀 开始下载 {symbol} {interval} 数据 ({start_year}-{end_year})")
        
        # 生成下载URL
        urls = self.generate_download_urls(symbol, interval, start_year, end_year)
        
        # 下载统计
        downloaded_files = []
        failed_files = []
        
        for url_info in urls:
            # 下载文件
            success = self.download_file(url_info['url'], url_info['local_path'])
            
            if success:
                downloaded_files.append(url_info)
            else:
                failed_files.append(url_info)
            
            # 请求间隔
            time.sleep(delay)
        
        logger.info(f"📊 下载完成: {len(downloaded_files)} 成功, {len(failed_files)} 失败")
        
        # 处理下载的文件
        if downloaded_files:
            self.process_downloaded_files(downloaded_files, symbol, interval)
        
        return len(downloaded_files) > 0
    
    def process_downloaded_files(self, downloaded_files: List[Dict], 
                               symbol: str, interval: str) -> None:
        """
        处理下载的文件，合并为完整数据集
        
        Args:
            downloaded_files: 下载成功的文件列表
            symbol: 交易对符号
            interval: 时间间隔
        """
        logger.info(f"📦 开始处理 {symbol} {interval} 数据...")
        
        all_data = []
        
        for file_info in downloaded_files:
            if file_info['local_path'].exists():
                df = self.extract_and_process_zip(file_info['local_path'])
                if df is not None:
                    all_data.append(df)
        
        if all_data:
            # 合并所有数据
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # 按时间排序
            combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
            
            # 去重
            combined_df = combined_df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
            
            # 保存处理后的数据
            output_file = self.processed_dir / f"{symbol}_{interval}_combined.csv"
            combined_df.to_csv(output_file, index=False)
            
            logger.info(f"✅ 合并完成: {output_file}")
            logger.info(f"📊 数据统计: {len(combined_df)} 条记录")
            logger.info(f"📅 时间范围: {combined_df['timestamp'].min()} 到 {combined_df['timestamp'].max()}")
            
            # 显示数据质量信息
            self.show_data_quality(combined_df, symbol, interval)
        
        else:
            logger.warning(f"⚠️ 没有可处理的数据: {symbol} {interval}")
    
    def show_data_quality(self, df: pd.DataFrame, symbol: str, interval: str) -> None:
        """
        显示数据质量信息
        
        Args:
            df: 数据框
            symbol: 交易对符号
            interval: 时间间隔
        """
        logger.info(f"📋 {symbol} {interval} 数据质量报告:")
        logger.info(f"   记录数: {len(df):,}")
        logger.info(f"   时间跨度: {(df['timestamp'].max() - df['timestamp'].min()).days} 天")
        logger.info(f"   数据完整性: {(1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.2f}%")
        logger.info(f"   平均日交易量: {df['volume'].mean():.2f}")
        logger.info(f"   价格范围: {df['low'].min():.2f} - {df['high'].max():.2f}")
    
    def download_multiple_symbols(self, symbols: List[str], intervals: List[str],
                                start_year: int = 2020, end_year: int = 2024) -> Dict:
        """
        批量下载多个交易对的数据
        
        Args:
            symbols: 交易对列表
            intervals: 时间间隔列表
            start_year: 开始年份
            end_year: 结束年份
            
        Returns:
            Dict: 下载结果统计
        """
        results = {
            'successful': [],
            'failed': []
        }
        
        total_tasks = len(symbols) * len(intervals)
        current_task = 0
        
        for symbol in symbols:
            for interval in intervals:
                current_task += 1
                logger.info(f"🎯 进度: {current_task}/{total_tasks} - {symbol} {interval}")
                
                try:
                    success = self.download_symbol_data(symbol, interval, start_year, end_year)
                    
                    if success:
                        results['successful'].append(f"{symbol}_{interval}")
                    else:
                        results['failed'].append(f"{symbol}_{interval}")
                        
                except Exception as e:
                    logger.error(f"❌ 下载失败: {symbol} {interval} - {str(e)}")
                    results['failed'].append(f"{symbol}_{interval}")
        
        # 显示总结
        logger.info(f"\n📊 批量下载完成:")
        logger.info(f"✅ 成功: {len(results['successful'])} 个")
        logger.info(f"❌ 失败: {len(results['failed'])} 个")
        
        if results['successful']:
            logger.info(f"✅ 成功项目: {', '.join(results['successful'])}")
        
        if results['failed']:
            logger.info(f"❌ 失败项目: {', '.join(results['failed'])}")
        
        return results

def main():
    """主函数"""
    print("🚀 Binance历史数据下载器")
    print("=" * 60)
    
    # 创建下载器
    downloader = BinanceHistoricalDownloader()
    
    # 配置下载参数
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT']
    intervals = ['1h', '1d']
    start_year = 2020
    end_year = 2024  # 到2024年
    
    print(f"📊 下载配置:")
    print(f"   交易对: {', '.join(symbols)}")
    print(f"   时间周期: {', '.join(intervals)}")
    print(f"   时间范围: {start_year}-{end_year}")
    print(f"   保存目录: {downloader.data_dir}")
    print()
    
    # 开始批量下载
    results = downloader.download_multiple_symbols(symbols, intervals, start_year, end_year)
    
    print(f"\n🎉 下载完成!")
    print(f"📁 数据保存在: {downloader.processed_dir}")
    
    return results

if __name__ == "__main__":
    results = main()