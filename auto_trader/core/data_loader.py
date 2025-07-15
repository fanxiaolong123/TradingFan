"""
数据加载器模块 - 负责历史K线数据的本地缓存管理

这个模块提供了完整的历史数据管理功能，包括：
- 本地CSV文件缓存管理
- 增量数据更新
- 多币种批量下载
- 数据完整性验证
- 与现有BinanceDataProvider的集成
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import logging
import time
from pathlib import Path
import json

from .data import BinanceDataProvider


class DataQualityChecker:
    """
    数据质量检查器类
    
    负责检查K线数据的质量，包括：
    - 数据完整性检查（缺失值、重复值）
    - 数据逻辑性检查（价格关系、时间序列）
    - 数据连续性检查（时间间隔、数据缺口）
    - 数据异常检查（异常价格、异常成交量）
    """
    
    def __init__(self):
        """初始化数据质量检查器"""
        self.logger = logging.getLogger(__name__)
    
    def check_data_quality(self, df: pd.DataFrame, symbol: str, interval: str, 
                          expected_interval: Optional[str] = None) -> Dict[str, Any]:
        """
        检查K线数据质量
        
        Args:
            df: 要检查的数据DataFrame
            symbol: 交易对符号
            interval: 时间间隔
            expected_interval: 期望的时间间隔（用于验证）
            
        Returns:
            Dict[str, Any]: 质量检查结果
        """
        quality_report = {
            'symbol': symbol,
            'interval': interval,
            'total_records': len(df),
            'check_time': datetime.now(),
            'issues': [],
            'warnings': [],
            'score': 100.0,
            'passed': True
        }
        
        if df.empty:
            quality_report['issues'].append("数据为空")
            quality_report['score'] = 0.0
            quality_report['passed'] = False
            return quality_report
        
        # 检查必需列
        self._check_required_columns(df, quality_report)
        
        # 检查数据完整性
        self._check_data_completeness(df, quality_report)
        
        # 检查数据逻辑性
        self._check_data_logic(df, quality_report)
        
        # 检查时间序列
        self._check_time_series(df, interval, quality_report)
        
        # 检查数据连续性
        self._check_data_continuity(df, interval, quality_report)
        
        # 检查数据异常
        self._check_data_anomalies(df, quality_report)
        
        # 计算最终评分
        self._calculate_final_score(quality_report)
        
        return quality_report
    
    def _check_required_columns(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """检查必需列是否存在"""
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            report['issues'].append(f"缺少必需列: {missing_columns}")
            report['score'] -= 20
    
    def _check_data_completeness(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """检查数据完整性"""
        # 检查空值
        null_counts = df.isnull().sum()
        if null_counts.any():
            null_info = null_counts[null_counts > 0].to_dict()
            report['issues'].append(f"发现空值: {null_info}")
            report['score'] -= min(15, len(null_info) * 3)
        
        # 检查重复时间戳
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                report['issues'].append(f"发现重复时间戳: {duplicate_timestamps} 个")
                report['score'] -= min(10, duplicate_timestamps)
        
        # 检查零值
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    if col == 'volume':
                        report['warnings'].append(f"{col} 有 {zero_count} 个零值")
                    else:
                        report['issues'].append(f"{col} 有 {zero_count} 个零值")
                        report['score'] -= min(5, zero_count)
    
    def _check_data_logic(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """检查数据逻辑性"""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return
        
        # 检查价格关系逻辑
        # high >= max(open, close) 且 high >= low
        high_issues = ((df['high'] < df['open']) | 
                      (df['high'] < df['close']) | 
                      (df['high'] < df['low'])).sum()
        
        if high_issues > 0:
            report['issues'].append(f"最高价逻辑错误: {high_issues} 个")
            report['score'] -= min(15, high_issues)
        
        # low <= min(open, close) 且 low <= high
        low_issues = ((df['low'] > df['open']) | 
                     (df['low'] > df['close']) | 
                     (df['low'] > df['high'])).sum()
        
        if low_issues > 0:
            report['issues'].append(f"最低价逻辑错误: {low_issues} 个")
            report['score'] -= min(15, low_issues)
        
        # 检查成交量为负数
        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                report['issues'].append(f"成交量为负数: {negative_volume} 个")
                report['score'] -= min(10, negative_volume)
    
    def _check_time_series(self, df: pd.DataFrame, interval: str, report: Dict[str, Any]) -> None:
        """检查时间序列"""
        if 'timestamp' not in df.columns or len(df) < 2:
            return
        
        # 确保时间戳已转换为datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            try:
                timestamps = pd.to_datetime(df['timestamp'])
            except:
                report['issues'].append("时间戳格式无效")
                report['score'] -= 10
                return
        else:
            timestamps = df['timestamp']
        
        # 检查时间顺序
        if not timestamps.is_monotonic_increasing:
            report['issues'].append("时间序列不是单调递增的")
            report['score'] -= 10
        
        # 检查时间间隔
        expected_delta = self._get_expected_timedelta(interval)
        if expected_delta:
            time_diffs = timestamps.diff().dropna()
            
            # 允许的时间差范围（考虑市场休市等因素）
            min_delta = expected_delta * 0.9
            max_delta = expected_delta * 5  # 允许最多5倍的间隔（处理休市）
            
            irregular_intervals = ((time_diffs < min_delta) | (time_diffs > max_delta)).sum()
            if irregular_intervals > 0:
                report['warnings'].append(f"时间间隔不规律: {irregular_intervals} 个")
                if irregular_intervals > len(df) * 0.1:  # 超过10%认为是问题
                    report['score'] -= 5
    
    def _check_data_continuity(self, df: pd.DataFrame, interval: str, report: Dict[str, Any]) -> None:
        """检查数据连续性"""
        if 'timestamp' not in df.columns or len(df) < 2:
            return
        
        expected_delta = self._get_expected_timedelta(interval)
        if not expected_delta:
            return
        
        # 检查数据缺口
        timestamps = pd.to_datetime(df['timestamp'])
        time_diffs = timestamps.diff().dropna()
        
        # 寻找明显的数据缺口（时间间隔超过期望值的3倍）
        gap_threshold = expected_delta * 3
        gaps = time_diffs[time_diffs > gap_threshold]
        
        if len(gaps) > 0:
            total_gap_time = gaps.sum()
            report['warnings'].append(f"发现 {len(gaps)} 个数据缺口，总计 {total_gap_time}")
            
            # 如果缺口过多，扣分
            if len(gaps) > 5:
                report['score'] -= min(10, len(gaps) - 5)
    
    def _check_data_anomalies(self, df: pd.DataFrame, report: Dict[str, Any]) -> None:
        """检查数据异常"""
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
            
            values = df[col].dropna()
            if len(values) == 0:
                continue
            
            # 使用IQR方法检测异常值
            Q1 = values.quantile(0.25)
            Q3 = values.quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((values < lower_bound) | (values > upper_bound)).sum()
                
                if outliers > 0:
                    outlier_rate = outliers / len(values)
                    if outlier_rate > 0.05:  # 超过5%认为是问题
                        report['issues'].append(f"{col} 异常值过多: {outliers} 个 ({outlier_rate:.2%})")
                        report['score'] -= min(10, outliers)
                    else:
                        report['warnings'].append(f"{col} 发现异常值: {outliers} 个")
    
    def _get_expected_timedelta(self, interval: str) -> Optional[timedelta]:
        """获取期望的时间间隔"""
        interval_map = {
            '1m': timedelta(minutes=1),
            '3m': timedelta(minutes=3),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '6h': timedelta(hours=6),
            '8h': timedelta(hours=8),
            '12h': timedelta(hours=12),
            '1d': timedelta(days=1),
            '3d': timedelta(days=3),
            '1w': timedelta(weeks=1),
            '1M': timedelta(days=30)  # 近似
        }
        
        return interval_map.get(interval)
    
    def _calculate_final_score(self, report: Dict[str, Any]) -> None:
        """计算最终评分"""
        # 确保评分在0-100之间
        report['score'] = max(0.0, min(100.0, report['score']))
        
        # 判断是否通过（评分大于70分且没有严重问题）
        critical_issues = [issue for issue in report['issues'] 
                          if any(keyword in issue for keyword in ['缺少必需列', '空值', '逻辑错误'])]
        
        if report['score'] < 70 or critical_issues:
            report['passed'] = False
        else:
            report['passed'] = True


class CacheManager:
    """
    缓存管理器类
    
    负责管理本地CSV文件的存储、读取和验证
    """
    
    def __init__(self, cache_dir: str = "data/"):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = Path(cache_dir)  # 缓存目录路径
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # 创建缓存目录
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 缓存元数据文件路径
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()  # 加载缓存元数据
    
    def _load_metadata(self) -> Dict:
        """
        加载缓存元数据
        
        Returns:
            Dict: 缓存元数据字典
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"加载缓存元数据失败: {e}")
                return {}
        return {}
    
    def _save_metadata(self) -> None:
        """
        保存缓存元数据到文件
        """
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"保存缓存元数据失败: {e}")
    
    def get_cache_file_path(self, symbol: str, interval: str) -> Path:
        """
        获取缓存文件路径
        
        Args:
            symbol: 交易对符号，如 "BTCUSDT"
            interval: 时间间隔，如 "1m", "1h", "1d"
            
        Returns:
            Path: 缓存文件路径
        """
        filename = f"{symbol}_{interval}.csv"
        return self.cache_dir / filename
    
    def cache_exists(self, symbol: str, interval: str) -> bool:
        """
        检查缓存文件是否存在
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            
        Returns:
            bool: 缓存文件是否存在
        """
        cache_file = self.get_cache_file_path(symbol, interval)
        return cache_file.exists() and cache_file.stat().st_size > 0
    
    def get_cache_time_range(self, symbol: str, interval: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        获取缓存数据的时间范围
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            
        Returns:
            Tuple[Optional[datetime], Optional[datetime]]: (最早时间, 最晚时间)
        """
        if not self.cache_exists(symbol, interval):
            return None, None
        
        try:
            # 读取缓存文件的第一行和最后一行获取时间范围
            cache_file = self.get_cache_file_path(symbol, interval)
            
            # 读取第一行（最早时间）
            first_line = pd.read_csv(cache_file, nrows=1)
            if len(first_line) == 0:
                return None, None
            
            # 读取最后一行（最晚时间）
            last_line = pd.read_csv(cache_file).tail(1)
            if len(last_line) == 0:
                return None, None
            
            # 转换时间戳
            start_time = pd.to_datetime(first_line['timestamp'].iloc[0])
            end_time = pd.to_datetime(last_line['timestamp'].iloc[0])
            
            return start_time, end_time
            
        except Exception as e:
            self.logger.error(f"获取缓存时间范围失败 {symbol}_{interval}: {e}")
            return None, None
    
    def load_cache(self, symbol: str, interval: str, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            
        Returns:
            Optional[pd.DataFrame]: 加载的数据，失败时返回None
        """
        if not self.cache_exists(symbol, interval):
            return None
        
        try:
            cache_file = self.get_cache_file_path(symbol, interval)
            
            # 读取CSV文件
            df = pd.read_csv(cache_file)
            
            # 验证数据格式
            if not self._validate_dataframe(df):
                self.logger.warning(f"缓存数据格式不正确: {symbol}_{interval}")
                return None
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 按时间范围过滤
            if start_time is not None:
                df = df[df['timestamp'] >= start_time]
            if end_time is not None:
                df = df[df['timestamp'] <= end_time]
            
            # 按时间戳排序
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"从缓存加载数据: {symbol}_{interval}, {len(df)} 条记录")
            return df
            
        except Exception as e:
            self.logger.error(f"加载缓存失败 {symbol}_{interval}: {e}")
            return None
    
    def save_cache(self, symbol: str, interval: str, df: pd.DataFrame, 
                   append: bool = False) -> bool:
        """
        保存数据到缓存
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            df: 要保存的数据
            append: 是否追加到现有文件
            
        Returns:
            bool: 保存是否成功
        """
        try:
            # 验证数据格式
            if not self._validate_dataframe(df):
                self.logger.error(f"数据格式不正确，无法保存: {symbol}_{interval}")
                return False
            
            cache_file = self.get_cache_file_path(symbol, interval)
            
            # 确保数据按时间排序
            df_sorted = df.sort_values('timestamp').reset_index(drop=True)
            
            if append and cache_file.exists():
                # 追加模式：读取现有数据并合并
                existing_df = self.load_cache(symbol, interval)
                if existing_df is not None:
                    # 合并数据并去重
                    combined_df = pd.concat([existing_df, df_sorted], ignore_index=True)
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                    df_sorted = combined_df
            
            # 保存到CSV文件
            df_sorted.to_csv(cache_file, index=False)
            
            # 更新元数据
            cache_key = f"{symbol}_{interval}"
            self.metadata[cache_key] = {
                'symbol': symbol,
                'interval': interval,
                'records_count': len(df_sorted),
                'start_time': df_sorted['timestamp'].min().isoformat(),
                'end_time': df_sorted['timestamp'].max().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            self._save_metadata()
            
            self.logger.info(f"保存缓存成功: {symbol}_{interval}, {len(df_sorted)} 条记录")
            return True
            
        except Exception as e:
            self.logger.error(f"保存缓存失败 {symbol}_{interval}: {e}")
            return False
    
    def _validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        验证DataFrame格式是否正确
        
        Args:
            df: 要验证的DataFrame
            
        Returns:
            bool: 格式是否正确
        """
        # 检查必需的列
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        if not all(col in df.columns for col in required_columns):
            return False
        
        # 检查数据类型
        if len(df) == 0:
            return True  # 空DataFrame视为有效
        
        try:
            # 尝试转换时间戳
            pd.to_datetime(df['timestamp'])
            
            # 检查价格和成交量字段是否为数值
            for col in ['open', 'high', 'low', 'close', 'volume']:
                pd.to_numeric(df[col])
            
            return True
            
        except Exception:
            return False
    
    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> bool:
        """
        清理缓存文件
        
        Args:
            symbol: 交易对符号（可选，不指定则清理所有）
            interval: 时间间隔（可选，不指定则清理所有）
            
        Returns:
            bool: 清理是否成功
        """
        try:
            if symbol and interval:
                # 清理指定的缓存文件
                cache_file = self.get_cache_file_path(symbol, interval)
                if cache_file.exists():
                    cache_file.unlink()
                    
                # 从元数据中移除
                cache_key = f"{symbol}_{interval}"
                if cache_key in self.metadata:
                    del self.metadata[cache_key]
                    self._save_metadata()
                    
                self.logger.info(f"清理缓存: {symbol}_{interval}")
            else:
                # 清理所有缓存文件
                for cache_file in self.cache_dir.glob("*.csv"):
                    cache_file.unlink()
                
                # 清理元数据
                self.metadata = {}
                self._save_metadata()
                
                self.logger.info("清理所有缓存文件")
            
            return True
            
        except Exception as e:
            self.logger.error(f"清理缓存失败: {e}")
            return False
    
    def get_cache_info(self) -> Dict:
        """
        获取缓存信息
        
        Returns:
            Dict: 缓存信息字典
        """
        info = {
            'cache_dir': str(self.cache_dir),
            'total_files': len(list(self.cache_dir.glob("*.csv"))),
            'total_size_mb': 0,
            'files': []
        }
        
        # 计算总大小和文件列表
        for cache_file in self.cache_dir.glob("*.csv"):
            file_size = cache_file.stat().st_size
            info['total_size_mb'] += file_size / (1024 * 1024)
            
            # 从文件名解析交易对和时间间隔
            parts = cache_file.stem.split('_')
            if len(parts) >= 2:
                symbol = '_'.join(parts[:-1])
                interval = parts[-1]
                
                file_info = {
                    'symbol': symbol,
                    'interval': interval,
                    'file_size_mb': file_size / (1024 * 1024),
                    'file_path': str(cache_file)
                }
                
                # 添加元数据信息
                cache_key = f"{symbol}_{interval}"
                if cache_key in self.metadata:
                    file_info.update(self.metadata[cache_key])
                
                info['files'].append(file_info)
        
        info['total_size_mb'] = round(info['total_size_mb'], 2)
        return info


class DataDownloader:
    """
    数据下载器类
    
    负责从Binance API下载历史K线数据
    """
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 testnet: bool = True, request_delay: float = 0.5):
        """
        初始化数据下载器
        
        Args:
            api_key: Binance API密钥
            api_secret: Binance API密钥
            testnet: 是否使用测试网
            request_delay: 请求间隔时间（秒）
        """
        self.data_provider = BinanceDataProvider(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet
        )  # Binance数据提供者
        self.request_delay = request_delay  # 请求延迟时间
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
    
    def download_klines(self, symbol: str, interval: str,
                       start_time: datetime, end_time: datetime,
                       max_records_per_request: int = 1000) -> Optional[pd.DataFrame]:
        """
        下载K线数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间
            max_records_per_request: 每次请求的最大记录数
            
        Returns:
            Optional[pd.DataFrame]: 下载的数据，失败时返回None
        """
        try:
            all_data = []  # 存储所有下载的数据
            current_start = start_time
            
            self.logger.info(f"开始下载数据: {symbol}_{interval} ({start_time} 到 {end_time})")
            
            while current_start < end_time:
                # 计算当前批次的结束时间
                current_end = min(current_start + self._get_time_delta(interval, max_records_per_request),
                                 end_time)
                
                # 从API获取数据
                batch_data = self.data_provider.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    start_time=current_start,
                    end_time=current_end,
                    limit=max_records_per_request
                )
                
                if batch_data is not None and len(batch_data) > 0:
                    all_data.append(batch_data)
                    self.logger.debug(f"下载批次数据: {len(batch_data)} 条 "
                                    f"({current_start} 到 {current_end})")
                    
                    # 更新下次请求的开始时间
                    if 'timestamp' in batch_data.columns:
                        last_timestamp = pd.to_datetime(batch_data['timestamp'].iloc[-1])
                        current_start = last_timestamp + timedelta(seconds=1)
                    else:
                        current_start = current_end
                else:
                    self.logger.warning(f"批次数据为空: {current_start} 到 {current_end}")
                    current_start = current_end
                
                # 添加请求延迟，避免触发API限制
                time.sleep(self.request_delay)
            
            # 合并所有数据
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # 去重并排序
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
                
                self.logger.info(f"下载完成: {symbol}_{interval}, {len(combined_df)} 条记录")
                return combined_df
            else:
                self.logger.warning(f"未获取到任何数据: {symbol}_{interval}")
                return None
                
        except Exception as e:
            self.logger.error(f"下载数据失败 {symbol}_{interval}: {e}")
            return None
    
    def _get_time_delta(self, interval: str, records_count: int) -> timedelta:
        """
        根据时间间隔和记录数计算时间增量
        
        Args:
            interval: 时间间隔
            records_count: 记录数量
            
        Returns:
            timedelta: 时间增量
        """
        # 时间间隔映射
        interval_map = {
            '1m': timedelta(minutes=1),
            '3m': timedelta(minutes=3),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '30m': timedelta(minutes=30),
            '1h': timedelta(hours=1),
            '2h': timedelta(hours=2),
            '4h': timedelta(hours=4),
            '6h': timedelta(hours=6),
            '8h': timedelta(hours=8),
            '12h': timedelta(hours=12),
            '1d': timedelta(days=1),
            '3d': timedelta(days=3),
            '1w': timedelta(weeks=1),
            '1M': timedelta(days=30)  # 近似值
        }
        
        unit_delta = interval_map.get(interval, timedelta(hours=1))
        return unit_delta * records_count


class KlineDataManager:
    """
    K线数据管理器
    
    主要管理类，提供完整的历史K线数据管理功能
    """
    
    def __init__(self, cache_dir: str = "data/", api_key: Optional[str] = None,
                 api_secret: Optional[str] = None, testnet: bool = True,
                 request_delay: float = 0.5, enable_quality_check: bool = True):
        """
        初始化K线数据管理器
        
        Args:
            cache_dir: 缓存目录
            api_key: Binance API密钥
            api_secret: Binance API密钥
            testnet: 是否使用测试网
            request_delay: 请求延迟时间
            enable_quality_check: 是否启用数据质量检查
        """
        self.cache_manager = CacheManager(cache_dir)  # 缓存管理器
        self.data_downloader = DataDownloader(
            api_key=api_key,
            api_secret=api_secret,
            testnet=testnet,
            request_delay=request_delay
        )  # 数据下载器
        
        # 数据质量检查器
        self.enable_quality_check = enable_quality_check
        if enable_quality_check:
            self.quality_checker = DataQualityChecker()
            self.quality_reports: List[Dict[str, Any]] = []  # 质量报告历史
        else:
            self.quality_checker = None
            self.quality_reports = []
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 设置日志格式
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def get_klines(self, symbol: str, interval: str, 
                   start_time: Union[str, datetime], end_time: Union[str, datetime],
                   force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        获取K线数据（主要接口）
        
        Args:
            symbol: 交易对符号，如 "BTCUSDT"
            interval: 时间间隔，如 "1m", "1h", "1d"
            start_time: 开始时间，支持字符串或datetime
            end_time: 结束时间，支持字符串或datetime
            force_refresh: 是否强制刷新（忽略缓存）
            
        Returns:
            Optional[pd.DataFrame]: K线数据，包含 ["timestamp", "open", "high", "low", "close", "volume"] 列
        """
        try:
            # 转换时间格式
            start_dt = self._parse_time(start_time)
            end_dt = self._parse_time(end_time)
            
            if start_dt is None or end_dt is None:
                self.logger.error(f"时间格式错误: {start_time} 到 {end_time}")
                return None
            
            if start_dt >= end_dt:
                self.logger.error(f"开始时间必须早于结束时间: {start_time} 到 {end_time}")
                return None
            
            self.logger.info(f"获取K线数据: {symbol}_{interval} ({start_dt} 到 {end_dt})")
            
            # 如果强制刷新，直接下载新数据
            if force_refresh:
                return self._download_and_cache(symbol, interval, start_dt, end_dt)
            
            # 检查缓存
            cached_data = self._get_cached_data(symbol, interval, start_dt, end_dt)
            if cached_data is not None:
                return cached_data
            
            # 缓存未命中，需要下载数据
            return self._download_and_cache(symbol, interval, start_dt, end_dt)
            
        except Exception as e:
            self.logger.error(f"获取K线数据失败: {e}")
            return None
    
    def _parse_time(self, time_input: Union[str, datetime]) -> Optional[datetime]:
        """
        解析时间输入
        
        Args:
            time_input: 时间输入（字符串或datetime）
            
        Returns:
            Optional[datetime]: 解析后的时间，失败时返回None
        """
        if isinstance(time_input, datetime):
            return time_input
        
        if isinstance(time_input, str):
            # 支持多种时间格式
            formats = [
                '%Y-%m-%d',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M',
                '%Y/%m/%d',
                '%Y/%m/%d %H:%M:%S',
                '%Y/%m/%d %H:%M'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(time_input, fmt)
                except ValueError:
                    continue
            
            # 尝试使用pandas解析
            try:
                return pd.to_datetime(time_input)
            except Exception:
                pass
        
        return None
    
    def _get_cached_data(self, symbol: str, interval: str,
                        start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            Optional[pd.DataFrame]: 缓存的数据，如果需要更新则返回None
        """
        # 检查缓存是否存在
        if not self.cache_manager.cache_exists(symbol, interval):
            self.logger.info(f"缓存不存在: {symbol}_{interval}")
            return None
        
        # 获取缓存的时间范围
        cache_start, cache_end = self.cache_manager.get_cache_time_range(symbol, interval)
        
        if cache_start is None or cache_end is None:
            self.logger.warning(f"无法获取缓存时间范围: {symbol}_{interval}")
            return None
        
        # 检查缓存是否完全覆盖请求的时间范围
        if cache_start <= start_time and cache_end >= end_time:
            # 缓存完全覆盖，直接加载
            self.logger.info(f"缓存命中: {symbol}_{interval}")
            return self.cache_manager.load_cache(symbol, interval, start_time, end_time)
        else:
            # 缓存部分覆盖或不覆盖，需要更新
            self.logger.info(f"缓存需要更新: {symbol}_{interval}, "
                           f"缓存范围: {cache_start} 到 {cache_end}, "
                           f"请求范围: {start_time} 到 {end_time}")
            return None
    
    def _download_and_cache(self, symbol: str, interval: str,
                           start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """
        下载数据并缓存
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            Optional[pd.DataFrame]: 下载的数据
        """
        # 下载数据
        downloaded_data = self.data_downloader.download_klines(
            symbol, interval, start_time, end_time
        )
        
        if downloaded_data is None or len(downloaded_data) == 0:
            self.logger.error(f"下载数据失败: {symbol}_{interval}")
            return None
        
        # 数据质量检查
        if self.enable_quality_check and self.quality_checker:
            try:
                quality_report = self.quality_checker.check_data_quality(
                    downloaded_data, symbol, interval
                )
                
                # 保存质量报告
                self.quality_reports.append(quality_report)
                
                # 记录质量信息
                if quality_report['score'] < 80:
                    self.logger.warning(f"数据质量警告 {symbol}_{interval}: "
                                      f"评分 {quality_report['score']:.1f}/100")
                    if quality_report['issues']:
                        self.logger.warning(f"问题: {quality_report['issues']}")
                
                if not quality_report['passed']:
                    self.logger.error(f"数据质量检查未通过: {symbol}_{interval}")
                    # 可以选择是否返回质量不佳的数据
                    # return None  # 取消注释以拒绝低质量数据
                
            except Exception as e:
                self.logger.error(f"数据质量检查失败: {symbol}_{interval} - {e}")
        
        # 保存到缓存
        if self.cache_manager.save_cache(symbol, interval, downloaded_data):
            self.logger.info(f"数据已缓存: {symbol}_{interval}")
        else:
            self.logger.warning(f"缓存保存失败: {symbol}_{interval}")
        
        return downloaded_data
    
    def batch_download(self, symbols: List[str], intervals: List[str],
                      start_time: Union[str, datetime], end_time: Union[str, datetime],
                      force_refresh: bool = False) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
        """
        批量下载多个币种的K线数据
        
        Args:
            symbols: 交易对符号列表
            intervals: 时间间隔列表
            start_time: 开始时间
            end_time: 结束时间
            force_refresh: 是否强制刷新
            
        Returns:
            Dict[str, Dict[str, Optional[pd.DataFrame]]]: 批量下载结果
        """
        results = {}
        
        total_tasks = len(symbols) * len(intervals)
        current_task = 0
        
        self.logger.info(f"开始批量下载: {len(symbols)} 个币种, {len(intervals)} 个时间间隔")
        
        for symbol in symbols:
            results[symbol] = {}
            
            for interval in intervals:
                current_task += 1
                
                self.logger.info(f"批量下载进度: {current_task}/{total_tasks} "
                               f"({symbol}_{interval})")
                
                # 获取单个币种的数据
                data = self.get_klines(symbol, interval, start_time, end_time, force_refresh)
                results[symbol][interval] = data
                
                # 添加延迟，避免API限制
                time.sleep(self.data_downloader.request_delay)
        
        self.logger.info(f"批量下载完成: {total_tasks} 个任务")
        return results
    
    def get_cache_info(self) -> Dict:
        """
        获取缓存信息
        
        Returns:
            Dict: 缓存信息
        """
        return self.cache_manager.get_cache_info()
    
    def clear_cache(self, symbol: Optional[str] = None, interval: Optional[str] = None) -> bool:
        """
        清理缓存
        
        Args:
            symbol: 交易对符号（可选）
            interval: 时间间隔（可选）
            
        Returns:
            bool: 清理是否成功
        """
        return self.cache_manager.clear_cache(symbol, interval)
    
    def check_data_quality(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """
        检查指定数据的质量
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            
        Returns:
            Optional[Dict[str, Any]]: 质量检查结果，如果没有数据则返回None
        """
        if not self.enable_quality_check or not self.quality_checker:
            return None
        
        # 从缓存加载数据
        cached_data = self.cache_manager.load_cache(symbol, interval)
        if cached_data is None or cached_data.empty:
            return None
        
        # 执行质量检查
        return self.quality_checker.check_data_quality(cached_data, symbol, interval)
    
    def get_quality_reports(self, symbol: Optional[str] = None, 
                           interval: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取质量报告
        
        Args:
            symbol: 交易对符号（可选，筛选特定交易对的报告）
            interval: 时间间隔（可选，筛选特定时间间隔的报告）
            
        Returns:
            List[Dict[str, Any]]: 质量报告列表
        """
        if not self.quality_reports:
            return []
        
        reports = self.quality_reports
        
        # 按symbol筛选
        if symbol:
            reports = [r for r in reports if r.get('symbol') == symbol]
        
        # 按interval筛选
        if interval:
            reports = [r for r in reports if r.get('interval') == interval]
        
        return reports
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """
        获取质量汇总信息
        
        Returns:
            Dict[str, Any]: 质量汇总信息
        """
        if not self.quality_reports:
            return {
                'total_reports': 0,
                'average_score': 0,
                'passed_count': 0,
                'failed_count': 0,
                'latest_report': None
            }
        
        scores = [r['score'] for r in self.quality_reports]
        passed_count = sum(1 for r in self.quality_reports if r['passed'])
        failed_count = len(self.quality_reports) - passed_count
        
        # 获取最新报告
        latest_report = max(self.quality_reports, 
                          key=lambda r: r['check_time'])
        
        return {
            'total_reports': len(self.quality_reports),
            'average_score': sum(scores) / len(scores),
            'passed_count': passed_count,
            'failed_count': failed_count,
            'latest_report': latest_report,
            'score_distribution': {
                'excellent': sum(1 for s in scores if s >= 90),
                'good': sum(1 for s in scores if 80 <= s < 90),
                'fair': sum(1 for s in scores if 70 <= s < 80),
                'poor': sum(1 for s in scores if s < 70)
            }
        }