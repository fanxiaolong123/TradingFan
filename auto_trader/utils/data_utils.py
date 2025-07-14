"""
数据处理工具模块

这个模块提供了数据处理的各种工具函数，包括：
- 数据清洗和预处理
- 技术指标计算
- 数据转换和格式化
- 数据验证和异常处理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
import json
import csv
from pathlib import Path
import warnings

from .logger import get_logger

logger = get_logger(__name__)


def clean_ohlcv_data(df: pd.DataFrame, 
                     remove_duplicates: bool = True,
                     fill_missing: bool = True,
                     validate_prices: bool = True,
                     remove_outliers: bool = False,
                     outlier_threshold: float = 3.0) -> pd.DataFrame:
    """
    清洗OHLCV数据
    
    Args:
        df: 包含OHLCV数据的DataFrame
        remove_duplicates: 是否移除重复数据
        fill_missing: 是否填充缺失值
        validate_prices: 是否验证价格数据
        remove_outliers: 是否移除异常值
        outlier_threshold: 异常值阈值（标准差倍数）
        
    Returns:
        pd.DataFrame: 清洗后的数据
    """
    logger.info("开始清洗OHLCV数据")
    
    if df.empty:
        logger.warning("输入数据为空")
        return df
    
    # 创建副本
    cleaned_df = df.copy()
    
    # 确保时间戳列存在且为datetime类型
    if 'timestamp' in cleaned_df.columns:
        cleaned_df['timestamp'] = pd.to_datetime(cleaned_df['timestamp'])
        cleaned_df = cleaned_df.sort_values('timestamp')
    
    # 移除重复数据
    if remove_duplicates:
        initial_count = len(cleaned_df)
        if 'timestamp' in cleaned_df.columns:
            cleaned_df = cleaned_df.drop_duplicates(subset=['timestamp'], keep='first')
        else:
            cleaned_df = cleaned_df.drop_duplicates()
        
        removed_count = initial_count - len(cleaned_df)
        if removed_count > 0:
            logger.info(f"移除了 {removed_count} 条重复数据")
    
    # 验证价格数据
    if validate_prices:
        price_columns = ['open', 'high', 'low', 'close']
        available_columns = [col for col in price_columns if col in cleaned_df.columns]
        
        # 检查负价格
        for col in available_columns:
            negative_mask = cleaned_df[col] <= 0
            if negative_mask.any():
                logger.warning(f"发现 {negative_mask.sum()} 个负价格或零价格在 {col} 列")
                cleaned_df = cleaned_df[~negative_mask]
        
        # 检查OHLC关系
        if all(col in cleaned_df.columns for col in ['open', 'high', 'low', 'close']):
            # High应该是最高价
            invalid_high = (cleaned_df['high'] < cleaned_df[['open', 'close']].max(axis=1))
            if invalid_high.any():
                logger.warning(f"发现 {invalid_high.sum()} 个无效的最高价")
                cleaned_df = cleaned_df[~invalid_high]
            
            # Low应该是最低价
            invalid_low = (cleaned_df['low'] > cleaned_df[['open', 'close']].min(axis=1))
            if invalid_low.any():
                logger.warning(f"发现 {invalid_low.sum()} 个无效的最低价")
                cleaned_df = cleaned_df[~invalid_low]
    
    # 移除异常值
    if remove_outliers:
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in numeric_columns if col in cleaned_df.columns]
        
        for col in available_columns:
            # 计算Z-score
            z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
            outlier_mask = z_scores > outlier_threshold
            
            if outlier_mask.any():
                logger.warning(f"在 {col} 列中发现 {outlier_mask.sum()} 个异常值")
                cleaned_df = cleaned_df[~outlier_mask]
    
    # 填充缺失值
    if fill_missing:
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            logger.info(f"填充 {missing_count} 个缺失值")
            
            # 对于价格数据，使用前向填充
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in cleaned_df.columns:
                    cleaned_df[col] = cleaned_df[col].fillna(method='ffill')
                    cleaned_df[col] = cleaned_df[col].fillna(method='bfill')
            
            # 对于成交量，使用0填充
            if 'volume' in cleaned_df.columns:
                cleaned_df['volume'] = cleaned_df['volume'].fillna(0)
    
    # 重置索引
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    logger.info(f"数据清洗完成，从 {len(df)} 条记录清洗为 {len(cleaned_df)} 条记录")
    
    return cleaned_df


def calculate_returns(df: pd.DataFrame, 
                     price_column: str = 'close',
                     periods: int = 1,
                     method: str = 'simple') -> pd.Series:
    """
    计算收益率
    
    Args:
        df: 包含价格数据的DataFrame
        price_column: 价格列名
        periods: 计算周期
        method: 计算方法 ('simple', 'log')
        
    Returns:
        pd.Series: 收益率序列
    """
    if price_column not in df.columns:
        raise ValueError(f"列 {price_column} 不存在")
    
    prices = df[price_column]
    
    if method == 'simple':
        returns = prices.pct_change(periods=periods)
    elif method == 'log':
        returns = np.log(prices / prices.shift(periods))
    else:
        raise ValueError(f"不支持的计算方法: {method}")
    
    return returns


def calculate_volatility(df: pd.DataFrame, 
                        price_column: str = 'close',
                        window: int = 20,
                        annualize: bool = True) -> pd.Series:
    """
    计算波动率
    
    Args:
        df: 包含价格数据的DataFrame
        price_column: 价格列名
        window: 计算窗口
        annualize: 是否年化
        
    Returns:
        pd.Series: 波动率序列
    """
    returns = calculate_returns(df, price_column, method='log')
    volatility = returns.rolling(window=window).std()
    
    if annualize:
        # 假设一年有252个交易日
        volatility = volatility * np.sqrt(252)
    
    return volatility


def calculate_moving_average(df: pd.DataFrame, 
                           price_column: str = 'close',
                           window: int = 20,
                           ma_type: str = 'simple') -> pd.Series:
    """
    计算移动平均线
    
    Args:
        df: 包含价格数据的DataFrame
        price_column: 价格列名
        window: 计算窗口
        ma_type: 移动平均类型 ('simple', 'exponential', 'weighted')
        
    Returns:
        pd.Series: 移动平均线序列
    """
    if price_column not in df.columns:
        raise ValueError(f"列 {price_column} 不存在")
    
    prices = df[price_column]
    
    if ma_type == 'simple':
        ma = prices.rolling(window=window).mean()
    elif ma_type == 'exponential':
        ma = prices.ewm(span=window).mean()
    elif ma_type == 'weighted':
        weights = np.arange(1, window + 1)
        ma = prices.rolling(window=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    else:
        raise ValueError(f"不支持的移动平均类型: {ma_type}")
    
    return ma


def calculate_bollinger_bands(df: pd.DataFrame, 
                             price_column: str = 'close',
                             window: int = 20,
                             num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算布林带
    
    Args:
        df: 包含价格数据的DataFrame
        price_column: 价格列名
        window: 计算窗口
        num_std: 标准差倍数
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (中轨, 上轨, 下轨)
    """
    prices = df[price_column]
    
    # 计算中轨（移动平均线）
    middle_band = prices.rolling(window=window).mean()
    
    # 计算标准差
    std = prices.rolling(window=window).std()
    
    # 计算上轨和下轨
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return middle_band, upper_band, lower_band


def calculate_rsi(df: pd.DataFrame, 
                  price_column: str = 'close',
                  window: int = 14) -> pd.Series:
    """
    计算相对强弱指数(RSI)
    
    Args:
        df: 包含价格数据的DataFrame
        price_column: 价格列名
        window: 计算窗口
        
    Returns:
        pd.Series: RSI序列
    """
    prices = df[price_column]
    
    # 计算价格变化
    delta = prices.diff()
    
    # 分离上涨和下跌
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # 计算平均收益和平均损失
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # 计算RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(df: pd.DataFrame, 
                   price_column: str = 'close',
                   fast_period: int = 12,
                   slow_period: int = 26,
                   signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    计算MACD指标
    
    Args:
        df: 包含价格数据的DataFrame
        price_column: 价格列名
        fast_period: 快速EMA周期
        slow_period: 慢速EMA周期
        signal_period: 信号线周期
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (MACD线, 信号线, 柱状图)
    """
    prices = df[price_column]
    
    # 计算快速和慢速EMA
    ema_fast = prices.ewm(span=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period).mean()
    
    # 计算MACD线
    macd_line = ema_fast - ema_slow
    
    # 计算信号线
    signal_line = macd_line.ewm(span=signal_period).mean()
    
    # 计算柱状图
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def resample_data(df: pd.DataFrame, 
                  timeframe: str,
                  timestamp_column: str = 'timestamp') -> pd.DataFrame:
    """
    重采样数据到指定时间框架
    
    Args:
        df: 包含时间序列数据的DataFrame
        timeframe: 目标时间框架 ('1min', '5min', '1H', '1D'等)
        timestamp_column: 时间戳列名
        
    Returns:
        pd.DataFrame: 重采样后的数据
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"时间戳列 {timestamp_column} 不存在")
    
    # 确保时间戳列为datetime类型
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    
    # 设置时间戳为索引
    df = df.set_index(timestamp_column)
    
    # 定义聚合规则
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # 只保留存在的列
    available_agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}
    
    # 重采样
    resampled = df.resample(timeframe).agg(available_agg_rules)
    
    # 移除空值行
    resampled = resampled.dropna()
    
    # 重置索引
    resampled = resampled.reset_index()
    
    return resampled


def normalize_data(df: pd.DataFrame, 
                   columns: Optional[List[str]] = None,
                   method: str = 'minmax') -> pd.DataFrame:
    """
    标准化数据
    
    Args:
        df: 输入DataFrame
        columns: 要标准化的列名列表，None表示所有数值列
        method: 标准化方法 ('minmax', 'zscore', 'robust')
        
    Returns:
        pd.DataFrame: 标准化后的数据
    """
    normalized_df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"列 {col} 不存在，跳过")
            continue
        
        if method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val != min_val:
                normalized_df[col] = (df[col] - min_val) / (max_val - min_val)
        
        elif method == 'zscore':
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val != 0:
                normalized_df[col] = (df[col] - mean_val) / std_val
        
        elif method == 'robust':
            median_val = df[col].median()
            mad_val = (df[col] - median_val).abs().median()
            if mad_val != 0:
                normalized_df[col] = (df[col] - median_val) / mad_val
        
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
    
    return normalized_df


def detect_missing_periods(df: pd.DataFrame, 
                          timestamp_column: str = 'timestamp',
                          expected_interval: str = '1H') -> List[Tuple[datetime, datetime]]:
    """
    检测缺失的时间段
    
    Args:
        df: 包含时间序列数据的DataFrame
        timestamp_column: 时间戳列名
        expected_interval: 期望的时间间隔
        
    Returns:
        List[Tuple[datetime, datetime]]: 缺失时间段列表
    """
    if timestamp_column not in df.columns:
        raise ValueError(f"时间戳列 {timestamp_column} 不存在")
    
    # 确保时间戳列为datetime类型并排序
    df = df.copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    df = df.sort_values(timestamp_column)
    
    # 生成期望的时间序列
    start_time = df[timestamp_column].min()
    end_time = df[timestamp_column].max()
    
    expected_timestamps = pd.date_range(start=start_time, end=end_time, freq=expected_interval)
    
    # 找到缺失的时间戳
    actual_timestamps = set(df[timestamp_column])
    missing_timestamps = [ts for ts in expected_timestamps if ts not in actual_timestamps]
    
    # 将连续的缺失时间戳合并为时间段
    missing_periods = []
    if missing_timestamps:
        missing_timestamps.sort()
        
        period_start = missing_timestamps[0]
        period_end = missing_timestamps[0]
        
        for i in range(1, len(missing_timestamps)):
            current_ts = missing_timestamps[i]
            expected_next = period_end + pd.Timedelta(expected_interval)
            
            if current_ts == expected_next:
                period_end = current_ts
            else:
                missing_periods.append((period_start, period_end))
                period_start = current_ts
                period_end = current_ts
        
        missing_periods.append((period_start, period_end))
    
    return missing_periods


def save_data(df: pd.DataFrame, 
              file_path: str,
              format: str = 'csv',
              **kwargs) -> None:
    """
    保存数据到文件
    
    Args:
        df: 要保存的DataFrame
        file_path: 文件路径
        format: 文件格式 ('csv', 'json', 'parquet', 'excel')
        **kwargs: 额外参数
    """
    # 创建目录
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'csv':
            df.to_csv(file_path, index=False, encoding='utf-8', **kwargs)
        elif format == 'json':
            df.to_json(file_path, orient='records', date_format='iso', **kwargs)
        elif format == 'parquet':
            df.to_parquet(file_path, index=False, **kwargs)
        elif format == 'excel':
            df.to_excel(file_path, index=False, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {format}")
        
        logger.info(f"数据已保存到: {file_path}")
        
    except Exception as e:
        logger.error(f"保存数据失败: {e}")
        raise


def load_data(file_path: str, 
              format: str = None,
              **kwargs) -> pd.DataFrame:
    """
    从文件加载数据
    
    Args:
        file_path: 文件路径
        format: 文件格式，None表示自动检测
        **kwargs: 额外参数
        
    Returns:
        pd.DataFrame: 加载的数据
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 自动检测文件格式
    if format is None:
        format = Path(file_path).suffix.lower()[1:]
    
    try:
        if format == 'csv':
            df = pd.read_csv(file_path, **kwargs)
        elif format == 'json':
            df = pd.read_json(file_path, **kwargs)
        elif format == 'parquet':
            df = pd.read_parquet(file_path, **kwargs)
        elif format in ['xlsx', 'xls']:
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError(f"不支持的文件格式: {format}")
        
        logger.info(f"数据已从 {file_path} 加载，共 {len(df)} 行")
        
        return df
        
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise


def validate_data_quality(df: pd.DataFrame, 
                         timestamp_column: str = 'timestamp') -> Dict[str, Any]:
    """
    验证数据质量
    
    Args:
        df: 要验证的DataFrame
        timestamp_column: 时间戳列名
        
    Returns:
        Dict[str, Any]: 数据质量报告
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': {},
        'duplicate_rows': 0,
        'data_types': {},
        'time_coverage': {},
        'quality_score': 0.0
    }
    
    # 检查缺失值
    missing_values = df.isnull().sum()
    report['missing_values'] = missing_values.to_dict()
    
    # 检查重复行
    if timestamp_column in df.columns:
        report['duplicate_rows'] = df.duplicated(subset=[timestamp_column]).sum()
    else:
        report['duplicate_rows'] = df.duplicated().sum()
    
    # 检查数据类型
    report['data_types'] = df.dtypes.astype(str).to_dict()
    
    # 检查时间覆盖
    if timestamp_column in df.columns:
        df_temp = df.copy()
        df_temp[timestamp_column] = pd.to_datetime(df_temp[timestamp_column])
        
        report['time_coverage'] = {
            'start_time': df_temp[timestamp_column].min().isoformat(),
            'end_time': df_temp[timestamp_column].max().isoformat(),
            'time_span_days': (df_temp[timestamp_column].max() - df_temp[timestamp_column].min()).days
        }
    
    # 计算质量分数
    quality_score = 1.0
    
    # 扣分项：缺失值
    missing_ratio = missing_values.sum() / (len(df) * len(df.columns))
    quality_score -= missing_ratio * 0.3
    
    # 扣分项：重复行
    duplicate_ratio = report['duplicate_rows'] / len(df) if len(df) > 0 else 0
    quality_score -= duplicate_ratio * 0.2
    
    # 确保分数在0-1之间
    report['quality_score'] = max(0.0, min(1.0, quality_score))
    
    return report


def create_features(df: pd.DataFrame, 
                   price_column: str = 'close',
                   volume_column: str = 'volume') -> pd.DataFrame:
    """
    创建技术分析特征
    
    Args:
        df: 包含价格数据的DataFrame
        price_column: 价格列名
        volume_column: 成交量列名
        
    Returns:
        pd.DataFrame: 包含特征的DataFrame
    """
    features_df = df.copy()
    
    # 收益率特征
    features_df['returns'] = calculate_returns(df, price_column)
    features_df['returns_1d'] = features_df['returns'].shift(1)
    features_df['returns_5d'] = features_df['returns'].rolling(5).mean()
    features_df['returns_20d'] = features_df['returns'].rolling(20).mean()
    
    # 移动平均线特征
    features_df['sma_5'] = calculate_moving_average(df, price_column, 5)
    features_df['sma_20'] = calculate_moving_average(df, price_column, 20)
    features_df['sma_50'] = calculate_moving_average(df, price_column, 50)
    
    # 移动平均线比率
    features_df['price_sma5_ratio'] = features_df[price_column] / features_df['sma_5']
    features_df['price_sma20_ratio'] = features_df[price_column] / features_df['sma_20']
    features_df['sma5_sma20_ratio'] = features_df['sma_5'] / features_df['sma_20']
    
    # 波动率特征
    features_df['volatility_5d'] = calculate_volatility(df, price_column, 5)
    features_df['volatility_20d'] = calculate_volatility(df, price_column, 20)
    
    # 布林带特征
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df, price_column)
    features_df['bb_middle'] = bb_middle
    features_df['bb_upper'] = bb_upper
    features_df['bb_lower'] = bb_lower
    features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
    features_df['bb_position'] = (features_df[price_column] - bb_lower) / (bb_upper - bb_lower)
    
    # RSI特征
    features_df['rsi'] = calculate_rsi(df, price_column)
    
    # MACD特征
    macd_line, signal_line, histogram = calculate_macd(df, price_column)
    features_df['macd_line'] = macd_line
    features_df['macd_signal'] = signal_line
    features_df['macd_histogram'] = histogram
    
    # 成交量特征
    if volume_column in df.columns:
        features_df['volume_sma_20'] = features_df[volume_column].rolling(20).mean()
        features_df['volume_ratio'] = features_df[volume_column] / features_df['volume_sma_20']
        features_df['price_volume'] = features_df[price_column] * features_df[volume_column]
    
    # 价格位置特征
    features_df['high_low_ratio'] = df['high'] / df['low'] if 'high' in df.columns and 'low' in df.columns else None
    features_df['close_high_ratio'] = df['close'] / df['high'] if 'close' in df.columns and 'high' in df.columns else None
    features_df['close_low_ratio'] = df['close'] / df['low'] if 'close' in df.columns and 'low' in df.columns else None
    
    # 移除无限值和NaN值
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    
    return features_df