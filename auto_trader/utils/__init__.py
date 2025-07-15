"""
工具模块初始化文件

这个模块提供了系统所需的各种工具函数，包括：
- 配置管理
- 日志记录
- 数据处理
- 时间工具
"""

from .config import ConfigManager, get_config, reload_config
from .logger import (
    setup_logging, 
    get_logger, 
    log_function_call, 
    log_execution_time,
    with_logging,
    LogContext
)
from .data_utils import (
    clean_ohlcv_data,
    calculate_returns,
    calculate_volatility,
    calculate_moving_average,
    calculate_bollinger_bands,
    calculate_rsi,
    calculate_macd,
    resample_data,
    normalize_data,
    detect_missing_periods,
    save_data,
    load_data,
    validate_data_quality,
    create_features
)
from .time_utils import (
    TimeUtils,
    now,
    to_utc,
    from_utc,
    is_trading_time,
    format_duration,
    parse_timeframe,
    get_period_start,
    get_period_end
)
from .signal_visualizer import SignalVisualizer, create_sample_data

__all__ = [
    # 配置管理
    'ConfigManager',
    'get_config',
    'reload_config',
    
    # 日志记录
    'setup_logging',
    'get_logger',
    'log_function_call',
    'log_execution_time',
    'with_logging',
    'LogContext',
    
    # 数据处理
    'clean_ohlcv_data',
    'calculate_returns',
    'calculate_volatility',
    'calculate_moving_average',
    'calculate_bollinger_bands',
    'calculate_rsi',
    'calculate_macd',
    'resample_data',
    'normalize_data',
    'detect_missing_periods',
    'save_data',
    'load_data',
    'validate_data_quality',
    'create_features',
    
    # 时间工具
    'TimeUtils',
    'now',
    'to_utc',
    'from_utc',
    'is_trading_time',
    'format_duration',
    'parse_timeframe',
    'get_period_start',
    'get_period_end',
    
    # 信号可视化
    'SignalVisualizer',
    'create_sample_data'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'TradingFan'
__email__ = 'trading@fan.com'
__description__ = '量化交易系统工具模块'