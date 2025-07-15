"""
UI模块

提供完整的Web界面，包括：
- 主仪表板
- 策略监控界面
- 资产管理
- 回测分析
- 交易日志
"""

from .dashboard import StreamlitDashboard
from .pages import (
    StrategyMonitor,
    AssetManager, 
    BacktestAnalyzer,
    TradeLogger,
    ConfigManager
)

__all__ = [
    'StreamlitDashboard',
    'StrategyMonitor',
    'AssetManager',
    'BacktestAnalyzer', 
    'TradeLogger',
    'ConfigManager'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'TradingFan'
__description__ = 'AutoTrader专业量化交易系统Web界面'