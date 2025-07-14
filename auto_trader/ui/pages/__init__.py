"""
UI页面模块初始化文件

这个模块包含各个功能页面类：
- StrategyMonitor: 策略监控页面
- AssetManager: 资产管理页面  
- BacktestAnalyzer: 回测分析页面
- TradeLogger: 交易日志页面
- ConfigManager: 配置管理页面
"""

from .strategy_monitor import StrategyMonitor
from .asset_manager import AssetManager
from .backtest_analyzer import BacktestAnalyzer
from .trade_logger import TradeLogger
from .config_manager import ConfigManager

__all__ = [
    'StrategyMonitor',
    'AssetManager', 
    'BacktestAnalyzer',
    'TradeLogger',
    'ConfigManager'
]