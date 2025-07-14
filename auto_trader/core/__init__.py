"""
核心模块初始化文件

这个模块包含交易系统的核心功能：
- 数据管理
- 经纪商接口
- 账户管理
- 风险控制
- 回测引擎
"""

from .data import DataManager, BinanceDataProvider, KlineData, MarketTicker
from .broker import BinanceBroker, SimulatedBroker, Order, Position, OrderStatus, OrderSide
from .account import AccountManager, AccountBalance, TradeRecord, PerformanceMetrics, AccountType
from .risk import RiskManager, RiskLimits, RiskCheckResult, RiskLevel, RiskMetrics
from .backtest import BacktestEngine, BacktestConfig, BacktestResult, BacktestStatus

__all__ = [
    # 数据模块
    'DataManager',
    'BinanceDataProvider', 
    'KlineData',
    'MarketTicker',
    
    # 经纪商模块
    'BinanceBroker',
    'SimulatedBroker',
    'Order',
    'Position', 
    'OrderStatus',
    'OrderSide',
    
    # 账户模块
    'AccountManager',
    'AccountBalance',
    'TradeRecord',
    'PerformanceMetrics',
    'AccountType',
    
    # 风险模块
    'RiskManager',
    'RiskLimits',
    'RiskCheckResult', 
    'RiskLevel',
    'RiskMetrics',
    
    # 回测模块
    'BacktestEngine',
    'BacktestConfig',
    'BacktestResult',
    'BacktestStatus'
]