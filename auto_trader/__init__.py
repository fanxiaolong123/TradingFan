"""
AutoTrader 量化交易系统

这是一个完整的量化自动交易系统，支持策略回测、模拟交易和实盘交易。
"""

__version__ = "1.0.0"
__author__ = "TradingFan"
__email__ = "trading@fan.com"
__description__ = "全球顶尖的量化自动交易系统"

# 导入核心模块
from . import core
from . import strategies
from . import utils

__all__ = [
    'core',
    'strategies', 
    'utils'
]