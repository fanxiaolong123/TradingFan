"""
UI!W���

�*!WЛWebLb(�
- Ve�L�ѧ
- ��D���U:
- �KӜ��
- ���U:
- VeMn�
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

# H,�o
__version__ = '1.0.0'
__author__ = 'TradingFan'
__description__ = 'AutoTrader����WebLb'