"""
策略抽象基类模块

这个模块定义了所有交易策略必须遵循的统一接口和数据结构。
每个具体策略都应该继承Strategy基类并实现其抽象方法。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
from datetime import datetime


class SignalType(Enum):
    """
    交易信号类型枚举
    
    BUY: 买入信号
    SELL: 卖出信号
    HOLD: 持有信号（不做任何操作）
    CLOSE_LONG: 平多仓信号
    CLOSE_SHORT: 平空仓信号
    """
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"


class OrderType(Enum):
    """
    订单类型枚举
    
    MARKET: 市价单
    LIMIT: 限价单
    STOP_LOSS: 止损单
    TAKE_PROFIT: 止盈单
    """
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


@dataclass
class TradeSignal:
    """
    交易信号数据类
    
    包含策略生成的交易信号的所有必要信息
    """
    # 基本信号信息
    symbol: str                          # 交易对符号，如"BTCUSDT"
    signal_type: SignalType             # 信号类型（买入/卖出/持有等）
    timestamp: datetime                 # 信号生成时间
    
    # 价格和数量信息
    price: Optional[float] = None       # 建议交易价格（None表示市价）
    quantity: Optional[float] = None    # 交易数量（None表示按仓位管理计算）
    quantity_percent: Optional[float] = None  # 使用总资金的百分比（0.0-1.0）
    
    # 订单参数
    order_type: OrderType = OrderType.MARKET  # 订单类型
    
    # 风控参数
    stop_loss: Optional[float] = None   # 止损价格
    take_profit: Optional[float] = None # 止盈价格
    
    # 策略信息
    strategy_name: str = ""             # 生成信号的策略名称
    confidence: float = 1.0             # 信号置信度（0.0-1.0）
    
    # 额外信息
    metadata: Dict[str, Any] = None     # 策略相关的额外数据
    
    def __post_init__(self):
        """初始化后处理，设置默认值"""
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StrategyConfig:
    """
    策略配置数据类
    
    存储策略运行所需的所有配置参数
    """
    # 基本配置
    name: str                           # 策略名称
    symbol: str                         # 交易对
    timeframe: str                      # 时间周期（如"1m", "5m", "1h"）
    
    # 策略参数（由具体策略定义）
    parameters: Dict[str, Any]          # 策略特定参数
    
    # 风控配置
    max_position_percent: float = 0.1   # 最大仓位占总资金比例
    stop_loss_percent: float = 0.02     # 止损百分比
    take_profit_percent: float = 0.04   # 止盈百分比
    
    # 运行配置
    enabled: bool = True                # 策略是否启用
    dry_run: bool = False               # 是否为模拟模式
    
    def __post_init__(self):
        """初始化后处理，设置默认值"""
        if self.parameters is None:
            self.parameters = {}


@dataclass
class OrderFillEvent:
    """
    订单成交事件数据类
    
    当订单成交时传递给策略的事件信息
    """
    order_id: str                       # 订单ID
    symbol: str                         # 交易对
    side: str                          # 买卖方向（"BUY"/"SELL"）
    quantity: float                    # 成交数量
    price: float                       # 成交价格
    timestamp: datetime                # 成交时间
    commission: float = 0.0            # 手续费
    commission_asset: str = ""         # 手续费币种
    
    # 额外信息
    order_type: str = ""               # 订单类型
    status: str = ""                   # 订单状态
    metadata: Dict[str, Any] = None    # 额外数据
    
    def __post_init__(self):
        """初始化后处理，设置默认值"""
        if self.metadata is None:
            self.metadata = {}


class Strategy(ABC):
    """
    策略抽象基类
    
    所有交易策略都必须继承这个基类并实现其抽象方法。
    这个基类定义了策略的标准接口，确保所有策略都能被统一管理和调用。
    """
    
    def __init__(self, config: StrategyConfig):
        """
        初始化策略
        
        Args:
            config: 策略配置对象
        """
        self.config = config                    # 策略配置
        self.name = config.name                 # 策略名称
        self.symbol = config.symbol             # 交易对
        self.timeframe = config.timeframe       # 时间周期
        self.parameters = config.parameters     # 策略参数
        
        # 策略状态
        self.is_initialized = False             # 是否已初始化
        self.last_signal_time = None           # 最后一次信号时间
        self.position = 0.0                    # 当前仓位（正数为多仓，负数为空仓）
        self.unrealized_pnl = 0.0              # 未实现盈亏
        self.realized_pnl = 0.0                # 已实现盈亏
        
        # 策略历史数据
        self.signal_history: List[TradeSignal] = []     # 信号历史
        self.order_history: List[OrderFillEvent] = []   # 订单历史
        self.performance_metrics: Dict[str, float] = {} # 绩效指标
        
        # 内部状态
        self._last_data: Optional[pd.DataFrame] = None  # 最后一次接收的数据
        
    @abstractmethod
    def initialize(self) -> None:
        """
        策略初始化方法（抽象方法）
        
        在策略开始运行前调用，用于初始化技术指标、状态变量等。
        子类必须实现这个方法。
        """
        pass
    
    @abstractmethod
    def on_data(self, data: pd.DataFrame) -> List[TradeSignal]:
        """
        数据处理方法（抽象方法）
        
        当新的市场数据到达时调用，策略在这里分析数据并生成交易信号。
        
        Args:
            data: 包含OHLCV数据的DataFrame，列名包括：
                 - timestamp: 时间戳
                 - open: 开盘价
                 - high: 最高价
                 - low: 最低价
                 - close: 收盘价
                 - volume: 成交量
                 
        Returns:
            List[TradeSignal]: 生成的交易信号列表
        """
        pass
    
    def on_order_fill(self, order_event: OrderFillEvent) -> Optional[List[TradeSignal]]:
        """
        订单成交处理方法（可选实现）
        
        当订单成交时调用，策略可以在这里处理仓位变化、更新状态等。
        
        Args:
            order_event: 订单成交事件
            
        Returns:
            Optional[List[TradeSignal]]: 可选的额外交易信号（如止损止盈调整）
        """
        # 更新仓位
        if order_event.side == "BUY":
            self.position += order_event.quantity
        elif order_event.side == "SELL":
            self.position -= order_event.quantity
            
        # 记录订单历史
        self.order_history.append(order_event)
        
        # 默认不返回额外信号
        return None
    
    def on_error(self, error: Exception) -> None:
        """
        错误处理方法（可选实现）
        
        当策略运行过程中发生错误时调用。
        
        Args:
            error: 发生的异常
        """
        print(f"策略 {self.name} 发生错误: {error}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        获取策略绩效指标
        
        Returns:
            Dict[str, float]: 包含各种绩效指标的字典
        """
        if not self.order_history:
            return {}
            
        # 计算基本绩效指标
        total_trades = len(self.order_history)
        total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # 计算胜率（简化版）
        profitable_trades = sum(1 for order in self.order_history 
                              if hasattr(order, 'pnl') and order.pnl > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        self.performance_metrics.update({
            'total_trades': total_trades,
            'total_pnl': total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'win_rate': win_rate,
            'current_position': self.position,
        })
        
        return self.performance_metrics
    
    def reset(self) -> None:
        """
        重置策略状态
        
        清除所有历史数据和状态，将策略恢复到初始状态。
        """
        self.is_initialized = False
        self.last_signal_time = None
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.signal_history.clear()
        self.order_history.clear()
        self.performance_metrics.clear()
        self._last_data = None
    
    def validate_signal(self, signal: TradeSignal) -> bool:
        """
        验证交易信号的有效性
        
        Args:
            signal: 要验证的交易信号
            
        Returns:
            bool: 信号是否有效
        """
        # 基本验证
        if not signal.symbol:
            return False
            
        if signal.signal_type not in SignalType:
            return False
            
        # 价格验证
        if signal.price is not None and signal.price <= 0:
            return False
            
        # 数量验证
        if signal.quantity is not None and signal.quantity <= 0:
            return False
            
        if signal.quantity_percent is not None:
            if not (0 <= signal.quantity_percent <= 1):
                return False
                
        # 置信度验证
        if not (0 <= signal.confidence <= 1):
            return False
            
        return True
    
    def create_signal(self, 
                     signal_type: SignalType,
                     price: Optional[float] = None,
                     quantity: Optional[float] = None,
                     quantity_percent: Optional[float] = None,
                     order_type: OrderType = OrderType.MARKET,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None,
                     confidence: float = 1.0,
                     metadata: Optional[Dict[str, Any]] = None) -> TradeSignal:
        """
        创建交易信号的便捷方法
        
        Args:
            signal_type: 信号类型
            price: 价格
            quantity: 数量
            quantity_percent: 数量百分比
            order_type: 订单类型
            stop_loss: 止损价格
            take_profit: 止盈价格
            confidence: 置信度
            metadata: 额外数据
            
        Returns:
            TradeSignal: 创建的交易信号
        """
        return TradeSignal(
            symbol=self.symbol,
            signal_type=signal_type,
            timestamp=datetime.now(),
            price=price,
            quantity=quantity,
            quantity_percent=quantity_percent,
            order_type=order_type,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=self.name,
            confidence=confidence,
            metadata=metadata or {}
        )
    
    def __repr__(self) -> str:
        """策略对象的字符串表示"""
        return (f"Strategy(name='{self.name}', symbol='{self.symbol}', "
                f"timeframe='{self.timeframe}', position={self.position})")