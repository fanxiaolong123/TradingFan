"""
均值回归策略示例

这是一个基于移动平均线的简单均值回归策略示例，
展示了如何继承Strategy基类并实现具体的交易逻辑。
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime

from .base import Strategy, StrategyConfig, TradeSignal, SignalType, OrderType


class MeanReversionStrategy(Strategy):
    """
    均值回归策略
    
    基于移动平均线的均值回归策略：
    - 当价格偏离移动平均线超过一定阈值时，认为会回归均值
    - 价格低于均线-阈值时买入
    - 价格高于均线+阈值时卖出
    """
    
    def __init__(self, config: StrategyConfig):
        """
        初始化均值回归策略
        
        Args:
            config: 策略配置，parameters中应包含：
                   - ma_period: 移动平均线周期，默认20
                   - deviation_threshold: 偏离阈值（百分比），默认0.02（2%）
                   - min_volume: 最小成交量要求，默认1000
        """
        super().__init__(config)
        
        # 策略参数
        self.ma_period = self.parameters.get('ma_period', 20)           # 移动平均线周期
        self.deviation_threshold = self.parameters.get('deviation_threshold', 0.02)  # 偏离阈值
        self.min_volume = self.parameters.get('min_volume', 1000)       # 最小成交量
        
        # 技术指标数据
        self.ma_values: List[float] = []                # 移动平均线值
        self.price_history: List[float] = []            # 价格历史
        self.volume_history: List[float] = []           # 成交量历史
        
        # 策略状态
        self.last_signal_type = SignalType.HOLD         # 最后信号类型
        self.entry_price: Optional[float] = None        # 入场价格
        self.signal_cooldown_bars = 3                   # 信号冷却期（K线数量）
        self.bars_since_last_signal = 0                 # 距离上次信号的K线数量
        
    def initialize(self) -> None:
        """初始化策略"""
        print(f"初始化均值回归策略: {self.name}")
        print(f"交易对: {self.symbol}")
        print(f"时间周期: {self.timeframe}")
        print(f"移动平均线周期: {self.ma_period}")
        print(f"偏离阈值: {self.deviation_threshold:.2%}")
        print(f"最小成交量: {self.min_volume}")
        
        # 清空历史数据
        self.ma_values.clear()
        self.price_history.clear()
        self.volume_history.clear()
        
        # 重置状态
        self.last_signal_type = SignalType.HOLD
        self.entry_price = None
        self.bars_since_last_signal = 0
        
        self.is_initialized = True
        print("策略初始化完成")
    
    def on_data(self, data: pd.DataFrame) -> List[TradeSignal]:
        """
        处理新的市场数据
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[TradeSignal]: 生成的交易信号
        """
        if data.empty:
            return []
            
        # 保存最新数据
        self._last_data = data
        
        # 获取最新的价格和成交量
        latest_row = data.iloc[-1]
        current_price = latest_row['close']
        current_volume = latest_row['volume']
        current_time = latest_row['timestamp'] if 'timestamp' in latest_row else datetime.now()
        
        # 更新历史数据
        self.price_history.append(current_price)
        self.volume_history.append(current_volume)
        
        # 保持历史数据长度不超过需要的长度
        max_history_length = max(self.ma_period * 2, 100)
        if len(self.price_history) > max_history_length:
            self.price_history = self.price_history[-max_history_length:]
            self.volume_history = self.volume_history[-max_history_length:]
        
        # 计算移动平均线
        ma_value = self._calculate_moving_average()
        if ma_value is None:
            return []  # 数据不足，无法计算指标
            
        self.ma_values.append(ma_value)
        if len(self.ma_values) > max_history_length:
            self.ma_values = self.ma_values[-max_history_length:]
        
        # 更新冷却期计数
        self.bars_since_last_signal += 1
        
        # 生成交易信号
        signals = self._generate_signals(current_price, ma_value, current_volume, current_time)
        
        # 记录信号历史
        for signal in signals:
            if self.validate_signal(signal):
                self.signal_history.append(signal)
                self.last_signal_time = signal.timestamp
                self.last_signal_type = signal.signal_type
                self.bars_since_last_signal = 0  # 重置冷却期
        
        return signals
    
    def _calculate_moving_average(self) -> Optional[float]:
        """
        计算移动平均线
        
        Returns:
            Optional[float]: 移动平均线值，如果数据不足则返回None
        """
        if len(self.price_history) < self.ma_period:
            return None
            
        # 计算简单移动平均线
        recent_prices = self.price_history[-self.ma_period:]
        return sum(recent_prices) / len(recent_prices)
    
    def _generate_signals(self, 
                         current_price: float, 
                         ma_value: float, 
                         current_volume: float,
                         timestamp: datetime) -> List[TradeSignal]:
        """
        根据当前市场状态生成交易信号
        
        Args:
            current_price: 当前价格
            ma_value: 移动平均线值
            current_volume: 当前成交量
            timestamp: 当前时间
            
        Returns:
            List[TradeSignal]: 生成的信号列表
        """
        signals = []
        
        # 检查成交量是否满足要求
        if current_volume < self.min_volume:
            return signals
        
        # 检查是否在冷却期内
        if self.bars_since_last_signal < self.signal_cooldown_bars:
            return signals
        
        # 计算价格偏离度
        deviation = (current_price - ma_value) / ma_value
        upper_threshold = self.deviation_threshold
        lower_threshold = -self.deviation_threshold
        
        # 生成买入信号：价格显著低于均线且当前无多仓
        if (deviation <= lower_threshold and 
            self.position <= 0 and 
            self.last_signal_type != SignalType.BUY):
            
            # 计算止损和止盈价格
            stop_loss_price = current_price * (1 - self.config.stop_loss_percent)
            take_profit_price = current_price * (1 + self.config.take_profit_percent)
            
            signal = self.create_signal(
                signal_type=SignalType.BUY,
                price=None,  # 市价买入
                quantity_percent=self.config.max_position_percent,
                order_type=OrderType.MARKET,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
                confidence=min(abs(deviation) / upper_threshold, 1.0),  # 偏离越大置信度越高
                metadata={
                    'ma_value': ma_value,
                    'deviation': deviation,
                    'volume': current_volume,
                    'strategy_reason': '价格显著低于均线，预期回归'
                }
            )
            signals.append(signal)
            self.entry_price = current_price
            
        # 生成卖出信号：价格显著高于均线且当前有多仓
        elif (deviation >= upper_threshold and 
              self.position > 0 and 
              self.last_signal_type != SignalType.SELL):
            
            signal = self.create_signal(
                signal_type=SignalType.SELL,
                price=None,  # 市价卖出
                quantity_percent=1.0,  # 全部平仓
                order_type=OrderType.MARKET,
                confidence=min(deviation / upper_threshold, 1.0),
                metadata={
                    'ma_value': ma_value,
                    'deviation': deviation,
                    'volume': current_volume,
                    'entry_price': self.entry_price,
                    'profit_pct': ((current_price - self.entry_price) / self.entry_price) if self.entry_price else 0,
                    'strategy_reason': '价格显著高于均线，预期回归'
                }
            )
            signals.append(signal)
            
        # 生成平仓信号：价格回到均线附近且有持仓
        elif (abs(deviation) <= upper_threshold * 0.3 and  # 回到均线附近
              self.position != 0 and 
              self.entry_price is not None):
            
            # 计算当前盈亏
            if self.position > 0:  # 多仓
                profit_pct = (current_price - self.entry_price) / self.entry_price
                if profit_pct > 0.005:  # 盈利超过0.5%才平仓
                    signal = self.create_signal(
                        signal_type=SignalType.SELL,
                        price=None,
                        quantity_percent=1.0,
                        order_type=OrderType.MARKET,
                        confidence=0.8,
                        metadata={
                            'ma_value': ma_value,
                            'deviation': deviation,
                            'profit_pct': profit_pct,
                            'strategy_reason': '价格回归均线，获利了结'
                        }
                    )
                    signals.append(signal)
        
        return signals
    
    def on_order_fill(self, order_event) -> Optional[List[TradeSignal]]:
        """
        处理订单成交事件
        
        Args:
            order_event: 订单成交事件
            
        Returns:
            Optional[List[TradeSignal]]: 可能的额外信号
        """
        # 调用父类方法更新基本状态
        additional_signals = super().on_order_fill(order_event)
        
        # 更新策略特定状态
        if order_event.side == "BUY":
            self.entry_price = order_event.price
            print(f"均值回归策略买入成交: {order_event.quantity:.6f} @ {order_event.price:.2f}")
        elif order_event.side == "SELL":
            if self.entry_price:
                profit = (order_event.price - self.entry_price) / self.entry_price
                print(f"均值回归策略卖出成交: {order_event.quantity:.6f} @ {order_event.price:.2f}, "
                      f"收益率: {profit:.2%}")
            self.entry_price = None
        
        return additional_signals
    
    def get_performance_metrics(self) -> dict:
        """
        获取策略绩效指标
        
        Returns:
            dict: 绩效指标字典
        """
        metrics = super().get_performance_metrics()
        
        # 添加策略特定指标
        if len(self.ma_values) > 0:
            current_ma = self.ma_values[-1]
            current_price = self.price_history[-1] if self.price_history else 0
            current_deviation = (current_price - current_ma) / current_ma if current_ma > 0 else 0
            
            metrics.update({
                'current_ma': current_ma,
                'current_deviation': current_deviation,
                'ma_period': self.ma_period,
                'deviation_threshold': self.deviation_threshold,
                'entry_price': self.entry_price,
                'bars_since_last_signal': self.bars_since_last_signal,
            })
        
        return metrics
    
    def __repr__(self) -> str:
        """策略对象的字符串表示"""
        return (f"MeanReversionStrategy(name='{self.name}', symbol='{self.symbol}', "
                f"ma_period={self.ma_period}, threshold={self.deviation_threshold:.2%})")