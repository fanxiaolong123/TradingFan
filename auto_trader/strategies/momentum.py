"""
动量策略模块

基于价格动量和趋势跟踪的交易策略：
- 使用RSI、MACD、移动平均线等技术指标
- 识别强势上涨/下跌趋势
- 动量确认和入场时机选择
- 动态止损和趋势跟踪
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .base import Strategy, StrategyConfig, TradeSignal, SignalType, OrderType


class MomentumStrategy(Strategy):
    """
    动量策略实现
    
    策略逻辑：
    1. 使用多个技术指标确认动量方向
    2. 在强势趋势中进行趋势跟踪
    3. 利用RSI防止过度买入/卖出
    4. 动态调整止损位
    """
    
    def __init__(self, config: StrategyConfig):
        """
        初始化动量策略
        
        Args:
            config: 策略配置对象
        """
        super().__init__(config)
        
        # 策略参数
        self.short_ma_period = self.parameters.get('short_ma_period', 10)      # 短期移动平均周期
        self.long_ma_period = self.parameters.get('long_ma_period', 30)        # 长期移动平均周期
        self.rsi_period = self.parameters.get('rsi_period', 14)                # RSI周期
        self.rsi_overbought = self.parameters.get('rsi_overbought', 70)        # RSI超买阈值
        self.rsi_oversold = self.parameters.get('rsi_oversold', 30)            # RSI超卖阈值
        self.momentum_period = self.parameters.get('momentum_period', 20)      # 动量计算周期
        self.momentum_threshold = self.parameters.get('momentum_threshold', 0.05)  # 动量阈值
        self.volume_threshold = self.parameters.get('volume_threshold', 1.2)   # 成交量确认倍数
        self.position_size = self.parameters.get('position_size', 0.2)         # 仓位大小
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.03)        # 止损百分比
        self.take_profit_pct = self.parameters.get('take_profit_pct', 0.08)    # 止盈百分比
        
        # MACD参数
        self.macd_fast = self.parameters.get('macd_fast', 12)                  # MACD快线周期
        self.macd_slow = self.parameters.get('macd_slow', 26)                  # MACD慢线周期
        self.macd_signal = self.parameters.get('macd_signal', 9)               # MACD信号线周期
        
        # 内部状态
        self.entry_price = 0.0                                                 # 入场价格
        self.trailing_stop = 0.0                                               # 追踪止损价格
        self.trend_direction = 0                                               # 趋势方向 (1: 上涨, -1: 下跌, 0: 震荡)
        self.signal_strength = 0.0                                             # 信号强度
        
        # 技术指标缓存
        self.indicators_cache = {}
        
        print(f"初始化动量策略: {self.name}")
        print(f"交易对: {self.symbol}")
        print(f"短期均线: {self.short_ma_period}, 长期均线: {self.long_ma_period}")
        print(f"RSI周期: {self.rsi_period}, 超买: {self.rsi_overbought}, 超卖: {self.rsi_oversold}")
        print(f"动量周期: {self.momentum_period}, 阈值: {self.momentum_threshold:.2%}")
    
    def initialize(self) -> None:
        """初始化策略"""
        self.is_initialized = True
        print("动量策略初始化完成")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算技术指标
        
        Args:
            data: OHLCV数据
            
        Returns:
            Dict: 包含各种技术指标的字典
        """
        indicators = {}
        
        # 移动平均线
        indicators['sma_short'] = data['close'].rolling(window=self.short_ma_period).mean()
        indicators['sma_long'] = data['close'].rolling(window=self.long_ma_period).mean()
        
        # RSI计算
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD计算
        exp1 = data['close'].ewm(span=self.macd_fast).mean()
        exp2 = data['close'].ewm(span=self.macd_slow).mean()
        indicators['macd'] = exp1 - exp2
        indicators['macd_signal'] = indicators['macd'].ewm(span=self.macd_signal).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # 动量计算
        indicators['momentum'] = data['close'].pct_change(periods=self.momentum_period)
        
        # 成交量移动平均
        if 'volume' in data.columns:
            indicators['volume_ma'] = data['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_ma']
        else:
            indicators['volume_ma'] = pd.Series([1.0] * len(data), index=data.index)
            indicators['volume_ratio'] = pd.Series([1.0] * len(data), index=data.index)
        
        # 波动率计算
        indicators['volatility'] = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
        
        # 价格通道 (布林带)
        indicators['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 2)
        indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 2)
        indicators['bb_position'] = (data['close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        return indicators
    
    def analyze_trend(self, indicators: Dict[str, Any]) -> int:
        """
        分析趋势方向
        
        Args:
            indicators: 技术指标字典
            
        Returns:
            int: 趋势方向 (1: 上涨, -1: 下跌, 0: 震荡)
        """
        latest_idx = len(indicators['sma_short']) - 1
        
        if latest_idx < self.long_ma_period:
            return 0
        
        # 移动平均线趋势
        sma_short = indicators['sma_short'].iloc[latest_idx]
        sma_long = indicators['sma_long'].iloc[latest_idx]
        
        if pd.isna(sma_short) or pd.isna(sma_long):
            return 0
        
        ma_trend = 1 if sma_short > sma_long else -1
        
        # MACD趋势
        macd = indicators['macd'].iloc[latest_idx]
        macd_signal = indicators['macd_signal'].iloc[latest_idx]
        macd_hist = indicators['macd_histogram'].iloc[latest_idx]
        
        if pd.isna(macd) or pd.isna(macd_signal):
            macd_trend = 0
        else:
            macd_trend = 1 if macd > macd_signal and macd_hist > 0 else -1
        
        # 动量趋势
        momentum = indicators['momentum'].iloc[latest_idx]
        momentum_trend = 0
        if not pd.isna(momentum):
            if momentum > self.momentum_threshold:
                momentum_trend = 1
            elif momentum < -self.momentum_threshold:
                momentum_trend = -1
        
        # 综合判断趋势
        trend_votes = [ma_trend, macd_trend, momentum_trend]
        positive_votes = sum(1 for vote in trend_votes if vote == 1)
        negative_votes = sum(1 for vote in trend_votes if vote == -1)
        
        if positive_votes >= 2:
            return 1
        elif negative_votes >= 2:
            return -1
        else:
            return 0
    
    def calculate_signal_strength(self, indicators: Dict[str, Any], trend: int) -> float:
        """
        计算信号强度
        
        Args:
            indicators: 技术指标字典
            trend: 趋势方向
            
        Returns:
            float: 信号强度 (0.0-1.0)
        """
        latest_idx = len(indicators['rsi']) - 1
        
        if latest_idx < self.long_ma_period:
            return 0.0
        
        strength = 0.0
        
        # RSI强度
        rsi = indicators['rsi'].iloc[latest_idx]
        if not pd.isna(rsi):
            if trend == 1:  # 上涨趋势
                # RSI不能过高
                rsi_strength = max(0, (self.rsi_overbought - rsi) / (self.rsi_overbought - 50))
            else:  # 下跌趋势
                # RSI不能过低
                rsi_strength = max(0, (rsi - self.rsi_oversold) / (50 - self.rsi_oversold))
            strength += rsi_strength * 0.3
        
        # MACD强度
        macd_hist = indicators['macd_histogram'].iloc[latest_idx]
        if not pd.isna(macd_hist):
            macd_strength = min(1.0, abs(macd_hist) / 0.01)  # 归一化到0-1
            strength += macd_strength * 0.3
        
        # 动量强度
        momentum = indicators['momentum'].iloc[latest_idx]
        if not pd.isna(momentum):
            momentum_strength = min(1.0, abs(momentum) / (self.momentum_threshold * 2))
            strength += momentum_strength * 0.2
        
        # 成交量确认
        volume_ratio = indicators['volume_ratio'].iloc[latest_idx]
        if not pd.isna(volume_ratio):
            volume_strength = min(1.0, volume_ratio / self.volume_threshold)
            strength += volume_strength * 0.2
        
        return min(1.0, strength)
    
    def should_enter_position(self, indicators: Dict[str, Any], current_price: float) -> Optional[SignalType]:
        """
        判断是否应该开仓
        
        Args:
            indicators: 技术指标字典
            current_price: 当前价格
            
        Returns:
            Optional[SignalType]: 入场信号类型或None
        """
        # 已有仓位时不开新仓
        if self.position != 0:
            return None
        
        # 分析趋势
        trend = self.analyze_trend(indicators)
        if trend == 0:
            return None
        
        # 计算信号强度
        strength = self.calculate_signal_strength(indicators, trend)
        if strength < 0.6:  # 信号强度阈值
            return None
        
        self.trend_direction = trend
        self.signal_strength = strength
        
        latest_idx = len(indicators['rsi']) - 1
        rsi = indicators['rsi'].iloc[latest_idx]
        
        # 做多条件
        if trend == 1:
            # RSI不能过度超买
            if not pd.isna(rsi) and rsi < self.rsi_overbought:
                # 布林带位置检查
                bb_position = indicators['bb_position'].iloc[latest_idx]
                if not pd.isna(bb_position) and bb_position < 0.8:  # 不在布林带顶部
                    return SignalType.BUY
        
        # 做空条件  
        elif trend == -1:
            # RSI不能过度超卖
            if not pd.isna(rsi) and rsi > self.rsi_oversold:
                # 布林带位置检查
                bb_position = indicators['bb_position'].iloc[latest_idx]
                if not pd.isna(bb_position) and bb_position > 0.2:  # 不在布林带底部
                    return SignalType.SELL
        
        return None
    
    def should_exit_position(self, indicators: Dict[str, Any], current_price: float) -> Optional[SignalType]:
        """
        判断是否应该平仓
        
        Args:
            indicators: 技术指标字典
            current_price: 当前价格
            
        Returns:
            Optional[SignalType]: 平仓信号类型或None
        """
        if self.position == 0:
            return None
        
        latest_idx = len(indicators['rsi']) - 1
        
        # 止损检查
        if self.entry_price > 0:
            if self.position > 0:  # 多仓
                # 固定止损
                if current_price <= self.entry_price * (1 - self.stop_loss_pct):
                    return SignalType.SELL
                
                # 追踪止损
                if self.trailing_stop > 0 and current_price <= self.trailing_stop:
                    return SignalType.SELL
                
                # 止盈
                if current_price >= self.entry_price * (1 + self.take_profit_pct):
                    return SignalType.SELL
                
            else:  # 空仓
                # 固定止损
                if current_price >= self.entry_price * (1 + self.stop_loss_pct):
                    return SignalType.BUY
                
                # 追踪止损
                if self.trailing_stop > 0 and current_price >= self.trailing_stop:
                    return SignalType.BUY
                
                # 止盈
                if current_price <= self.entry_price * (1 - self.take_profit_pct):
                    return SignalType.BUY
        
        # 趋势反转检查
        current_trend = self.analyze_trend(indicators)
        if current_trend != self.trend_direction and current_trend != 0:
            if self.position > 0:
                return SignalType.SELL
            else:
                return SignalType.BUY
        
        # RSI过度买卖检查
        rsi = indicators['rsi'].iloc[latest_idx]
        if not pd.isna(rsi):
            if self.position > 0 and rsi > self.rsi_overbought:
                return SignalType.SELL
            elif self.position < 0 and rsi < self.rsi_oversold:
                return SignalType.BUY
        
        # MACD背离检查
        macd_hist = indicators['macd_histogram'].iloc[latest_idx]
        if not pd.isna(macd_hist):
            if self.position > 0 and macd_hist < 0:
                return SignalType.SELL
            elif self.position < 0 and macd_hist > 0:
                return SignalType.BUY
        
        return None
    
    def update_trailing_stop(self, current_price: float) -> None:
        """
        更新追踪止损价格
        
        Args:
            current_price: 当前价格
        """
        if self.position == 0 or self.entry_price == 0:
            return
        
        trailing_distance = self.stop_loss_pct * 1.5  # 追踪距离
        
        if self.position > 0:  # 多仓
            new_trailing_stop = current_price * (1 - trailing_distance)
            if self.trailing_stop == 0 or new_trailing_stop > self.trailing_stop:
                self.trailing_stop = new_trailing_stop
        
        else:  # 空仓
            new_trailing_stop = current_price * (1 + trailing_distance)
            if self.trailing_stop == 0 or new_trailing_stop < self.trailing_stop:
                self.trailing_stop = new_trailing_stop
    
    def on_data(self, data: pd.DataFrame) -> List[TradeSignal]:
        """
        处理新数据
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[TradeSignal]: 生成的交易信号列表
        """
        if len(data) < self.long_ma_period:
            return []
        
        # 计算技术指标
        indicators = self.calculate_indicators(data)
        self.indicators_cache = indicators
        
        # 获取当前价格
        current_price = data['close'].iloc[-1]
        
        # 更新追踪止损
        self.update_trailing_stop(current_price)
        
        signals = []
        
        # 检查平仓信号
        exit_signal = self.should_exit_position(indicators, current_price)
        if exit_signal:
            if exit_signal == SignalType.BUY:
                signal_type = SignalType.CLOSE_SHORT if self.position < 0 else SignalType.BUY
            else:
                signal_type = SignalType.CLOSE_LONG if self.position > 0 else SignalType.SELL
            
            # 创建平仓信号
            signal = self.create_signal(
                signal_type=signal_type,
                quantity_percent=abs(self.position) / current_price if self.entry_price > 0 else self.position_size,
                confidence=0.8,
                metadata={
                    'action': 'exit',
                    'exit_reason': 'stop_loss_or_trend_change',
                    'entry_price': self.entry_price,
                    'current_price': current_price,
                    'pnl_percent': ((current_price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0
                }
            )
            signals.append(signal)
            
            print(f"[MOMENTUM] 平仓信号: {signal_type.value} @ {current_price:.2f}")
        
        # 检查开仓信号
        else:
            enter_signal = self.should_enter_position(indicators, current_price)
            if enter_signal:
                # 创建开仓信号
                signal = self.create_signal(
                    signal_type=enter_signal,
                    quantity_percent=self.position_size,
                    confidence=self.signal_strength,
                    metadata={
                        'action': 'enter',
                        'trend_direction': self.trend_direction,
                        'signal_strength': self.signal_strength,
                        'rsi': indicators['rsi'].iloc[-1] if not pd.isna(indicators['rsi'].iloc[-1]) else 0,
                        'macd_histogram': indicators['macd_histogram'].iloc[-1] if not pd.isna(indicators['macd_histogram'].iloc[-1]) else 0,
                        'momentum': indicators['momentum'].iloc[-1] if not pd.isna(indicators['momentum'].iloc[-1]) else 0
                    }
                )
                signals.append(signal)
                
                print(f"[MOMENTUM] 开仓信号: {enter_signal.value} @ {current_price:.2f}, 强度: {self.signal_strength:.2f}")
        
        return signals
    
    def on_order_fill(self, order_event) -> Optional[List[TradeSignal]]:
        """
        订单成交事件处理
        
        Args:
            order_event: 订单成交事件
            
        Returns:
            Optional[List[TradeSignal]]: 可选的额外交易信号
        """
        super().on_order_fill(order_event)
        
        # 更新入场价格和追踪止损
        if order_event.side == "BUY":
            if self.position > 0:  # 新开多仓或加仓
                self.entry_price = order_event.price
                self.trailing_stop = 0  # 重置追踪止损
                print(f"[MOMENTUM] 多仓开仓: {self.position:.4f} @ {self.entry_price:.2f}")
        
        elif order_event.side == "SELL":
            if self.position < 0:  # 新开空仓或加仓
                self.entry_price = order_event.price
                self.trailing_stop = 0  # 重置追踪止损
                print(f"[MOMENTUM] 空仓开仓: {self.position:.4f} @ {self.entry_price:.2f}")
            
            elif self.position == 0:  # 平仓
                print(f"[MOMENTUM] 平仓完成，PnL: {self.realized_pnl:.2f}")
                self.entry_price = 0
                self.trailing_stop = 0
                self.trend_direction = 0
        
        return None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            Dict[str, Any]: 策略状态信息
        """
        info = {
            'strategy_type': 'Momentum',
            'position': self.position,
            'entry_price': self.entry_price,
            'trailing_stop': self.trailing_stop,
            'trend_direction': self.trend_direction,
            'signal_strength': self.signal_strength,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'parameters': {
                'short_ma_period': self.short_ma_period,
                'long_ma_period': self.long_ma_period,
                'rsi_period': self.rsi_period,
                'momentum_period': self.momentum_period,
                'position_size': self.position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }
        }
        
        # 添加最新技术指标
        if self.indicators_cache:
            latest_idx = -1
            info['indicators'] = {
                'rsi': self.indicators_cache.get('rsi', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('rsi', [])) > 0 else None,
                'macd': self.indicators_cache.get('macd', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('macd', [])) > 0 else None,
                'momentum': self.indicators_cache.get('momentum', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('momentum', [])) > 0 else None,
                'volume_ratio': self.indicators_cache.get('volume_ratio', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('volume_ratio', [])) > 0 else None
            }
        
        return info