"""
突破策略模块

基于支撑阻力位突破的交易策略：
- 识别关键支撑阻力位
- 监控价格突破行为
- 成交量确认突破有效性
- 假突破过滤机制
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .base import Strategy, StrategyConfig, TradeSignal, SignalType, OrderType


class BreakoutStrategy(Strategy):
    """
    突破策略实现
    
    策略逻辑：
    1. 识别重要的支撑阻力位
    2. 监控价格突破关键位置
    3. 使用成交量确认突破的有效性
    4. 设置假突破过滤条件
    5. 动态调整止损和止盈
    """
    
    def __init__(self, config: StrategyConfig):
        """
        初始化突破策略
        
        Args:
            config: 策略配置对象
        """
        super().__init__(config)
        
        # 策略参数
        self.lookback_period = self.parameters.get('lookback_period', 50)      # 支撑阻力位识别回望周期
        self.breakout_threshold = self.parameters.get('breakout_threshold', 0.01)  # 突破阈值百分比
        self.volume_confirm_multiplier = self.parameters.get('volume_confirm_multiplier', 1.5)  # 成交量确认倍数
        self.position_size = self.parameters.get('position_size', 0.3)         # 仓位大小
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.02)        # 止损百分比
        self.take_profit_pct = self.parameters.get('take_profit_pct', 0.06)    # 止盈百分比
        self.false_breakout_timeout = self.parameters.get('false_breakout_timeout', 5)  # 假突破超时周期
        self.min_consolidation_periods = self.parameters.get('min_consolidation_periods', 10)  # 最小盘整周期
        self.support_resistance_strength = self.parameters.get('support_resistance_strength', 3)  # 支撑阻力位强度要求
        
        # ATR参数 (用于动态止损)
        self.atr_period = self.parameters.get('atr_period', 14)               # ATR周期
        self.atr_multiplier = self.parameters.get('atr_multiplier', 2.0)      # ATR止损倍数
        
        # 内部状态
        self.support_levels = []                                               # 支撑位列表
        self.resistance_levels = []                                            # 阻力位列表
        self.last_breakout_time = None                                         # 最后突破时间
        self.breakout_price = 0.0                                             # 突破价格
        self.entry_price = 0.0                                                # 入场价格
        self.breakout_direction = 0                                           # 突破方向 (1: 向上, -1: 向下)
        self.confirmation_candles = 0                                         # 确认K线数量
        
        # 技术指标缓存
        self.indicators_cache = {}
        
        print(f"初始化突破策略: {self.name}")
        print(f"交易对: {self.symbol}")
        print(f"回望周期: {self.lookback_period}, 突破阈值: {self.breakout_threshold:.2%}")
        print(f"成交量确认: {self.volume_confirm_multiplier}x, 仓位大小: {self.position_size:.1%}")
    
    def initialize(self) -> None:
        """初始化策略"""
        self.is_initialized = True
        print("突破策略初始化完成")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算技术指标
        
        Args:
            data: OHLCV数据
            
        Returns:
            Dict: 包含各种技术指标的字典
        """
        indicators = {}
        
        # ATR计算 (Average True Range)
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # 成交量移动平均
        if 'volume' in data.columns:
            indicators['volume_ma'] = data['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_ma']
        else:
            indicators['volume_ma'] = pd.Series([1.0] * len(data), index=data.index)
            indicators['volume_ratio'] = pd.Series([1.0] * len(data), index=data.index)
        
        # 波动率指标
        indicators['volatility'] = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
        
        # 价格范围
        indicators['price_range'] = (data['high'] - data['low']) / data['close']
        
        # 布林带
        bb_middle = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        indicators['bb_upper'] = bb_middle + (bb_std * 2)
        indicators['bb_lower'] = bb_middle - (bb_std * 2)
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / bb_middle
        
        # 价格在布林带中的位置
        indicators['bb_position'] = (data['close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        return indicators
    
    def identify_support_resistance(self, data: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """
        识别支撑阻力位
        
        Args:
            data: OHLCV数据
            
        Returns:
            Tuple[List[float], List[float]]: (支撑位列表, 阻力位列表)
        """
        if len(data) < self.lookback_period:
            return [], []
        
        # 使用滑动窗口寻找局部极值
        window = 5  # 局部极值窗口大小
        
        # 寻找局部高点 (阻力位)
        high_peaks = []
        for i in range(window, len(data) - window):
            if all(data['high'].iloc[i] >= data['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(data['high'].iloc[i] >= data['high'].iloc[i+j] for j in range(1, window+1)):
                high_peaks.append((i, data['high'].iloc[i]))
        
        # 寻找局部低点 (支撑位)
        low_troughs = []
        for i in range(window, len(data) - window):
            if all(data['low'].iloc[i] <= data['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(data['low'].iloc[i] <= data['low'].iloc[i+j] for j in range(1, window+1)):
                low_troughs.append((i, data['low'].iloc[i]))
        
        # 过滤和聚合相近的关键位
        resistance_levels = self._cluster_levels([price for _, price in high_peaks])
        support_levels = self._cluster_levels([price for _, price in low_troughs])
        
        # 验证支撑阻力位的有效性
        current_price = data['close'].iloc[-1]
        resistance_levels = [level for level in resistance_levels if self._validate_level(data, level, 'resistance')]
        support_levels = [level for level in support_levels if self._validate_level(data, level, 'support')]
        
        # 只保留相对于当前价格有意义的位
        resistance_levels = [level for level in resistance_levels if level > current_price]
        support_levels = [level for level in support_levels if level < current_price]
        
        # 按距离当前价格排序
        resistance_levels.sort()
        support_levels.sort(reverse=True)
        
        # 只保留最近的几个关键位
        resistance_levels = resistance_levels[:5]
        support_levels = support_levels[:5]
        
        return support_levels, resistance_levels
    
    def _cluster_levels(self, levels: List[float], threshold: float = 0.01) -> List[float]:
        """
        聚合相近的价格位
        
        Args:
            levels: 价格位列表
            threshold: 聚合阈值
            
        Returns:
            List[float]: 聚合后的价格位列表
        """
        if not levels:
            return []
        
        levels = sorted(levels)
        clustered = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - clustered[-1]) / clustered[-1] > threshold:
                clustered.append(level)
            else:
                # 合并到现有cluster
                clustered[-1] = (clustered[-1] + level) / 2
        
        return clustered
    
    def _validate_level(self, data: pd.DataFrame, level: float, level_type: str) -> bool:
        """
        验证支撑阻力位的有效性
        
        Args:
            data: OHLCV数据
            level: 价格位
            level_type: 类型 ('support' 或 'resistance')
            
        Returns:
            bool: 是否有效
        """
        # 计算价格触及该位的次数
        tolerance = level * 0.005  # 0.5% 容忍度
        
        if level_type == 'resistance':
            touches = sum(1 for high in data['high'].iloc[-self.lookback_period:] 
                         if abs(high - level) <= tolerance)
        else:  # support
            touches = sum(1 for low in data['low'].iloc[-self.lookback_period:] 
                         if abs(low - level) <= tolerance)
        
        return touches >= self.support_resistance_strength
    
    def detect_consolidation(self, data: pd.DataFrame) -> bool:
        """
        检测是否处于盘整状态
        
        Args:
            data: OHLCV数据
            
        Returns:
            bool: 是否处于盘整状态
        """
        if len(data) < self.min_consolidation_periods:
            return False
        
        recent_data = data.iloc[-self.min_consolidation_periods:]
        
        # 计算价格范围
        high_max = recent_data['high'].max()
        low_min = recent_data['low'].min()
        price_range = (high_max - low_min) / recent_data['close'].mean()
        
        # 计算波动率
        returns = recent_data['close'].pct_change()
        volatility = returns.std()
        
        # 盘整条件：价格范围小且波动率低
        is_consolidating = price_range < 0.05 and volatility < 0.02
        
        return is_consolidating
    
    def detect_breakout(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[Tuple[str, float, float]]:
        """
        检测突破
        
        Args:
            data: OHLCV数据
            indicators: 技术指标
            
        Returns:
            Optional[Tuple[str, float, float]]: (突破方向, 突破价格, 目标价格) 或 None
        """
        if len(data) < 2:
            return None
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        current_volume = data['volume'].iloc[-1] if 'volume' in data.columns else 1.0
        
        # 成交量确认
        volume_ma = indicators['volume_ma'].iloc[-1]
        volume_confirmed = current_volume >= volume_ma * self.volume_confirm_multiplier
        
        # 检查向上突破阻力位
        for resistance in self.resistance_levels:
            if (prev_price <= resistance * (1 - self.breakout_threshold/2) and 
                current_price > resistance * (1 + self.breakout_threshold) and
                volume_confirmed):
                
                # 计算目标价格 (阻力位突破后的目标)
                target_price = resistance * (1 + self.take_profit_pct)
                return ('up', resistance, target_price)
        
        # 检查向下突破支撑位
        for support in self.support_levels:
            if (prev_price >= support * (1 + self.breakout_threshold/2) and 
                current_price < support * (1 - self.breakout_threshold) and
                volume_confirmed):
                
                # 计算目标价格 (支撑位突破后的目标)
                target_price = support * (1 - self.take_profit_pct)
                return ('down', support, target_price)
        
        return None
    
    def is_false_breakout(self, data: pd.DataFrame, breakout_price: float, direction: str) -> bool:
        """
        检测是否为假突破
        
        Args:
            data: OHLCV数据
            breakout_price: 突破价格
            direction: 突破方向
            
        Returns:
            bool: 是否为假突破
        """
        if not self.last_breakout_time:
            return False
        
        # 检查突破后的价格行为
        current_price = data['close'].iloc[-1]
        
        if direction == 'up':
            # 向上突破后价格又跌回突破位以下
            return current_price < breakout_price * (1 - self.breakout_threshold/2)
        else:
            # 向下突破后价格又涨回突破位以上
            return current_price > breakout_price * (1 + self.breakout_threshold/2)
    
    def should_enter_position(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[SignalType]:
        """
        判断是否应该开仓
        
        Args:
            data: OHLCV数据
            indicators: 技术指标
            
        Returns:
            Optional[SignalType]: 入场信号类型或None
        """
        # 已有仓位时不开新仓
        if self.position != 0:
            return None
        
        # 更新支撑阻力位
        self.support_levels, self.resistance_levels = self.identify_support_resistance(data)
        
        # 检测盘整状态（可选，某些情况下盘整后的突破更有效）
        is_consolidating = self.detect_consolidation(data)
        
        # 检测突破
        breakout_info = self.detect_breakout(data, indicators)
        if not breakout_info:
            return None
        
        direction, breakout_price, target_price = breakout_info
        
        # 记录突破信息
        self.last_breakout_time = datetime.now()
        self.breakout_price = breakout_price
        self.breakout_direction = 1 if direction == 'up' else -1
        self.confirmation_candles = 0
        
        print(f"[BREAKOUT] 检测到{direction}突破: {breakout_price:.2f} -> 目标: {target_price:.2f}")
        
        if direction == 'up':
            return SignalType.BUY
        else:
            return SignalType.SELL
    
    def should_exit_position(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> Optional[SignalType]:
        """
        判断是否应该平仓
        
        Args:
            data: OHLCV数据
            indicators: 技术指标
            
        Returns:
            Optional[SignalType]: 平仓信号类型或None
        """
        if self.position == 0:
            return None
        
        current_price = data['close'].iloc[-1]
        
        # 止损检查
        if self.entry_price > 0:
            atr = indicators['atr'].iloc[-1] if not pd.isna(indicators['atr'].iloc[-1]) else 0
            
            if self.position > 0:  # 多仓
                # 固定止损
                stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
                
                # ATR动态止损
                if atr > 0:
                    atr_stop_loss = self.entry_price - (atr * self.atr_multiplier)
                    stop_loss_price = max(stop_loss_price, atr_stop_loss)
                
                if current_price <= stop_loss_price:
                    print(f"[BREAKOUT] 多仓止损: {current_price:.2f} <= {stop_loss_price:.2f}")
                    return SignalType.SELL
                
                # 止盈
                if current_price >= self.entry_price * (1 + self.take_profit_pct):
                    print(f"[BREAKOUT] 多仓止盈: {current_price:.2f} >= {self.entry_price * (1 + self.take_profit_pct):.2f}")
                    return SignalType.SELL
            
            else:  # 空仓
                # 固定止损
                stop_loss_price = self.entry_price * (1 + self.stop_loss_pct)
                
                # ATR动态止损
                if atr > 0:
                    atr_stop_loss = self.entry_price + (atr * self.atr_multiplier)
                    stop_loss_price = min(stop_loss_price, atr_stop_loss)
                
                if current_price >= stop_loss_price:
                    print(f"[BREAKOUT] 空仓止损: {current_price:.2f} >= {stop_loss_price:.2f}")
                    return SignalType.BUY
                
                # 止盈
                if current_price <= self.entry_price * (1 - self.take_profit_pct):
                    print(f"[BREAKOUT] 空仓止盈: {current_price:.2f} <= {self.entry_price * (1 - self.take_profit_pct):.2f}")
                    return SignalType.BUY
        
        # 假突破检查
        if self.breakout_price > 0:
            direction = 'up' if self.breakout_direction == 1 else 'down'
            if self.is_false_breakout(data, self.breakout_price, direction):
                print(f"[BREAKOUT] 检测到假突破，平仓")
                return SignalType.SELL if self.position > 0 else SignalType.BUY
        
        # 反向突破检查
        new_breakout = self.detect_breakout(data, indicators)
        if new_breakout:
            new_direction, _, _ = new_breakout
            if ((new_direction == 'down' and self.position > 0) or 
                (new_direction == 'up' and self.position < 0)):
                print(f"[BREAKOUT] 反向突破，平仓")
                return SignalType.SELL if self.position > 0 else SignalType.BUY
        
        return None
    
    def on_data(self, data: pd.DataFrame) -> List[TradeSignal]:
        """
        处理新数据
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[TradeSignal]: 生成的交易信号列表
        """
        if len(data) < self.lookback_period:
            return []
        
        # 计算技术指标
        indicators = self.calculate_indicators(data)
        self.indicators_cache = indicators
        
        # 获取当前价格
        current_price = data['close'].iloc[-1]
        
        # 更新确认K线数量
        if self.last_breakout_time:
            self.confirmation_candles += 1
        
        signals = []
        
        # 检查平仓信号
        exit_signal = self.should_exit_position(data, indicators)
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
                    'exit_reason': 'stop_loss_or_false_breakout',
                    'entry_price': self.entry_price,
                    'current_price': current_price,
                    'breakout_price': self.breakout_price,
                    'pnl_percent': ((current_price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0
                }
            )
            signals.append(signal)
            
            print(f"[BREAKOUT] 平仓信号: {signal_type.value} @ {current_price:.2f}")
        
        # 检查开仓信号
        else:
            enter_signal = self.should_enter_position(data, indicators)
            if enter_signal:
                # 创建开仓信号
                signal = self.create_signal(
                    signal_type=enter_signal,
                    quantity_percent=self.position_size,
                    confidence=0.9,  # 突破信号通常置信度较高
                    metadata={
                        'action': 'enter',
                        'breakout_direction': self.breakout_direction,
                        'breakout_price': self.breakout_price,
                        'support_levels': self.support_levels,
                        'resistance_levels': self.resistance_levels,
                        'volume_ratio': indicators['volume_ratio'].iloc[-1] if not pd.isna(indicators['volume_ratio'].iloc[-1]) else 0,
                        'atr': indicators['atr'].iloc[-1] if not pd.isna(indicators['atr'].iloc[-1]) else 0
                    }
                )
                signals.append(signal)
                
                print(f"[BREAKOUT] 开仓信号: {enter_signal.value} @ {current_price:.2f}, 突破价: {self.breakout_price:.2f}")
        
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
        
        # 更新入场价格
        if order_event.side == "BUY":
            if self.position > 0:  # 新开多仓
                self.entry_price = order_event.price
                print(f"[BREAKOUT] 多仓开仓: {self.position:.4f} @ {self.entry_price:.2f}")
        
        elif order_event.side == "SELL":
            if self.position < 0:  # 新开空仓
                self.entry_price = order_event.price
                print(f"[BREAKOUT] 空仓开仓: {self.position:.4f} @ {self.entry_price:.2f}")
            
            elif self.position == 0:  # 平仓
                print(f"[BREAKOUT] 平仓完成，PnL: {self.realized_pnl:.2f}")
                self.entry_price = 0
                self.breakout_price = 0
                self.breakout_direction = 0
                self.last_breakout_time = None
                self.confirmation_candles = 0
        
        return None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            Dict[str, Any]: 策略状态信息
        """
        info = {
            'strategy_type': 'Breakout',
            'position': self.position,
            'entry_price': self.entry_price,
            'breakout_price': self.breakout_price,
            'breakout_direction': self.breakout_direction,
            'confirmation_candles': self.confirmation_candles,
            'support_levels': self.support_levels,
            'resistance_levels': self.resistance_levels,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'parameters': {
                'lookback_period': self.lookback_period,
                'breakout_threshold': self.breakout_threshold,
                'volume_confirm_multiplier': self.volume_confirm_multiplier,
                'position_size': self.position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }
        }
        
        # 添加最新技术指标
        if self.indicators_cache:
            latest_idx = -1
            info['indicators'] = {
                'atr': self.indicators_cache.get('atr', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('atr', [])) > 0 else None,
                'volume_ratio': self.indicators_cache.get('volume_ratio', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('volume_ratio', [])) > 0 else None,
                'bb_width': self.indicators_cache.get('bb_width', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('bb_width', [])) > 0 else None,
                'bb_position': self.indicators_cache.get('bb_position', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('bb_position', [])) > 0 else None
            }
        
        return info