"""
趋势跟踪策略模块

基于多重技术指标的趋势跟踪策略：
- 综合多个趋势指标确认趋势方向
- 使用Supertrend、Parabolic SAR等跟踪趋势
- 动态调整止损位置
- 趋势强度评估和风险管理
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .base import Strategy, StrategyConfig, TradeSignal, SignalType, OrderType


class TrendFollowingStrategy(Strategy):
    """
    趋势跟踪策略实现
    
    策略逻辑：
    1. 使用多个指标确认趋势方向和强度
    2. 在趋势确认后进入市场
    3. 使用多种止损方式保护利润
    4. 动态调整仓位大小
    """
    
    def __init__(self, config: StrategyConfig):
        """
        初始化趋势跟踪策略
        
        Args:
            config: 策略配置对象
        """
        super().__init__(config)
        
        # 策略参数
        self.ema_fast = self.parameters.get('ema_fast', 12)                    # 快速EMA周期
        self.ema_slow = self.parameters.get('ema_slow', 26)                    # 慢速EMA周期
        self.supertrend_period = self.parameters.get('supertrend_period', 10)  # Supertrend周期
        self.supertrend_multiplier = self.parameters.get('supertrend_multiplier', 3.0)  # Supertrend倍数
        self.adx_period = self.parameters.get('adx_period', 14)                # ADX周期
        self.adx_threshold = self.parameters.get('adx_threshold', 25)          # ADX趋势强度阈值
        self.psar_af = self.parameters.get('psar_af', 0.02)                   # Parabolic SAR加速因子
        self.psar_max_af = self.parameters.get('psar_max_af', 0.2)            # Parabolic SAR最大加速因子
        
        # 仓位和风险参数
        self.base_position_size = self.parameters.get('base_position_size', 0.2)  # 基础仓位大小
        self.max_position_size = self.parameters.get('max_position_size', 0.5)    # 最大仓位大小
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.03)           # 止损百分比
        self.trailing_stop_pct = self.parameters.get('trailing_stop_pct', 0.02)   # 追踪止损百分比
        
        # 确认参数
        self.trend_confirmation_candles = self.parameters.get('trend_confirmation_candles', 3)  # 趋势确认K线数
        self.min_trend_strength = self.parameters.get('min_trend_strength', 0.6)  # 最小趋势强度
        
        # 内部状态
        self.trend_direction = 0                                               # 趋势方向 (1: 上涨, -1: 下跌, 0: 无趋势)
        self.trend_strength = 0.0                                             # 趋势强度 (0-1)
        self.trend_start_time = None                                           # 趋势开始时间
        self.entry_price = 0.0                                                # 入场价格
        self.highest_price = 0.0                                              # 最高价格 (用于追踪止损)
        self.lowest_price = 0.0                                               # 最低价格 (用于追踪止损)
        self.dynamic_stop_loss = 0.0                                          # 动态止损价格
        self.position_size_multiplier = 1.0                                   # 仓位大小倍数
        
        # 技术指标缓存
        self.indicators_cache = {}
        
        print(f"初始化趋势跟踪策略: {self.name}")
        print(f"交易对: {self.symbol}")
        print(f"EMA: {self.ema_fast}/{self.ema_slow}, Supertrend: {self.supertrend_period}/{self.supertrend_multiplier}")
        print(f"ADX阈值: {self.adx_threshold}, 基础仓位: {self.base_position_size:.1%}")
    
    def initialize(self) -> None:
        """初始化策略"""
        self.is_initialized = True
        print("趋势跟踪策略初始化完成")
    
    def calculate_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算技术指标
        
        Args:
            data: OHLCV数据
            
        Returns:
            Dict: 包含各种技术指标的字典
        """
        indicators = {}
        
        # EMA指标
        indicators['ema_fast'] = data['close'].ewm(span=self.ema_fast).mean()
        indicators['ema_slow'] = data['close'].ewm(span=self.ema_slow).mean()
        indicators['ema_diff'] = indicators['ema_fast'] - indicators['ema_slow']
        
        # ATR计算 (用于Supertrend)
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        indicators['atr'] = true_range.rolling(window=self.supertrend_period).mean()
        
        # Supertrend计算
        indicators['supertrend'], indicators['supertrend_direction'] = self._calculate_supertrend(
            data, indicators['atr']
        )
        
        # ADX计算 (趋势强度指标)
        indicators['adx'], indicators['di_plus'], indicators['di_minus'] = self._calculate_adx(data)
        
        # Parabolic SAR计算
        indicators['psar'] = self._calculate_parabolic_sar(data)
        
        # MACD指标
        exp1 = data['close'].ewm(span=12).mean()
        exp2 = data['close'].ewm(span=26).mean()
        indicators['macd'] = exp1 - exp2
        indicators['macd_signal'] = indicators['macd'].ewm(span=9).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # 布林带
        bb_middle = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        indicators['bb_upper'] = bb_middle + (bb_std * 2)
        indicators['bb_lower'] = bb_middle - (bb_std * 2)
        indicators['bb_position'] = (data['close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # 成交量指标
        if 'volume' in data.columns:
            indicators['volume_ma'] = data['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_ma']
            # OBV (On Balance Volume)
            indicators['obv'] = self._calculate_obv(data)
        else:
            indicators['volume_ma'] = pd.Series([1.0] * len(data), index=data.index)
            indicators['volume_ratio'] = pd.Series([1.0] * len(data), index=data.index)
            indicators['obv'] = pd.Series([0.0] * len(data), index=data.index)
        
        return indicators
    
    def _calculate_supertrend(self, data: pd.DataFrame, atr: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """计算Supertrend指标"""
        hl2 = (data['high'] + data['low']) / 2
        
        # 计算基本上下轨
        upper_band = hl2 + (self.supertrend_multiplier * atr)
        lower_band = hl2 - (self.supertrend_multiplier * atr)
        
        # 计算最终上下轨
        final_upper_band = pd.Series(index=data.index, dtype=float)
        final_lower_band = pd.Series(index=data.index, dtype=float)
        supertrend = pd.Series(index=data.index, dtype=float)
        direction = pd.Series(index=data.index, dtype=int)
        
        for i in range(len(data)):
            if i == 0:
                final_upper_band.iloc[i] = upper_band.iloc[i]
                final_lower_band.iloc[i] = lower_band.iloc[i]
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                # 计算最终上轨
                if upper_band.iloc[i] < final_upper_band.iloc[i-1] or data['close'].iloc[i-1] > final_upper_band.iloc[i-1]:
                    final_upper_band.iloc[i] = upper_band.iloc[i]
                else:
                    final_upper_band.iloc[i] = final_upper_band.iloc[i-1]
                
                # 计算最终下轨
                if lower_band.iloc[i] > final_lower_band.iloc[i-1] or data['close'].iloc[i-1] < final_lower_band.iloc[i-1]:
                    final_lower_band.iloc[i] = lower_band.iloc[i]
                else:
                    final_lower_band.iloc[i] = final_lower_band.iloc[i-1]
                
                # 计算Supertrend
                if direction.iloc[i-1] == 1 and data['close'].iloc[i] <= final_lower_band.iloc[i]:
                    direction.iloc[i] = -1
                elif direction.iloc[i-1] == -1 and data['close'].iloc[i] >= final_upper_band.iloc[i]:
                    direction.iloc[i] = 1
                else:
                    direction.iloc[i] = direction.iloc[i-1]
                
                if direction.iloc[i] == 1:
                    supertrend.iloc[i] = final_lower_band.iloc[i]
                else:
                    supertrend.iloc[i] = final_upper_band.iloc[i]
        
        return supertrend, direction
    
    def _calculate_adx(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算ADX指标"""
        # 计算方向性移动
        up_move = data['high'] - data['high'].shift(1)
        down_move = data['low'].shift(1) - data['low']
        
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=data.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=data.index)
        
        # 计算真实范围
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # 平滑处理
        plus_dm_smooth = plus_dm.rolling(window=self.adx_period).mean()
        minus_dm_smooth = minus_dm.rolling(window=self.adx_period).mean()
        tr_smooth = true_range.rolling(window=self.adx_period).mean()
        
        # 计算DI+和DI-
        di_plus = 100 * (plus_dm_smooth / tr_smooth)
        di_minus = 100 * (minus_dm_smooth / tr_smooth)
        
        # 计算DX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # 计算ADX
        adx = dx.rolling(window=self.adx_period).mean()
        
        return adx, di_plus, di_minus
    
    def _calculate_parabolic_sar(self, data: pd.DataFrame) -> pd.Series:
        """计算Parabolic SAR指标"""
        psar = pd.Series(index=data.index, dtype=float)
        af = self.psar_af
        ep = 0.0
        trend = 1  # 1 for uptrend, -1 for downtrend
        
        for i in range(len(data)):
            if i == 0:
                psar.iloc[i] = data['low'].iloc[i]
                ep = data['high'].iloc[i]
            else:
                if trend == 1:  # 上升趋势
                    psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                    
                    # 确保SAR不高于前两个周期的最低价
                    if i >= 2:
                        psar.iloc[i] = min(psar.iloc[i], data['low'].iloc[i-1], data['low'].iloc[i-2])
                    else:
                        psar.iloc[i] = min(psar.iloc[i], data['low'].iloc[i-1])
                    
                    # 检查趋势反转
                    if data['low'].iloc[i] < psar.iloc[i]:
                        trend = -1
                        psar.iloc[i] = ep
                        ep = data['low'].iloc[i]
                        af = self.psar_af
                    else:
                        if data['high'].iloc[i] > ep:
                            ep = data['high'].iloc[i]
                            af = min(af + self.psar_af, self.psar_max_af)
                
                else:  # 下降趋势
                    psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                    
                    # 确保SAR不低于前两个周期的最高价
                    if i >= 2:
                        psar.iloc[i] = max(psar.iloc[i], data['high'].iloc[i-1], data['high'].iloc[i-2])
                    else:
                        psar.iloc[i] = max(psar.iloc[i], data['high'].iloc[i-1])
                    
                    # 检查趋势反转
                    if data['high'].iloc[i] > psar.iloc[i]:
                        trend = 1
                        psar.iloc[i] = ep
                        ep = data['high'].iloc[i]
                        af = self.psar_af
                    else:
                        if data['low'].iloc[i] < ep:
                            ep = data['low'].iloc[i]
                            af = min(af + self.psar_af, self.psar_max_af)
        
        return psar
    
    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """计算OBV (On Balance Volume)指标"""
        obv = pd.Series(index=data.index, dtype=float)
        obv.iloc[0] = data['volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - data['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def analyze_trend(self, indicators: Dict[str, Any]) -> Tuple[int, float]:
        """
        分析趋势方向和强度
        
        Args:
            indicators: 技术指标字典
            
        Returns:
            Tuple[int, float]: (趋势方向, 趋势强度)
        """
        latest_idx = -1
        trend_signals = []
        weights = []
        
        # EMA趋势
        ema_fast = indicators['ema_fast'].iloc[latest_idx]
        ema_slow = indicators['ema_slow'].iloc[latest_idx]
        if not (pd.isna(ema_fast) or pd.isna(ema_slow)):
            ema_signal = 1 if ema_fast > ema_slow else -1
            trend_signals.append(ema_signal)
            weights.append(0.25)
        
        # Supertrend趋势
        supertrend_dir = indicators['supertrend_direction'].iloc[latest_idx]
        if not pd.isna(supertrend_dir):
            trend_signals.append(int(supertrend_dir))
            weights.append(0.3)
        
        # MACD趋势
        macd = indicators['macd'].iloc[latest_idx]
        macd_signal = indicators['macd_signal'].iloc[latest_idx]
        if not (pd.isna(macd) or pd.isna(macd_signal)):
            macd_trend = 1 if macd > macd_signal else -1
            trend_signals.append(macd_trend)
            weights.append(0.2)
        
        # Parabolic SAR趋势
        psar = indicators['psar'].iloc[latest_idx]
        current_price = indicators['ema_fast'].iloc[latest_idx]  # 使用EMA作为当前价格参考
        if not (pd.isna(psar) or pd.isna(current_price)):
            psar_trend = 1 if current_price > psar else -1
            trend_signals.append(psar_trend)
            weights.append(0.25)
        
        if not trend_signals:
            return 0, 0.0
        
        # 计算加权趋势方向
        weighted_sum = sum(signal * weight for signal, weight in zip(trend_signals, weights))
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0, 0.0
        
        weighted_average = weighted_sum / total_weight
        
        # 确定趋势方向
        if weighted_average > 0.3:
            trend_direction = 1
        elif weighted_average < -0.3:
            trend_direction = -1
        else:
            trend_direction = 0
        
        # 计算趋势强度
        trend_strength = abs(weighted_average)
        
        # ADX强度确认
        adx = indicators['adx'].iloc[latest_idx]
        if not pd.isna(adx):
            adx_strength = min(1.0, adx / 50)  # 归一化ADX到0-1
            trend_strength = trend_strength * adx_strength
        
        return trend_direction, trend_strength
    
    def calculate_dynamic_position_size(self, indicators: Dict[str, Any]) -> float:
        """
        计算动态仓位大小
        
        Args:
            indicators: 技术指标字典
            
        Returns:
            float: 仓位大小倍数
        """
        # 基于趋势强度调整仓位
        strength_multiplier = min(2.0, max(0.5, self.trend_strength * 2))
        
        # 基于ADX调整
        adx = indicators['adx'].iloc[-1]
        if not pd.isna(adx) and adx > self.adx_threshold:
            adx_multiplier = min(1.5, adx / 50)
        else:
            adx_multiplier = 0.8
        
        # 基于波动率调整
        atr = indicators['atr'].iloc[-1]
        if not pd.isna(atr):
            # 低波动率时可以加大仓位，高波动率时减小仓位
            volatility_multiplier = max(0.5, min(1.5, 1 / (atr / 100)))
        else:
            volatility_multiplier = 1.0
        
        total_multiplier = strength_multiplier * adx_multiplier * volatility_multiplier
        return min(self.max_position_size / self.base_position_size, total_multiplier)
    
    def update_dynamic_stop_loss(self, data: pd.DataFrame, indicators: Dict[str, Any]) -> None:
        """
        更新动态止损
        
        Args:
            data: OHLCV数据
            indicators: 技术指标
        """
        if self.position == 0:
            return
        
        current_price = data['close'].iloc[-1]
        
        # 更新最高/最低价
        if self.position > 0:
            self.highest_price = max(self.highest_price, current_price)
        else:
            self.lowest_price = min(self.lowest_price, current_price)
        
        # Supertrend止损
        supertrend = indicators['supertrend'].iloc[-1]
        
        # Parabolic SAR止损
        psar = indicators['psar'].iloc[-1]
        
        # ATR动态止损
        atr = indicators['atr'].iloc[-1]
        
        if self.position > 0:  # 多仓
            # 使用多种止损中的最高者
            stop_candidates = []
            
            # 固定百分比止损
            if self.entry_price > 0:
                stop_candidates.append(self.entry_price * (1 - self.stop_loss_pct))
            
            # 追踪止损
            if self.highest_price > 0:
                stop_candidates.append(self.highest_price * (1 - self.trailing_stop_pct))
            
            # Supertrend止损
            if not pd.isna(supertrend):
                stop_candidates.append(supertrend)
            
            # Parabolic SAR止损
            if not pd.isna(psar) and psar < current_price:
                stop_candidates.append(psar)
            
            # ATR止损
            if not pd.isna(atr) and self.entry_price > 0:
                stop_candidates.append(current_price - (atr * 2))
            
            if stop_candidates:
                self.dynamic_stop_loss = max(stop_candidates)
        
        else:  # 空仓
            # 使用多种止损中的最低者
            stop_candidates = []
            
            # 固定百分比止损
            if self.entry_price > 0:
                stop_candidates.append(self.entry_price * (1 + self.stop_loss_pct))
            
            # 追踪止损
            if self.lowest_price > 0:
                stop_candidates.append(self.lowest_price * (1 + self.trailing_stop_pct))
            
            # Supertrend止损
            if not pd.isna(supertrend):
                stop_candidates.append(supertrend)
            
            # Parabolic SAR止损
            if not pd.isna(psar) and psar > current_price:
                stop_candidates.append(psar)
            
            # ATR止损
            if not pd.isna(atr) and self.entry_price > 0:
                stop_candidates.append(current_price + (atr * 2))
            
            if stop_candidates:
                self.dynamic_stop_loss = min(stop_candidates)
    
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
        
        # 分析趋势
        trend_direction, trend_strength = self.analyze_trend(indicators)
        
        # 趋势强度不足
        if trend_strength < self.min_trend_strength:
            return None
        
        # ADX确认趋势强度
        adx = indicators['adx'].iloc[-1]
        if pd.isna(adx) or adx < self.adx_threshold:
            return None
        
        # 更新趋势状态
        if trend_direction != self.trend_direction:
            self.trend_direction = trend_direction
            self.trend_start_time = datetime.now()
        
        self.trend_strength = trend_strength
        
        # 计算动态仓位大小
        self.position_size_multiplier = self.calculate_dynamic_position_size(indicators)
        
        # 成交量确认
        volume_ratio = indicators['volume_ratio'].iloc[-1]
        if not pd.isna(volume_ratio) and volume_ratio < 1.2:
            return None  # 成交量不足
        
        # 布林带位置检查 (避免在极端位置开仓)
        bb_position = indicators['bb_position'].iloc[-1]
        if not pd.isna(bb_position):
            if trend_direction == 1 and bb_position > 0.8:
                return None  # 价格在布林带顶部，避免追高
            elif trend_direction == -1 and bb_position < 0.2:
                return None  # 价格在布林带底部，避免杀跌
        
        # 返回入场信号
        if trend_direction == 1:
            return SignalType.BUY
        elif trend_direction == -1:
            return SignalType.SELL
        
        return None
    
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
        
        # 更新动态止损
        self.update_dynamic_stop_loss(data, indicators)
        
        # 动态止损检查
        if self.dynamic_stop_loss > 0:
            if self.position > 0 and current_price <= self.dynamic_stop_loss:
                return SignalType.SELL
            elif self.position < 0 and current_price >= self.dynamic_stop_loss:
                return SignalType.BUY
        
        # 趋势反转检查
        current_trend, current_strength = self.analyze_trend(indicators)
        
        # 趋势方向改变
        if current_trend != self.trend_direction and current_trend != 0:
            if self.position > 0:
                return SignalType.SELL
            else:
                return SignalType.BUY
        
        # 趋势强度减弱
        if current_strength < self.min_trend_strength * 0.6:
            if self.position > 0:
                return SignalType.SELL
            else:
                return SignalType.BUY
        
        # ADX减弱检查
        adx = indicators['adx'].iloc[-1]
        if not pd.isna(adx) and adx < self.adx_threshold * 0.7:
            if self.position > 0:
                return SignalType.SELL
            else:
                return SignalType.BUY
        
        return None
    
    def on_data(self, data: pd.DataFrame) -> List[TradeSignal]:
        """
        处理新数据
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            List[TradeSignal]: 生成的交易信号列表
        """
        if len(data) < max(self.ema_slow, self.supertrend_period, self.adx_period):
            return []
        
        # 计算技术指标
        indicators = self.calculate_indicators(data)
        self.indicators_cache = indicators
        
        # 获取当前价格
        current_price = data['close'].iloc[-1]
        
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
                quantity_percent=abs(self.position) / current_price if self.entry_price > 0 else self.base_position_size,
                confidence=0.8,
                metadata={
                    'action': 'exit',
                    'exit_reason': 'trend_change_or_stop_loss',
                    'entry_price': self.entry_price,
                    'current_price': current_price,
                    'dynamic_stop_loss': self.dynamic_stop_loss,
                    'trend_direction': self.trend_direction,
                    'trend_strength': self.trend_strength,
                    'pnl_percent': ((current_price - self.entry_price) / self.entry_price * 100) if self.entry_price > 0 else 0
                }
            )
            signals.append(signal)
            
            print(f"[TREND] 平仓信号: {signal_type.value} @ {current_price:.2f}")
        
        # 检查开仓信号
        else:
            enter_signal = self.should_enter_position(data, indicators)
            if enter_signal:
                # 计算仓位大小
                position_size = self.base_position_size * self.position_size_multiplier
                
                # 创建开仓信号
                signal = self.create_signal(
                    signal_type=enter_signal,
                    quantity_percent=position_size,
                    confidence=self.trend_strength,
                    metadata={
                        'action': 'enter',
                        'trend_direction': self.trend_direction,
                        'trend_strength': self.trend_strength,
                        'position_multiplier': self.position_size_multiplier,
                        'adx': indicators['adx'].iloc[-1] if not pd.isna(indicators['adx'].iloc[-1]) else 0,
                        'supertrend': indicators['supertrend'].iloc[-1] if not pd.isna(indicators['supertrend'].iloc[-1]) else 0,
                        'psar': indicators['psar'].iloc[-1] if not pd.isna(indicators['psar'].iloc[-1]) else 0
                    }
                )
                signals.append(signal)
                
                print(f"[TREND] 开仓信号: {enter_signal.value} @ {current_price:.2f}, 强度: {self.trend_strength:.2f}, 仓位: {position_size:.1%}")
        
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
        
        # 更新入场价格和止损参数
        if order_event.side == "BUY":
            if self.position > 0:  # 新开多仓
                self.entry_price = order_event.price
                self.highest_price = order_event.price
                self.dynamic_stop_loss = 0
                print(f"[TREND] 多仓开仓: {self.position:.4f} @ {self.entry_price:.2f}")
        
        elif order_event.side == "SELL":
            if self.position < 0:  # 新开空仓
                self.entry_price = order_event.price
                self.lowest_price = order_event.price
                self.dynamic_stop_loss = 0
                print(f"[TREND] 空仓开仓: {self.position:.4f} @ {self.entry_price:.2f}")
            
            elif self.position == 0:  # 平仓
                print(f"[TREND] 平仓完成，PnL: {self.realized_pnl:.2f}")
                self.entry_price = 0
                self.highest_price = 0
                self.lowest_price = 0
                self.dynamic_stop_loss = 0
                self.trend_direction = 0
                self.trend_strength = 0
        
        return None
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        获取策略信息
        
        Returns:
            Dict[str, Any]: 策略状态信息
        """
        info = {
            'strategy_type': 'TrendFollowing',
            'position': self.position,
            'entry_price': self.entry_price,
            'dynamic_stop_loss': self.dynamic_stop_loss,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'position_size_multiplier': self.position_size_multiplier,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'parameters': {
                'ema_fast': self.ema_fast,
                'ema_slow': self.ema_slow,
                'supertrend_period': self.supertrend_period,
                'adx_threshold': self.adx_threshold,
                'base_position_size': self.base_position_size,
                'max_position_size': self.max_position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'trailing_stop_pct': self.trailing_stop_pct
            }
        }
        
        # 添加最新技术指标
        if self.indicators_cache:
            latest_idx = -1
            info['indicators'] = {
                'adx': self.indicators_cache.get('adx', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('adx', [])) > 0 else None,
                'supertrend': self.indicators_cache.get('supertrend', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('supertrend', [])) > 0 else None,
                'psar': self.indicators_cache.get('psar', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('psar', [])) > 0 else None,
                'macd': self.indicators_cache.get('macd', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('macd', [])) > 0 else None,
                'ema_diff': self.indicators_cache.get('ema_diff', pd.Series()).iloc[latest_idx] if len(self.indicators_cache.get('ema_diff', [])) > 0 else None
            }
        
        return info