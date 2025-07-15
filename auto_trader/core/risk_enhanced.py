"""
增强风险控制模块

扩展原有风险管理功能，增加：
- 动态风险限额调整
- 交易冷静期管理
- 异常行为检测
- 市场状态监控
- 风险事件通知
- 自动风险处置
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import threading
import time

from .risk import RiskManager, RiskLimits, RiskLevel, RiskCheckResult, RiskCheckReport, RiskMetrics
from ..strategies.base import TradeSignal, SignalType
from .broker import Order, Position, OrderStatus, OrderSide
from .account import AccountManager, PerformanceMetrics


class AnomalyType(Enum):
    """异常类型枚举"""
    UNUSUAL_TRADING_VOLUME = "unusual_trading_volume"     # 异常交易量
    RAPID_PRICE_MOVEMENT = "rapid_price_movement"         # 快速价格变动
    STRATEGY_MALFUNCTION = "strategy_malfunction"         # 策略故障
    MARKET_DISRUPTION = "market_disruption"              # 市场中断
    CONNECTION_ISSUE = "connection_issue"                # 连接问题
    UNUSUAL_SPREAD = "unusual_spread"                    # 异常价差
    HIGH_LATENCY = "high_latency"                        # 高延迟
    LIQUIDITY_SHORTAGE = "liquidity_shortage"            # 流动性不足


class RiskEventSeverity(Enum):
    """风险事件严重程度"""
    INFO = "info"           # 信息
    WARNING = "warning"     # 警告
    CRITICAL = "critical"   # 严重
    EMERGENCY = "emergency" # 紧急


@dataclass
class RiskEvent:
    """风险事件"""
    event_id: str
    event_type: AnomalyType
    severity: RiskEventSeverity
    message: str
    timestamp: datetime
    symbol: Optional[str] = None
    strategy_name: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'metrics': self.metrics,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None
        }


@dataclass
class CoolingPeriod:
    """交易冷静期"""
    symbol: str
    start_time: datetime
    duration_minutes: int
    reason: str
    triggered_by: str = ""
    
    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(minutes=self.duration_minutes)
    
    @property
    def is_active(self) -> bool:
        return datetime.now() < self.end_time
    
    @property
    def remaining_minutes(self) -> int:
        if not self.is_active:
            return 0
        remaining = self.end_time - datetime.now()
        return int(remaining.total_seconds() / 60)


@dataclass
class MarketState:
    """市场状态"""
    symbol: str
    timestamp: datetime
    price: float
    volume_24h: float
    volatility: float
    spread_percent: float
    liquidity_score: float = 1.0  # 0-1之间，1为最佳流动性
    market_cap_usdt: Optional[float] = None
    
    def is_healthy(self, min_volume: float = 1000000, max_volatility: float = 0.2, 
                   max_spread: float = 0.01) -> bool:
        """判断市场状态是否健康"""
        return (self.volume_24h >= min_volume and 
                self.volatility <= max_volatility and 
                self.spread_percent <= max_spread and
                self.liquidity_score >= 0.5)


class DynamicRiskAdjuster:
    """动态风险调整器"""
    
    def __init__(self):
        self.volatility_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.performance_history: deque = deque(maxlen=100)
        self.market_stress_indicators: Dict[str, float] = {}
        
    def calculate_market_stress(self, market_states: Dict[str, MarketState]) -> float:
        """
        计算市场压力指数
        
        Args:
            market_states: 各交易对的市场状态
            
        Returns:
            float: 市场压力指数 (0-1，1为最高压力)
        """
        if not market_states:
            return 0.0
        
        stress_factors = []
        
        for symbol, state in market_states.items():
            # 波动率因子
            volatility_stress = min(1.0, state.volatility / 0.2)
            
            # 流动性因子
            liquidity_stress = 1.0 - state.liquidity_score
            
            # 价差因子
            spread_stress = min(1.0, state.spread_percent / 0.01)
            
            # 加权平均
            symbol_stress = (volatility_stress * 0.4 + 
                           liquidity_stress * 0.3 + 
                           spread_stress * 0.3)
            stress_factors.append(symbol_stress)
        
        # 计算总体市场压力
        market_stress = np.mean(stress_factors)
        
        # 更新历史记录
        self.market_stress_indicators['current'] = market_stress
        self.market_stress_indicators['timestamp'] = datetime.now().isoformat()
        
        return market_stress
    
    def adjust_risk_limits(self, base_limits: RiskLimits, market_stress: float, 
                          portfolio_performance: float) -> RiskLimits:
        """
        根据市场压力和组合表现动态调整风险限额
        
        Args:
            base_limits: 基础风险限额
            market_stress: 市场压力指数
            portfolio_performance: 组合表现（年化收益率）
            
        Returns:
            RiskLimits: 调整后的风险限额
        """
        # 复制基础限额
        adjusted_limits = RiskLimits(
            max_position_percent=base_limits.max_position_percent,
            max_total_position_percent=base_limits.max_total_position_percent,
            max_symbol_positions=base_limits.max_symbol_positions,
            max_daily_loss_percent=base_limits.max_daily_loss_percent,
            max_total_loss_percent=base_limits.max_total_loss_percent,
            max_drawdown_percent=base_limits.max_drawdown_percent,
            max_trades_per_hour=base_limits.max_trades_per_hour,
            max_trades_per_day=base_limits.max_trades_per_day,
            min_trade_interval_seconds=base_limits.min_trade_interval_seconds,
            max_price_deviation_percent=base_limits.max_price_deviation_percent,
            min_order_value_usdt=base_limits.min_order_value_usdt,
            max_order_value_usdt=base_limits.max_order_value_usdt,
            max_active_strategies=base_limits.max_active_strategies,
            max_correlation_threshold=base_limits.max_correlation_threshold,
            min_market_volume_24h=base_limits.min_market_volume_24h,
            max_volatility_threshold=base_limits.max_volatility_threshold
        )
        
        # 根据市场压力调整
        stress_multiplier = 1.0 - (market_stress * 0.5)  # 压力越大，限额越严格
        
        adjusted_limits.max_position_percent *= stress_multiplier
        adjusted_limits.max_total_position_percent *= stress_multiplier
        adjusted_limits.max_daily_loss_percent *= stress_multiplier
        
        # 根据组合表现调整
        if portfolio_performance > 0.2:  # 表现良好
            performance_multiplier = 1.2
        elif portfolio_performance < -0.1:  # 表现不佳
            performance_multiplier = 0.7
        else:
            performance_multiplier = 1.0
        
        adjusted_limits.max_position_percent *= performance_multiplier
        adjusted_limits.max_total_position_percent *= performance_multiplier
        
        # 确保限额在合理范围内
        adjusted_limits.max_position_percent = max(0.01, min(0.5, adjusted_limits.max_position_percent))
        adjusted_limits.max_total_position_percent = max(0.1, min(1.0, adjusted_limits.max_total_position_percent))
        adjusted_limits.max_daily_loss_percent = max(0.01, min(0.2, adjusted_limits.max_daily_loss_percent))
        
        return adjusted_limits


class AnomalyDetector:
    """异常检测器"""
    
    def __init__(self):
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.volume_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.latency_history: deque = deque(maxlen=100)
        self.strategy_performance: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
    def detect_price_anomalies(self, symbol: str, current_price: float, 
                              threshold_std: float = 3.0) -> Optional[RiskEvent]:
        """
        检测价格异常
        
        Args:
            symbol: 交易对
            current_price: 当前价格
            threshold_std: 异常阈值（标准差倍数）
            
        Returns:
            Optional[RiskEvent]: 检测到的异常事件
        """
        price_hist = self.price_history[symbol]
        price_hist.append(current_price)
        
        if len(price_hist) < 20:
            return None
        
        # 计算价格变化率
        recent_prices = list(price_hist)[-20:]
        price_changes = [abs((recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1]) 
                        for i in range(1, len(recent_prices))]
        
        if not price_changes:
            return None
        
        mean_change = np.mean(price_changes)
        std_change = np.std(price_changes)
        
        if std_change == 0:
            return None
        
        # 检查最新价格变化
        if len(recent_prices) >= 2:
            latest_change = abs((recent_prices[-1] - recent_prices[-2]) / recent_prices[-2])
            z_score = (latest_change - mean_change) / std_change
            
            if z_score > threshold_std:
                return RiskEvent(
                    event_id=f"price_anomaly_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    event_type=AnomalyType.RAPID_PRICE_MOVEMENT,
                    severity=RiskEventSeverity.WARNING if z_score < 5 else RiskEventSeverity.CRITICAL,
                    message=f"{symbol} 价格异常变动: {latest_change:.2%}, Z-score: {z_score:.2f}",
                    timestamp=datetime.now(),
                    symbol=symbol,
                    metrics={
                        'price_change_percent': latest_change,
                        'z_score': z_score,
                        'current_price': current_price,
                        'mean_change': mean_change,
                        'std_change': std_change
                    }
                )
        
        return None
    
    def detect_volume_anomalies(self, symbol: str, current_volume: float, 
                               threshold_multiplier: float = 5.0) -> Optional[RiskEvent]:
        """
        检测成交量异常
        
        Args:
            symbol: 交易对
            current_volume: 当前成交量
            threshold_multiplier: 异常阈值倍数
            
        Returns:
            Optional[RiskEvent]: 检测到的异常事件
        """
        volume_hist = self.volume_history[symbol]
        volume_hist.append(current_volume)
        
        if len(volume_hist) < 10:
            return None
        
        recent_volumes = list(volume_hist)[-10:]
        avg_volume = np.mean(recent_volumes[:-1])  # 排除当前值
        
        if avg_volume == 0:
            return None
        
        volume_ratio = current_volume / avg_volume
        
        if volume_ratio > threshold_multiplier or volume_ratio < (1/threshold_multiplier):
            severity = RiskEventSeverity.INFO
            if volume_ratio > 10 or volume_ratio < 0.1:
                severity = RiskEventSeverity.WARNING
            if volume_ratio > 20 or volume_ratio < 0.05:
                severity = RiskEventSeverity.CRITICAL
            
            return RiskEvent(
                event_id=f"volume_anomaly_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                event_type=AnomalyType.UNUSUAL_TRADING_VOLUME,
                severity=severity,
                message=f"{symbol} 成交量异常: {volume_ratio:.2f}x 平均值",
                timestamp=datetime.now(),
                symbol=symbol,
                metrics={
                    'volume_ratio': volume_ratio,
                    'current_volume': current_volume,
                    'average_volume': avg_volume,
                    'threshold_multiplier': threshold_multiplier
                }
            )
        
        return None
    
    def detect_strategy_malfunction(self, strategy_name: str, recent_returns: List[float], 
                                   loss_threshold: float = -0.1) -> Optional[RiskEvent]:
        """
        检测策略故障
        
        Args:
            strategy_name: 策略名称
            recent_returns: 最近的收益率
            loss_threshold: 损失阈值
            
        Returns:
            Optional[RiskEvent]: 检测到的异常事件
        """
        if not recent_returns or len(recent_returns) < 5:
            return None
        
        # 检查连续亏损
        consecutive_losses = 0
        total_loss = 0
        
        for ret in reversed(recent_returns):
            if ret < 0:
                consecutive_losses += 1
                total_loss += ret
            else:
                break
        
        # 检查是否连续亏损超过阈值
        if consecutive_losses >= 5 and total_loss < loss_threshold:
            return RiskEvent(
                event_id=f"strategy_malfunction_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                event_type=AnomalyType.STRATEGY_MALFUNCTION,
                severity=RiskEventSeverity.CRITICAL,
                message=f"策略 {strategy_name} 疑似故障: 连续{consecutive_losses}次亏损，总损失{total_loss:.2%}",
                timestamp=datetime.now(),
                strategy_name=strategy_name,
                metrics={
                    'consecutive_losses': consecutive_losses,
                    'total_loss': total_loss,
                    'loss_threshold': loss_threshold,
                    'recent_returns': recent_returns[-10:]  # 最近10次
                }
            )
        
        return None
    
    def detect_latency_issues(self, execution_latency_ms: float, 
                             threshold_ms: float = 1000) -> Optional[RiskEvent]:
        """
        检测执行延迟问题
        
        Args:
            execution_latency_ms: 执行延迟（毫秒）
            threshold_ms: 延迟阈值（毫秒）
            
        Returns:
            Optional[RiskEvent]: 检测到的异常事件
        """
        self.latency_history.append(execution_latency_ms)
        
        if execution_latency_ms > threshold_ms:
            # 检查是否持续高延迟
            recent_latencies = list(self.latency_history)[-5:]
            avg_recent_latency = np.mean(recent_latencies) if recent_latencies else execution_latency_ms
            
            severity = RiskEventSeverity.WARNING
            if avg_recent_latency > threshold_ms * 2:
                severity = RiskEventSeverity.CRITICAL
            
            return RiskEvent(
                event_id=f"high_latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                event_type=AnomalyType.HIGH_LATENCY,
                severity=severity,
                message=f"执行延迟过高: {execution_latency_ms:.0f}ms (阈值: {threshold_ms}ms)",
                timestamp=datetime.now(),
                metrics={
                    'execution_latency_ms': execution_latency_ms,
                    'threshold_ms': threshold_ms,
                    'avg_recent_latency': avg_recent_latency,
                    'recent_latencies': recent_latencies
                }
            )
        
        return None


class EnhancedRiskManager(RiskManager):
    """增强风险管理器"""
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None, 
                 notification_callback: Optional[Callable] = None):
        """
        初始化增强风险管理器
        
        Args:
            risk_limits: 风险限制配置
            notification_callback: 通知回调函数
        """
        super().__init__(risk_limits)
        
        # 增强组件
        self.dynamic_adjuster = DynamicRiskAdjuster()
        self.anomaly_detector = AnomalyDetector()
        self.notification_callback = notification_callback
        
        # 风险事件管理
        self.risk_events: List[RiskEvent] = []
        self.active_cooling_periods: Dict[str, CoolingPeriod] = {}
        
        # 市场状态监控
        self.market_states: Dict[str, MarketState] = {}
        self.last_market_update: Optional[datetime] = None
        
        # 自动处置配置
        self.auto_response_enabled = True
        self.emergency_actions: Dict[AnomalyType, Callable] = {}
        
        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        
        print("增强风险管理器初始化完成")
    
    def start_monitoring(self, check_interval_seconds: int = 30) -> None:
        """
        启动风险监控
        
        Args:
            check_interval_seconds: 检查间隔（秒）
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        print(f"风险监控已启动，检查间隔: {check_interval_seconds}秒")
    
    def stop_monitoring(self) -> None:
        """停止风险监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("风险监控已停止")
    
    def _monitoring_loop(self, check_interval: int) -> None:
        """监控循环"""
        while self.monitoring_active:
            try:
                self._periodic_risk_check()
                time.sleep(check_interval)
            except Exception as e:
                print(f"风险监控循环异常: {e}")
                time.sleep(check_interval)
    
    def _periodic_risk_check(self) -> None:
        """定期风险检查"""
        current_time = datetime.now()
        
        # 清理过期的冷静期
        self._cleanup_expired_cooling_periods()
        
        # 检查市场状态异常
        self._check_market_health()
        
        # 检查风险事件是否需要自动处置
        self._handle_automatic_responses()
        
        # 清理旧的风险事件
        cutoff_time = current_time - timedelta(days=7)
        self.risk_events = [event for event in self.risk_events if event.timestamp >= cutoff_time]
    
    def _cleanup_expired_cooling_periods(self) -> None:
        """清理过期的冷静期"""
        expired_symbols = [symbol for symbol, period in self.active_cooling_periods.items() 
                          if not period.is_active]
        
        for symbol in expired_symbols:
            del self.active_cooling_periods[symbol]
            print(f"冷静期结束: {symbol}")
    
    def _check_market_health(self) -> None:
        """检查市场健康状态"""
        unhealthy_markets = []
        
        for symbol, state in self.market_states.items():
            if not state.is_healthy():
                unhealthy_markets.append(symbol)
        
        if unhealthy_markets:
            event = RiskEvent(
                event_id=f"market_health_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                event_type=AnomalyType.MARKET_DISRUPTION,
                severity=RiskEventSeverity.WARNING,
                message=f"市场状态不健康: {', '.join(unhealthy_markets)}",
                timestamp=datetime.now(),
                metrics={'unhealthy_markets': unhealthy_markets}
            )
            self.add_risk_event(event)
    
    def _handle_automatic_responses(self) -> None:
        """处理自动响应"""
        if not self.auto_response_enabled:
            return
        
        # 获取未解决的严重事件
        critical_events = [event for event in self.risk_events 
                          if not event.resolved and event.severity in [RiskEventSeverity.CRITICAL, RiskEventSeverity.EMERGENCY]]
        
        for event in critical_events:
            if event.event_type in self.emergency_actions:
                try:
                    self.emergency_actions[event.event_type](event)
                    event.resolved = True
                    event.resolution_time = datetime.now()
                except Exception as e:
                    print(f"自动处置失败 {event.event_id}: {e}")
    
    def update_market_state(self, symbol: str, price: float, volume_24h: float, 
                           volatility: float, spread_percent: float, 
                           liquidity_score: float = 1.0) -> None:
        """
        更新市场状态
        
        Args:
            symbol: 交易对
            price: 当前价格
            volume_24h: 24小时成交量
            volatility: 波动率
            spread_percent: 价差百分比
            liquidity_score: 流动性评分
        """
        self.market_states[symbol] = MarketState(
            symbol=symbol,
            timestamp=datetime.now(),
            price=price,
            volume_24h=volume_24h,
            volatility=volatility,
            spread_percent=spread_percent,
            liquidity_score=liquidity_score
        )
        
        self.last_market_update = datetime.now()
        
        # 更新价格缓存
        self.update_price_cache(symbol, price)
        
        # 检测异常
        self._detect_and_handle_anomalies(symbol, price, volume_24h)
    
    def _detect_and_handle_anomalies(self, symbol: str, price: float, volume: float) -> None:
        """检测并处理异常"""
        # 价格异常检测
        price_anomaly = self.anomaly_detector.detect_price_anomalies(symbol, price)
        if price_anomaly:
            self.add_risk_event(price_anomaly)
        
        # 成交量异常检测
        volume_anomaly = self.anomaly_detector.detect_volume_anomalies(symbol, volume)
        if volume_anomaly:
            self.add_risk_event(volume_anomaly)
    
    def add_risk_event(self, event: RiskEvent) -> None:
        """
        添加风险事件
        
        Args:
            event: 风险事件
        """
        self.risk_events.append(event)
        
        # 发送通知
        if self.notification_callback:
            try:
                self.notification_callback(event)
            except Exception as e:
                print(f"风险事件通知发送失败: {e}")
        
        # 记录事件
        print(f"风险事件: {event.severity.value.upper()} - {event.message}")
        
        # 检查是否需要触发冷静期
        if event.severity in [RiskEventSeverity.CRITICAL, RiskEventSeverity.EMERGENCY]:
            self._trigger_cooling_period_if_needed(event)
    
    def _trigger_cooling_period_if_needed(self, event: RiskEvent) -> None:
        """根据事件触发冷静期"""
        if not event.symbol:
            return
        
        cooling_duration = 30  # 默认30分钟
        
        if event.event_type == AnomalyType.RAPID_PRICE_MOVEMENT:
            cooling_duration = 15
        elif event.event_type == AnomalyType.STRATEGY_MALFUNCTION:
            cooling_duration = 60
        elif event.event_type == AnomalyType.MARKET_DISRUPTION:
            cooling_duration = 45
        
        self.set_cooling_period(event.symbol, cooling_duration, f"自动触发: {event.message}")
    
    def set_cooling_period(self, symbol: str, duration_minutes: int, reason: str, 
                          triggered_by: str = "system") -> None:
        """
        设置交易冷静期
        
        Args:
            symbol: 交易对
            duration_minutes: 持续时间（分钟）
            reason: 原因
            triggered_by: 触发者
        """
        cooling_period = CoolingPeriod(
            symbol=symbol,
            start_time=datetime.now(),
            duration_minutes=duration_minutes,
            reason=reason,
            triggered_by=triggered_by
        )
        
        self.active_cooling_periods[symbol] = cooling_period
        
        print(f"设置冷静期: {symbol} - {duration_minutes}分钟 ({reason})")
    
    def is_in_cooling_period(self, symbol: str) -> bool:
        """
        检查是否在冷静期
        
        Args:
            symbol: 交易对
            
        Returns:
            bool: 是否在冷静期
        """
        if symbol not in self.active_cooling_periods:
            return False
        
        return self.active_cooling_periods[symbol].is_active
    
    def get_cooling_period_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取冷静期信息
        
        Args:
            symbol: 交易对
            
        Returns:
            Optional[Dict[str, Any]]: 冷静期信息
        """
        if symbol not in self.active_cooling_periods:
            return None
        
        period = self.active_cooling_periods[symbol]
        return {
            'symbol': period.symbol,
            'start_time': period.start_time.isoformat(),
            'end_time': period.end_time.isoformat(),
            'duration_minutes': period.duration_minutes,
            'remaining_minutes': period.remaining_minutes,
            'reason': period.reason,
            'triggered_by': period.triggered_by,
            'is_active': period.is_active
        }
    
    def check_enhanced_signal_risk(self, signal: TradeSignal, account_manager: AccountManager, 
                                  proposed_quantity: float, current_price: float) -> RiskCheckReport:
        """
        增强的交易信号风险检查
        
        Args:
            signal: 交易信号
            account_manager: 账户管理器
            proposed_quantity: 建议交易数量
            current_price: 当前价格
            
        Returns:
            RiskCheckReport: 风险检查报告
        """
        # 基础风险检查
        base_report = super().check_signal_risk(signal, account_manager, proposed_quantity, current_price)
        
        # 检查冷静期
        if self.is_in_cooling_period(signal.symbol):
            cooling_info = self.get_cooling_period_info(signal.symbol)
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.HIGH,
                message=f"交易对 {signal.symbol} 处于冷静期，剩余 {cooling_info['remaining_minutes']} 分钟",
                suggestions=[f"等待冷静期结束: {cooling_info['reason']}"],
                metrics=cooling_info
            )
        
        # 检查市场状态
        if signal.symbol in self.market_states:
            market_state = self.market_states[signal.symbol]
            if not market_state.is_healthy():
                return RiskCheckReport(
                    result=RiskCheckResult.WARNING,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"市场状态不佳: {signal.symbol}",
                    suggestions=["考虑降低仓位或延迟交易"],
                    metrics={
                        'volume_24h': market_state.volume_24h,
                        'volatility': market_state.volatility,
                        'spread_percent': market_state.spread_percent,
                        'liquidity_score': market_state.liquidity_score
                    }
                )
        
        # 动态调整风险限额
        if len(account_manager.daily_values) > 0:
            current_value = account_manager.daily_values[-1]['total_value']
            initial_value = account_manager.daily_values[0]['total_value']
            portfolio_performance = (current_value - initial_value) / initial_value if initial_value > 0 else 0
            
            market_stress = self.dynamic_adjuster.calculate_market_stress(self.market_states)
            adjusted_limits = self.dynamic_adjuster.adjust_risk_limits(
                self.risk_limits, market_stress, portfolio_performance
            )
            
            # 使用调整后的限额重新检查仓位风险
            position_value = proposed_quantity * current_price
            account_summary = account_manager.get_account_summary()
            total_value = account_summary.get('total_value_usdt', 0)
            
            if total_value > 0:
                position_percent = position_value / total_value
                if position_percent > adjusted_limits.max_position_percent:
                    return RiskCheckReport(
                        result=RiskCheckResult.REJECT,
                        risk_level=RiskLevel.HIGH,
                        message=f"动态调整后仓位超限: {position_percent:.2%} > {adjusted_limits.max_position_percent:.2%}",
                        suggestions=[f"市场压力: {market_stress:.2%}, 组合表现: {portfolio_performance:.2%}"],
                        metrics={
                            'original_limit': self.risk_limits.max_position_percent,
                            'adjusted_limit': adjusted_limits.max_position_percent,
                            'market_stress': market_stress,
                            'portfolio_performance': portfolio_performance
                        }
                    )
        
        return base_report
    
    def register_emergency_action(self, anomaly_type: AnomalyType, action: Callable) -> None:
        """
        注册紧急处置动作
        
        Args:
            anomaly_type: 异常类型
            action: 处置动作函数
        """
        self.emergency_actions[anomaly_type] = action
        print(f"注册紧急处置动作: {anomaly_type.value}")
    
    def get_risk_dashboard(self) -> Dict[str, Any]:
        """
        获取风险监控仪表板数据
        
        Returns:
            Dict[str, Any]: 仪表板数据
        """
        # 基础风险摘要
        base_summary = self.get_risk_summary()
        
        # 增强信息
        recent_events = [event.to_dict() for event in self.risk_events[-10:]]
        active_cooling_periods = {symbol: self.get_cooling_period_info(symbol) 
                                 for symbol in self.active_cooling_periods.keys()}
        
        market_health = {}
        for symbol, state in self.market_states.items():
            market_health[symbol] = {
                'is_healthy': state.is_healthy(),
                'volume_24h': state.volume_24h,
                'volatility': state.volatility,
                'spread_percent': state.spread_percent,
                'liquidity_score': state.liquidity_score,
                'last_update': state.timestamp.isoformat()
            }
        
        # 计算市场压力
        market_stress = self.dynamic_adjuster.calculate_market_stress(self.market_states)
        
        return {
            'base_risk_summary': base_summary,
            'market_stress': market_stress,
            'recent_risk_events': recent_events,
            'active_cooling_periods': active_cooling_periods,
            'market_health': market_health,
            'monitoring_status': {
                'is_active': self.monitoring_active,
                'last_market_update': self.last_market_update.isoformat() if self.last_market_update else None,
                'auto_response_enabled': self.auto_response_enabled
            },
            'statistics': {
                'total_risk_events': len(self.risk_events),
                'unresolved_events': len([e for e in self.risk_events if not e.resolved]),
                'critical_events_24h': len([e for e in self.risk_events 
                                          if e.severity in [RiskEventSeverity.CRITICAL, RiskEventSeverity.EMERGENCY] 
                                          and e.timestamp >= datetime.now() - timedelta(hours=24)]),
                'active_cooling_periods_count': len(self.active_cooling_periods)
            },
            'dashboard_time': datetime.now().isoformat()
        }
    
    def export_enhanced_risk_report(self) -> Dict[str, Any]:
        """
        导出增强风险报告
        
        Returns:
            Dict[str, Any]: 完整的增强风险报告
        """
        base_report = super().export_risk_report()
        
        enhanced_data = {
            'risk_events': [event.to_dict() for event in self.risk_events],
            'cooling_periods_history': [
                {
                    'symbol': period.symbol,
                    'start_time': period.start_time.isoformat(),
                    'end_time': period.end_time.isoformat(),
                    'duration_minutes': period.duration_minutes,
                    'reason': period.reason,
                    'triggered_by': period.triggered_by,
                    'is_active': period.is_active
                }
                for period in self.active_cooling_periods.values()
            ],
            'market_states': {
                symbol: {
                    'timestamp': state.timestamp.isoformat(),
                    'price': state.price,
                    'volume_24h': state.volume_24h,
                    'volatility': state.volatility,
                    'spread_percent': state.spread_percent,
                    'liquidity_score': state.liquidity_score,
                    'is_healthy': state.is_healthy()
                }
                for symbol, state in self.market_states.items()
            },
            'market_stress_indicators': self.dynamic_adjuster.market_stress_indicators,
            'anomaly_detection_stats': {
                'price_history_length': {symbol: len(hist) for symbol, hist in self.anomaly_detector.price_history.items()},
                'volume_history_length': {symbol: len(hist) for symbol, hist in self.anomaly_detector.volume_history.items()},
                'latency_history_length': len(self.anomaly_detector.latency_history)
            }
        }
        
        # 合并报告
        base_report.update(enhanced_data)
        
        return base_report