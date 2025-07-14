"""
风控模块 - 负责风险管理和控制

这个模块提供了全面的风险管理功能，包括：
- 仓位限制和资金管理
- 止损止盈控制
- 最大回撤监控
- 交易频率限制
- 异常行为检测
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd

from ..strategies.base import TradeSignal, SignalType
from .broker import Order, Position, OrderStatus, OrderSide
from .account import AccountManager, PerformanceMetrics


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "LOW"                  # 低风险
    MEDIUM = "MEDIUM"            # 中风险
    HIGH = "HIGH"                # 高风险
    CRITICAL = "CRITICAL"        # 极高风险


class RiskCheckResult(Enum):
    """风控检查结果枚举"""
    PASS = "PASS"                # 通过
    WARNING = "WARNING"          # 警告
    REJECT = "REJECT"            # 拒绝


@dataclass
class RiskCheckReport:
    """风控检查报告"""
    result: RiskCheckResult      # 检查结果
    risk_level: RiskLevel        # 风险等级
    message: str                 # 检查信息
    suggestions: List[str] = field(default_factory=list)  # 建议
    metrics: Dict[str, Any] = field(default_factory=dict) # 相关指标
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'result': self.result.value,
            'risk_level': self.risk_level.value,
            'message': self.message,
            'suggestions': self.suggestions,
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }


@dataclass
class RiskLimits:
    """风险限制配置"""
    # 仓位限制
    max_position_percent: float = 0.1       # 单个仓位最大占总资金比例
    max_total_position_percent: float = 0.8 # 总仓位最大占总资金比例
    max_symbol_positions: int = 5           # 单个交易对最大仓位数
    
    # 损失限制
    max_daily_loss_percent: float = 0.05    # 每日最大损失比例
    max_total_loss_percent: float = 0.20    # 总最大损失比例
    max_drawdown_percent: float = 0.15      # 最大回撤比例
    
    # 交易频率限制
    max_trades_per_hour: int = 10           # 每小时最大交易次数
    max_trades_per_day: int = 100           # 每日最大交易次数
    min_trade_interval_seconds: int = 60    # 最小交易间隔（秒）
    
    # 价格限制
    max_price_deviation_percent: float = 0.05  # 最大价格偏离市价比例
    min_order_value_usdt: float = 10.0      # 最小订单价值（USDT）
    max_order_value_usdt: float = 50000.0   # 最大订单价值（USDT）
    
    # 策略限制
    max_active_strategies: int = 5          # 最大活跃策略数
    max_correlation_threshold: float = 0.8  # 最大策略相关性阈值
    
    # 市场状态限制
    min_market_volume_24h: float = 1000000  # 最小24小时市场成交量
    max_volatility_threshold: float = 0.20  # 最大波动率阈值
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'max_position_percent': self.max_position_percent,
            'max_total_position_percent': self.max_total_position_percent,
            'max_symbol_positions': self.max_symbol_positions,
            'max_daily_loss_percent': self.max_daily_loss_percent,
            'max_total_loss_percent': self.max_total_loss_percent,
            'max_drawdown_percent': self.max_drawdown_percent,
            'max_trades_per_hour': self.max_trades_per_hour,
            'max_trades_per_day': self.max_trades_per_day,
            'min_trade_interval_seconds': self.min_trade_interval_seconds,
            'max_price_deviation_percent': self.max_price_deviation_percent,
            'min_order_value_usdt': self.min_order_value_usdt,
            'max_order_value_usdt': self.max_order_value_usdt,
            'max_active_strategies': self.max_active_strategies,
            'max_correlation_threshold': self.max_correlation_threshold,
            'min_market_volume_24h': self.min_market_volume_24h,
            'max_volatility_threshold': self.max_volatility_threshold
        }


@dataclass
class RiskMetrics:
    """风险指标"""
    # 仓位风险
    current_position_percent: float = 0.0   # 当前仓位比例
    total_position_percent: float = 0.0     # 总仓位比例
    position_concentration: float = 0.0     # 仓位集中度
    
    # 损失风险
    daily_pnl_percent: float = 0.0          # 当日盈亏比例
    total_pnl_percent: float = 0.0          # 总盈亏比例
    current_drawdown_percent: float = 0.0   # 当前回撤比例
    
    # 交易风险
    trades_last_hour: int = 0               # 最近一小时交易次数
    trades_today: int = 0                   # 今日交易次数
    avg_trade_interval_seconds: float = 0.0 # 平均交易间隔
    
    # 市场风险
    market_volatility: float = 0.0          # 市场波动率
    correlation_risk: float = 0.0           # 相关性风险
    
    # 策略风险
    active_strategies_count: int = 0        # 活跃策略数
    strategy_concentration: float = 0.0     # 策略集中度
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'current_position_percent': self.current_position_percent,
            'total_position_percent': self.total_position_percent,
            'position_concentration': self.position_concentration,
            'daily_pnl_percent': self.daily_pnl_percent,
            'total_pnl_percent': self.total_pnl_percent,
            'current_drawdown_percent': self.current_drawdown_percent,
            'trades_last_hour': self.trades_last_hour,
            'trades_today': self.trades_today,
            'avg_trade_interval_seconds': self.avg_trade_interval_seconds,
            'market_volatility': self.market_volatility,
            'correlation_risk': self.correlation_risk,
            'active_strategies_count': self.active_strategies_count,
            'strategy_concentration': self.strategy_concentration
        }


class RiskManager:
    """风险管理器"""
    
    def __init__(self, risk_limits: Optional[RiskLimits] = None):
        """
        初始化风险管理器
        
        Args:
            risk_limits: 风险限制配置
        """
        self.risk_limits = risk_limits or RiskLimits()
        
        # 风险状态
        self.current_risk_level = RiskLevel.LOW
        self.risk_metrics = RiskMetrics()
        
        # 交易记录（用于频率限制）
        self.recent_trades: List[Dict[str, Any]] = []
        
        # 价格缓存
        self.price_cache: Dict[str, float] = {}
        
        # 风险检查历史
        self.risk_check_history: List[RiskCheckReport] = []
        
        # 紧急状态
        self.emergency_stop = False
        self.emergency_reason = ""
        
        print("风险管理器初始化完成")
    
    def update_risk_limits(self, risk_limits: RiskLimits) -> None:
        """
        更新风险限制配置
        
        Args:
            risk_limits: 新的风险限制配置
        """
        self.risk_limits = risk_limits
        print("风险限制配置已更新")
    
    def update_price_cache(self, symbol: str, price: float) -> None:
        """
        更新价格缓存
        
        Args:
            symbol: 交易对符号
            price: 当前价格
        """
        self.price_cache[symbol] = price
    
    def calculate_risk_metrics(self, account_manager: AccountManager) -> RiskMetrics:
        """
        计算风险指标
        
        Args:
            account_manager: 账户管理器
            
        Returns:
            RiskMetrics: 风险指标对象
        """
        metrics = RiskMetrics()
        
        # 获取账户信息
        account_summary = account_manager.get_account_summary()
        total_value = account_summary.get('total_value_usdt', 0)
        
        if total_value <= 0:
            return metrics
        
        # 计算仓位风险
        total_position_value = 0.0
        position_values = []
        
        for symbol, position in account_manager.positions.items():
            if symbol in self.price_cache:
                position_value = abs(position.quantity * self.price_cache[symbol])
                total_position_value += position_value
                position_values.append(position_value)
        
        metrics.total_position_percent = total_position_value / total_value
        
        # 计算仓位集中度（赫芬达尔指数）
        if position_values:
            total_squared = sum((value / total_position_value) ** 2 for value in position_values)
            metrics.position_concentration = total_squared
        
        # 计算损失风险
        metrics.total_pnl_percent = account_summary.get('total_return', 0)
        
        # 计算当日盈亏
        if account_manager.daily_values:
            today_value = account_manager.daily_values[-1].get('total_value', total_value)
            yesterday_value = account_manager.daily_values[-2].get('total_value', total_value) if len(account_manager.daily_values) > 1 else total_value
            if yesterday_value > 0:
                metrics.daily_pnl_percent = (today_value - yesterday_value) / yesterday_value
        
        # 计算当前回撤
        if account_manager.daily_values:
            values = [day['total_value'] for day in account_manager.daily_values]
            if values:
                peak_value = max(values)
                current_value = values[-1]
                if peak_value > 0:
                    metrics.current_drawdown_percent = (peak_value - current_value) / peak_value
        
        # 计算交易频率风险
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        metrics.trades_last_hour = len([trade for trade in self.recent_trades 
                                      if trade['timestamp'] >= hour_ago])
        metrics.trades_today = len([trade for trade in self.recent_trades 
                                  if trade['timestamp'] >= today_start])
        
        # 计算平均交易间隔
        if len(self.recent_trades) > 1:
            intervals = []
            for i in range(1, len(self.recent_trades)):
                interval = (self.recent_trades[i]['timestamp'] - self.recent_trades[i-1]['timestamp']).total_seconds()
                intervals.append(interval)
            metrics.avg_trade_interval_seconds = np.mean(intervals)
        
        # 计算市场波动率（简化版）
        if len(account_manager.daily_values) > 1:
            daily_returns = []
            for i in range(1, len(account_manager.daily_values)):
                prev_value = account_manager.daily_values[i-1]['total_value']
                curr_value = account_manager.daily_values[i]['total_value']
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    daily_returns.append(daily_return)
            
            if daily_returns:
                metrics.market_volatility = np.std(daily_returns)
        
        # 更新风险指标
        self.risk_metrics = metrics
        
        return metrics
    
    def check_signal_risk(self, signal: TradeSignal, account_manager: AccountManager, 
                         proposed_quantity: float, current_price: float) -> RiskCheckReport:
        """
        检查交易信号的风险
        
        Args:
            signal: 交易信号
            account_manager: 账户管理器
            proposed_quantity: 建议交易数量
            current_price: 当前价格
            
        Returns:
            RiskCheckReport: 风险检查报告
        """
        # 更新风险指标
        self.calculate_risk_metrics(account_manager)
        
        # 检查紧急状态
        if self.emergency_stop:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.CRITICAL,
                message=f"紧急停止状态：{self.emergency_reason}",
                suggestions=["等待紧急状态解除"]
            )
        
        # 执行各项风险检查
        checks = [
            self._check_position_risk(signal, account_manager, proposed_quantity, current_price),
            self._check_loss_risk(signal, account_manager),
            self._check_trading_frequency_risk(signal),
            self._check_price_risk(signal, current_price),
            self._check_order_value_risk(signal, proposed_quantity, current_price),
            self._check_market_conditions_risk(signal)
        ]
        
        # 汇总检查结果
        final_report = self._aggregate_risk_checks(checks)
        
        # 记录检查历史
        self.risk_check_history.append(final_report)
        
        # 保留最近100条记录
        if len(self.risk_check_history) > 100:
            self.risk_check_history = self.risk_check_history[-100:]
        
        return final_report
    
    def _check_position_risk(self, signal: TradeSignal, account_manager: AccountManager, 
                            proposed_quantity: float, current_price: float) -> RiskCheckReport:
        """检查仓位风险"""
        account_summary = account_manager.get_account_summary()
        total_value = account_summary.get('total_value_usdt', 0)
        
        if total_value <= 0:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.CRITICAL,
                message="账户总价值无效",
                suggestions=["检查账户余额"]
            )
        
        # 计算建议仓位价值
        position_value = proposed_quantity * current_price
        position_percent = position_value / total_value
        
        # 检查单个仓位限制
        if position_percent > self.risk_limits.max_position_percent:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.HIGH,
                message=f"单个仓位比例过高：{position_percent:.2%} > {self.risk_limits.max_position_percent:.2%}",
                suggestions=[f"建议减少仓位至{self.risk_limits.max_position_percent:.2%}以下"],
                metrics={'position_percent': position_percent, 'limit': self.risk_limits.max_position_percent}
            )
        
        # 检查总仓位限制
        if self.risk_metrics.total_position_percent > self.risk_limits.max_total_position_percent:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.HIGH,
                message=f"总仓位比例过高：{self.risk_metrics.total_position_percent:.2%} > {self.risk_limits.max_total_position_percent:.2%}",
                suggestions=["建议先平仓部分头寸"],
                metrics={'total_position_percent': self.risk_metrics.total_position_percent}
            )
        
        # 检查仓位集中度
        if self.risk_metrics.position_concentration > 0.8:
            return RiskCheckReport(
                result=RiskCheckResult.WARNING,
                risk_level=RiskLevel.MEDIUM,
                message=f"仓位集中度较高：{self.risk_metrics.position_concentration:.2f}",
                suggestions=["建议分散投资"],
                metrics={'position_concentration': self.risk_metrics.position_concentration}
            )
        
        # 通过检查
        return RiskCheckReport(
            result=RiskCheckResult.PASS,
            risk_level=RiskLevel.LOW,
            message="仓位风险检查通过",
            metrics={
                'position_percent': position_percent,
                'total_position_percent': self.risk_metrics.total_position_percent,
                'position_concentration': self.risk_metrics.position_concentration
            }
        )
    
    def _check_loss_risk(self, signal: TradeSignal, account_manager: AccountManager) -> RiskCheckReport:
        """检查损失风险"""
        # 检查当日损失
        if self.risk_metrics.daily_pnl_percent < -self.risk_limits.max_daily_loss_percent:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.HIGH,
                message=f"当日损失超限：{self.risk_metrics.daily_pnl_percent:.2%} < -{self.risk_limits.max_daily_loss_percent:.2%}",
                suggestions=["建议停止交易，等待下一个交易日"],
                metrics={'daily_pnl_percent': self.risk_metrics.daily_pnl_percent}
            )
        
        # 检查总损失
        if self.risk_metrics.total_pnl_percent < -self.risk_limits.max_total_loss_percent:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.CRITICAL,
                message=f"总损失超限：{self.risk_metrics.total_pnl_percent:.2%} < -{self.risk_limits.max_total_loss_percent:.2%}",
                suggestions=["建议停止交易，重新评估策略"],
                metrics={'total_pnl_percent': self.risk_metrics.total_pnl_percent}
            )
        
        # 检查最大回撤
        if self.risk_metrics.current_drawdown_percent > self.risk_limits.max_drawdown_percent:
            return RiskCheckReport(
                result=RiskCheckResult.WARNING,
                risk_level=RiskLevel.MEDIUM,
                message=f"当前回撤过大：{self.risk_metrics.current_drawdown_percent:.2%} > {self.risk_limits.max_drawdown_percent:.2%}",
                suggestions=["建议降低仓位或暂停交易"],
                metrics={'current_drawdown_percent': self.risk_metrics.current_drawdown_percent}
            )
        
        # 通过检查
        return RiskCheckReport(
            result=RiskCheckResult.PASS,
            risk_level=RiskLevel.LOW,
            message="损失风险检查通过"
        )
    
    def _check_trading_frequency_risk(self, signal: TradeSignal) -> RiskCheckReport:
        """检查交易频率风险"""
        # 检查每小时交易次数
        if self.risk_metrics.trades_last_hour >= self.risk_limits.max_trades_per_hour:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.MEDIUM,
                message=f"每小时交易次数过多：{self.risk_metrics.trades_last_hour} >= {self.risk_limits.max_trades_per_hour}",
                suggestions=["建议等待一段时间后再交易"],
                metrics={'trades_last_hour': self.risk_metrics.trades_last_hour}
            )
        
        # 检查每日交易次数
        if self.risk_metrics.trades_today >= self.risk_limits.max_trades_per_day:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.MEDIUM,
                message=f"每日交易次数过多：{self.risk_metrics.trades_today} >= {self.risk_limits.max_trades_per_day}",
                suggestions=["建议明日再交易"],
                metrics={'trades_today': self.risk_metrics.trades_today}
            )
        
        # 检查交易间隔
        if (self.risk_metrics.avg_trade_interval_seconds > 0 and 
            self.risk_metrics.avg_trade_interval_seconds < self.risk_limits.min_trade_interval_seconds):
            return RiskCheckReport(
                result=RiskCheckResult.WARNING,
                risk_level=RiskLevel.LOW,
                message=f"交易间隔过短：{self.risk_metrics.avg_trade_interval_seconds:.0f}s < {self.risk_limits.min_trade_interval_seconds}s",
                suggestions=["建议增加交易间隔"],
                metrics={'avg_trade_interval_seconds': self.risk_metrics.avg_trade_interval_seconds}
            )
        
        # 通过检查
        return RiskCheckReport(
            result=RiskCheckResult.PASS,
            risk_level=RiskLevel.LOW,
            message="交易频率风险检查通过"
        )
    
    def _check_price_risk(self, signal: TradeSignal, current_price: float) -> RiskCheckReport:
        """检查价格风险"""
        if signal.price is None:
            # 市价单无需检查价格偏离
            return RiskCheckReport(
                result=RiskCheckResult.PASS,
                risk_level=RiskLevel.LOW,
                message="市价单价格风险检查通过"
            )
        
        # 检查价格偏离
        price_deviation = abs(signal.price - current_price) / current_price
        
        if price_deviation > self.risk_limits.max_price_deviation_percent:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.MEDIUM,
                message=f"价格偏离过大：{price_deviation:.2%} > {self.risk_limits.max_price_deviation_percent:.2%}",
                suggestions=["建议使用市价单或调整价格"],
                metrics={'price_deviation': price_deviation, 'signal_price': signal.price, 'current_price': current_price}
            )
        
        # 通过检查
        return RiskCheckReport(
            result=RiskCheckResult.PASS,
            risk_level=RiskLevel.LOW,
            message="价格风险检查通过"
        )
    
    def _check_order_value_risk(self, signal: TradeSignal, proposed_quantity: float, current_price: float) -> RiskCheckReport:
        """检查订单价值风险"""
        order_value = proposed_quantity * current_price
        
        # 检查最小订单价值
        if order_value < self.risk_limits.min_order_value_usdt:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.LOW,
                message=f"订单价值过小：{order_value:.2f} < {self.risk_limits.min_order_value_usdt}",
                suggestions=["建议增加订单数量"],
                metrics={'order_value': order_value, 'min_limit': self.risk_limits.min_order_value_usdt}
            )
        
        # 检查最大订单价值
        if order_value > self.risk_limits.max_order_value_usdt:
            return RiskCheckReport(
                result=RiskCheckResult.REJECT,
                risk_level=RiskLevel.HIGH,
                message=f"订单价值过大：{order_value:.2f} > {self.risk_limits.max_order_value_usdt}",
                suggestions=["建议减少订单数量"],
                metrics={'order_value': order_value, 'max_limit': self.risk_limits.max_order_value_usdt}
            )
        
        # 通过检查
        return RiskCheckReport(
            result=RiskCheckResult.PASS,
            risk_level=RiskLevel.LOW,
            message="订单价值风险检查通过"
        )
    
    def _check_market_conditions_risk(self, signal: TradeSignal) -> RiskCheckReport:
        """检查市场条件风险"""
        # 检查市场波动率
        if self.risk_metrics.market_volatility > self.risk_limits.max_volatility_threshold:
            return RiskCheckReport(
                result=RiskCheckResult.WARNING,
                risk_level=RiskLevel.MEDIUM,
                message=f"市场波动率过高：{self.risk_metrics.market_volatility:.2%} > {self.risk_limits.max_volatility_threshold:.2%}",
                suggestions=["建议降低仓位或暂停交易"],
                metrics={'market_volatility': self.risk_metrics.market_volatility}
            )
        
        # 通过检查
        return RiskCheckReport(
            result=RiskCheckResult.PASS,
            risk_level=RiskLevel.LOW,
            message="市场条件风险检查通过"
        )
    
    def _aggregate_risk_checks(self, checks: List[RiskCheckReport]) -> RiskCheckReport:
        """汇总风险检查结果"""
        # 按严重程度排序
        severity_order = [RiskCheckResult.REJECT, RiskCheckResult.WARNING, RiskCheckResult.PASS]
        risk_level_order = [RiskLevel.CRITICAL, RiskLevel.HIGH, RiskLevel.MEDIUM, RiskLevel.LOW]
        
        # 找出最严重的结果
        most_severe_result = RiskCheckResult.PASS
        most_severe_level = RiskLevel.LOW
        
        all_messages = []
        all_suggestions = []
        all_metrics = {}
        
        for check in checks:
            # 更新最严重结果
            if severity_order.index(check.result) < severity_order.index(most_severe_result):
                most_severe_result = check.result
            
            if risk_level_order.index(check.risk_level) < risk_level_order.index(most_severe_level):
                most_severe_level = check.risk_level
            
            # 收集信息
            all_messages.append(check.message)
            all_suggestions.extend(check.suggestions)
            all_metrics.update(check.metrics)
        
        # 生成汇总报告
        summary_message = f"风险检查完成，共{len(checks)}项检查"
        if most_severe_result != RiskCheckResult.PASS:
            failed_checks = [check for check in checks if check.result != RiskCheckResult.PASS]
            summary_message += f"，其中{len(failed_checks)}项未通过"
        
        return RiskCheckReport(
            result=most_severe_result,
            risk_level=most_severe_level,
            message=summary_message,
            suggestions=list(set(all_suggestions)),  # 去重
            metrics=all_metrics
        )
    
    def record_trade(self, symbol: str, side: OrderSide, quantity: float, price: float, strategy_name: str = "") -> None:
        """
        记录交易（用于频率限制）
        
        Args:
            symbol: 交易对符号
            side: 买卖方向
            quantity: 交易数量
            price: 交易价格
            strategy_name: 策略名称
        """
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side.value,
            'quantity': quantity,
            'price': price,
            'strategy_name': strategy_name
        }
        
        self.recent_trades.append(trade_record)
        
        # 保留最近24小时的记录
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.recent_trades = [trade for trade in self.recent_trades if trade['timestamp'] >= cutoff_time]
    
    def set_emergency_stop(self, reason: str) -> None:
        """
        设置紧急停止
        
        Args:
            reason: 停止原因
        """
        self.emergency_stop = True
        self.emergency_reason = reason
        self.current_risk_level = RiskLevel.CRITICAL
        print(f"风险管理器：紧急停止已启用 - {reason}")
    
    def clear_emergency_stop(self) -> None:
        """清除紧急停止"""
        self.emergency_stop = False
        self.emergency_reason = ""
        self.current_risk_level = RiskLevel.LOW
        print("风险管理器：紧急停止已清除")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        获取风险摘要
        
        Returns:
            Dict[str, Any]: 风险摘要信息
        """
        return {
            'current_risk_level': self.current_risk_level.value,
            'emergency_stop': self.emergency_stop,
            'emergency_reason': self.emergency_reason,
            'risk_metrics': self.risk_metrics.to_dict(),
            'risk_limits': self.risk_limits.to_dict(),
            'recent_trades_count': len(self.recent_trades),
            'risk_checks_count': len(self.risk_check_history),
            'last_check_time': self.risk_check_history[-1].to_dict()['timestamp'] if self.risk_check_history else None
        }
    
    def export_risk_report(self) -> Dict[str, Any]:
        """
        导出详细风险报告
        
        Returns:
            Dict[str, Any]: 完整的风险报告
        """
        return {
            'risk_summary': self.get_risk_summary(),
            'recent_risk_checks': [check.to_dict() for check in self.risk_check_history[-10:]],
            'recent_trades': self.recent_trades[-20:] if len(self.recent_trades) > 20 else self.recent_trades,
            'risk_limits': self.risk_limits.to_dict(),
            'report_time': datetime.now().isoformat()
        }