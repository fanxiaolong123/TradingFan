"""
StrategyOrchestrator - 增强版多策略交易编排器

统一管理多个币种和策略实例的核心调度器，支持：
- 多策略并行运行和资源调度
- 智能资金分配和冲突解决
- 动态策略管理和性能优化
- 实时监控和异常恢复
- 跨市场套利和相关性分析
- 策略组合优化和风险平衡
"""

import asyncio
import logging
import threading
import time
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
import numpy as np
from collections import defaultdict, deque

from .data import DataManager
from .broker import SimulatedBroker
from .account import AccountManager
from .risk import RiskManager
from ..strategies.base import Strategy, StrategyConfig, TradeSignal
from ..utils.logger import get_logger
from ..utils.config import get_config


class ResourceType(Enum):
    """资源类型枚举"""
    CAPITAL = "capital"          # 资金
    SYMBOL_ACCESS = "symbol_access"  # 交易对访问权限
    API_CALLS = "api_calls"      # API调用次数
    POSITION_SLOTS = "position_slots"  # 持仓位置


class Priority(Enum):
    """策略优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ResourceAllocation:
    """资源分配信息"""
    strategy_name: str
    resource_type: ResourceType
    allocated_amount: float
    max_amount: float
    usage: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def utilization_rate(self) -> float:
        """资源利用率"""
        if self.allocated_amount == 0:
            return 0.0
        return self.usage / self.allocated_amount


@dataclass
class StrategyMetrics:
    """策略详细指标"""
    name: str
    # 性能指标
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    # 运行指标
    signals_generated: int = 0
    trades_executed: int = 0
    active_positions: int = 0
    last_signal_time: Optional[datetime] = None
    
    # 资源使用
    capital_allocated: float = 0.0
    capital_used: float = 0.0
    api_calls_today: int = 0
    
    # 风险指标
    var_95: float = 0.0  # 95% VaR
    correlation_with_market: float = 0.0
    
    def update_performance(self, returns: List[float]):
        """更新性能指标"""
        if not returns:
            return
        
        returns_array = np.array(returns)
        self.total_return = float(np.prod(1 + returns_array) - 1)
        
        if len(returns_array) > 1:
            self.sharpe_ratio = float(np.mean(returns_array) / np.std(returns_array) * np.sqrt(252))
            
            # 计算最大回撤
            cumulative = np.cumprod(1 + returns_array)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            self.max_drawdown = float(np.min(drawdowns))
            
            # 计算VaR
            self.var_95 = float(np.percentile(returns_array, 5))


@dataclass
class ConflictResolution:
    """冲突解决记录"""
    timestamp: datetime
    conflict_type: str
    strategies_involved: List[str]
    resolution_method: str
    outcome: str
    details: Dict[str, Any] = field(default_factory=dict)


class CapitalAllocator:
    """智能资金分配器"""
    
    def __init__(self, total_capital: float, min_allocation: float = 0.01):
        """
        初始化资金分配器
        
        Args:
            total_capital: 总资金量
            min_allocation: 最小分配比例
        """
        self.total_capital = total_capital
        self.min_allocation = min_allocation
        self.allocations: Dict[str, float] = {}
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.rebalance_frequency = timedelta(hours=4)  # 4小时重新平衡一次
        self.last_rebalance = datetime.now()
    
    def allocate_capital(self, strategies: Dict[str, 'StrategyInstance']) -> Dict[str, float]:
        """
        智能分配资金
        
        Args:
            strategies: 策略实例字典
            
        Returns:
            Dict[str, float]: 每个策略的资金分配
        """
        if not strategies:
            return {}
        
        # 检查是否需要重新平衡
        if datetime.now() - self.last_rebalance < self.rebalance_frequency:
            return self.allocations
        
        # 计算策略权重
        weights = self._calculate_strategy_weights(strategies)
        
        # 分配资金
        new_allocations = {}
        reserved_capital = self.total_capital * 0.1  # 预留10%资金
        available_capital = self.total_capital - reserved_capital
        
        for strategy_name, weight in weights.items():
            allocation = available_capital * weight
            allocation = max(allocation, self.total_capital * self.min_allocation)
            new_allocations[strategy_name] = allocation
        
        # 确保总分配不超过可用资金
        total_allocated = sum(new_allocations.values())
        if total_allocated > available_capital:
            scale_factor = available_capital / total_allocated
            new_allocations = {k: v * scale_factor for k, v in new_allocations.items()}
        
        self.allocations = new_allocations
        self.last_rebalance = datetime.now()
        
        return new_allocations
    
    def _calculate_strategy_weights(self, strategies: Dict[str, 'StrategyInstance']) -> Dict[str, float]:
        """计算策略权重"""
        weights = {}
        
        # 如果是首次分配，使用等权重
        if not any(self.performance_history.values()):
            equal_weight = 1.0 / len(strategies)
            return {name: equal_weight for name in strategies.keys()}
        
        # 基于性能计算权重
        strategy_scores = {}
        
        for name, instance in strategies.items():
            if hasattr(instance, 'metrics') and instance.metrics:
                metrics = instance.metrics
                
                # 综合评分：夏普比率 * 0.4 + 收益率 * 0.3 - 最大回撤 * 0.3
                score = (
                    metrics.sharpe_ratio * 0.4 +
                    metrics.total_return * 0.3 -
                    abs(metrics.max_drawdown) * 0.3
                )
                
                # 胜率加成
                score += metrics.win_rate * 0.1
                
                strategy_scores[name] = max(score, 0.1)  # 最低分0.1
            else:
                strategy_scores[name] = 0.5  # 默认分数
        
        # 归一化权重
        total_score = sum(strategy_scores.values())
        if total_score > 0:
            weights = {name: score / total_score for name, score in strategy_scores.items()}
        else:
            equal_weight = 1.0 / len(strategies)
            weights = {name: equal_weight for name in strategies.keys()}
        
        return weights


class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self):
        self.resolution_history: List[ConflictResolution] = []
        self.strategy_priorities: Dict[str, Priority] = {}
    
    def resolve_symbol_conflict(self, conflicted_signals: List[Tuple[str, TradeSignal]]) -> List[Tuple[str, TradeSignal]]:
        """
        解决同一交易对的信号冲突
        
        Args:
            conflicted_signals: 冲突信号列表，每个元素为(策略名, 信号)
            
        Returns:
            List[Tuple[str, TradeSignal]]: 解决冲突后的信号列表
        """
        if len(conflicted_signals) <= 1:
            return conflicted_signals
        
        symbol = conflicted_signals[0][1].symbol
        
        # 按信号类型分组
        buy_signals = [(name, signal) for name, signal in conflicted_signals 
                      if signal.signal_type.value in ['BUY']]
        sell_signals = [(name, signal) for name, signal in conflicted_signals 
                       if signal.signal_type.value in ['SELL']]
        
        resolved_signals = []
        
        # 处理买入信号冲突
        if len(buy_signals) > 1:
            best_buy = self._select_best_signal(buy_signals, 'BUY')
            if best_buy:
                resolved_signals.append(best_buy)
        elif len(buy_signals) == 1:
            resolved_signals.extend(buy_signals)
        
        # 处理卖出信号冲突
        if len(sell_signals) > 1:
            best_sell = self._select_best_signal(sell_signals, 'SELL')
            if best_sell:
                resolved_signals.append(best_sell)
        elif len(sell_signals) == 1:
            resolved_signals.extend(sell_signals)
        
        # 处理买卖信号对冲
        if buy_signals and sell_signals:
            # 计算净信号强度
            buy_strength = sum(signal.confidence for _, signal in buy_signals)
            sell_strength = sum(signal.confidence for _, signal in sell_signals)
            
            if abs(buy_strength - sell_strength) > 0.3:  # 信号强度差异显著
                if buy_strength > sell_strength:
                    resolved_signals = [self._select_best_signal(buy_signals, 'BUY')]
                else:
                    resolved_signals = [self._select_best_signal(sell_signals, 'SELL')]
            else:
                # 信号强度相近，取消交易
                resolved_signals = []
        
        # 记录冲突解决
        if len(conflicted_signals) != len(resolved_signals):
            resolution = ConflictResolution(
                timestamp=datetime.now(),
                conflict_type="symbol_conflict",
                strategies_involved=[name for name, _ in conflicted_signals],
                resolution_method="priority_and_confidence",
                outcome=f"Reduced from {len(conflicted_signals)} to {len(resolved_signals)} signals",
                details={'symbol': symbol}
            )
            self.resolution_history.append(resolution)
        
        return resolved_signals
    
    def _select_best_signal(self, signals: List[Tuple[str, TradeSignal]], signal_type: str) -> Optional[Tuple[str, TradeSignal]]:
        """根据优先级和置信度选择最佳信号"""
        if not signals:
            return None
        
        def signal_score(strategy_name: str, signal: TradeSignal) -> float:
            priority_score = self.strategy_priorities.get(strategy_name, Priority.MEDIUM).value
            confidence_score = signal.confidence
            return priority_score * 0.6 + confidence_score * 0.4
        
        best_signal = max(signals, key=lambda x: signal_score(x[0], x[1]))
        return best_signal
    
    def set_strategy_priority(self, strategy_name: str, priority: Priority):
        """设置策略优先级"""
        self.strategy_priorities[strategy_name] = priority


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.strategy_returns: Dict[str, List[float]] = defaultdict(list)
        self.correlation_matrix: Optional[np.ndarray] = None
        self.strategy_names: List[str] = []
    
    def update_returns(self, strategy_name: str, return_value: float):
        """更新策略收益"""
        self.strategy_returns[strategy_name].append(return_value)
        
        # 保持最近1000个收益记录
        if len(self.strategy_returns[strategy_name]) > 1000:
            self.strategy_returns[strategy_name] = self.strategy_returns[strategy_name][-1000:]
    
    def calculate_portfolio_correlation(self) -> Optional[np.ndarray]:
        """计算策略组合相关性矩阵"""
        if len(self.strategy_returns) < 2:
            return None
        
        # 获取所有策略的收益序列
        strategy_names = list(self.strategy_returns.keys())
        returns_matrix = []
        
        min_length = min(len(returns) for returns in self.strategy_returns.values() if returns)
        if min_length < 10:  # 至少需要10个数据点
            return None
        
        for name in strategy_names:
            returns = self.strategy_returns[name][-min_length:]
            returns_matrix.append(returns)
        
        returns_matrix = np.array(returns_matrix)
        self.correlation_matrix = np.corrcoef(returns_matrix)
        self.strategy_names = strategy_names
        
        return self.correlation_matrix
    
    def get_diversification_score(self) -> float:
        """计算组合多样化得分"""
        if self.correlation_matrix is None:
            self.calculate_portfolio_correlation()
        
        if self.correlation_matrix is None or len(self.correlation_matrix) < 2:
            return 1.0
        
        # 计算平均相关系数（排除对角线）
        n = len(self.correlation_matrix)
        total_correlation = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_correlation += abs(self.correlation_matrix[i, j])
                count += 1
        
        if count == 0:
            return 1.0
        
        avg_correlation = total_correlation / count
        
        # 多样化得分 = 1 - 平均相关系数
        return max(0.0, 1.0 - avg_correlation)


class EngineState(Enum):
    """引擎状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class StrategyInstance:
    """增强版策略实例信息"""
    name: str
    strategy: Strategy
    config: StrategyConfig
    state: str = "stopped"
    last_update: Optional[datetime] = None
    performance: Dict[str, Any] = None
    error_count: int = 0
    last_error: Optional[str] = None
    
    # 增强字段
    priority: Priority = Priority.MEDIUM
    metrics: Optional[StrategyMetrics] = None
    resource_allocations: Dict[ResourceType, ResourceAllocation] = field(default_factory=dict)
    signal_queue: queue.Queue = field(default_factory=queue.Queue)
    health_score: float = 1.0
    last_health_check: Optional[datetime] = None
    restart_count: int = 0
    max_restarts: int = 3
    
    def __post_init__(self):
        if self.performance is None:
            self.performance = {}
        if self.metrics is None:
            self.metrics = StrategyMetrics(name=self.name)
    
    def update_health_score(self):
        """更新策略健康度评分"""
        score = 1.0
        
        # 基于错误率扣分
        if self.error_count > 0:
            error_penalty = min(0.5, self.error_count * 0.1)
            score -= error_penalty
        
        # 基于重启次数扣分
        if self.restart_count > 0:
            restart_penalty = min(0.3, self.restart_count * 0.1)
            score -= restart_penalty
        
        # 基于最后更新时间扣分
        if self.last_update:
            time_since_update = (datetime.now() - self.last_update).total_seconds()
            if time_since_update > 300:  # 5分钟无更新
                time_penalty = min(0.2, (time_since_update - 300) / 1800)  # 最多扣0.2分
                score -= time_penalty
        
        self.health_score = max(0.0, score)
        self.last_health_check = datetime.now()
        
        return self.health_score
    
    def can_restart(self) -> bool:
        """检查是否可以重启"""
        return self.restart_count < self.max_restarts
    
    def record_restart(self):
        """记录重启"""
        self.restart_count += 1


@dataclass
class EngineMetrics:
    """引擎运行指标"""
    total_strategies: int = 0
    running_strategies: int = 0
    stopped_strategies: int = 0
    error_strategies: int = 0
    total_signals: int = 0
    total_trades: int = 0
    uptime: float = 0.0
    last_update: Optional[datetime] = None


class StrategyOrchestrator:
    """
    增强版多策略交易编排器
    
    负责：
    - 智能策略生命周期管理
    - 动态资源分配和冲突解决
    - 性能监控和自动优化
    - 跨策略协调和组合管理
    - 实时风险控制和异常恢复
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化策略编排器
        
        Args:
            config: 编排器配置，如果为None则从配置文件加载
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__)
        
        # 引擎状态
        self.state = EngineState.STOPPED
        self.start_time: Optional[datetime] = None
        self.stop_event = threading.Event()
        
        # 组件管理
        self.data_manager: Optional[DataManager] = None
        self.broker: Optional[SimulatedBroker] = None
        self.account_manager: Optional[AccountManager] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # 策略管理
        self.strategies: Dict[str, StrategyInstance] = {}
        self.strategy_threads: Dict[str, threading.Thread] = {}
        self.strategy_locks: Dict[str, threading.Lock] = {}
        
        # 增强功能组件
        self.capital_allocator = CapitalAllocator(
            total_capital=self.config.get('trading.initial_capital', 10000.0)
        )
        self.conflict_resolver = ConflictResolver()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # 信号处理
        self.signal_queue: queue.Queue = queue.Queue()
        self.signal_processor_thread: Optional[threading.Thread] = None
        
        # 资源管理
        self.resource_allocations: Dict[str, Dict[ResourceType, ResourceAllocation]] = defaultdict(dict)
        self.symbol_locks: Dict[str, threading.Lock] = {}
        
        # 性能指标
        self.metrics = EngineMetrics()
        self.metrics_lock = threading.Lock()
        
        # 健康监控
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.auto_recovery_enabled = True
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            'strategy_started': [],
            'strategy_stopped': [],
            'strategy_error': [],
            'strategy_recovered': [],
            'signal_generated': [],
            'signal_conflicted': [],
            'signal_resolved': [],
            'trade_executed': [],
            'capital_rebalanced': [],
            'engine_started': [],
            'engine_stopped': [],
            'engine_error': []
        }
        
        self.logger.info("StrategyOrchestrator初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化引擎组件
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info("开始初始化TraderEngine组件...")
            
            # 初始化数据管理器
            self.data_manager = DataManager()
            
            # 初始化账户管理器
            account_config = self.config.get_account_config()
            self.account_manager = AccountManager()
            initial_balance = account_config.get('initial_balance', {'USDT': 10000.0})
            self.account_manager.set_initial_balance(initial_balance)
            
            # 初始化风险管理器
            risk_config = self.config.get_risk_management_config()
            self.risk_manager = RiskManager(risk_config)
            
            # 初始化模拟经纪商
            self.broker = SimulatedBroker(
                initial_balance=initial_balance,
                commission_rate=self.config.get('trading.default_commission_rate', 0.001)
            )
            
            # 从配置加载策略
            self._load_strategies_from_config()
            
            self.logger.info("StrategyOrchestrator组件初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"StrategyOrchestrator初始化失败: {e}")
            self.state = EngineState.ERROR
            return False
    
    def _start_signal_processor(self):
        """启动信号处理器"""
        def process_signals():
            """信号处理主循环"""
            self.logger.info("信号处理器启动")
            
            pending_signals: Dict[str, List[Tuple[str, TradeSignal]]] = defaultdict(list)
            
            while not self.stop_event.is_set():
                try:
                    # 获取信号（带超时）
                    try:
                        strategy_name, signal = self.signal_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    
                    # 按交易对分组信号
                    symbol = signal.symbol
                    pending_signals[symbol].append((strategy_name, signal))
                    
                    # 触发信号生成事件
                    self._trigger_event('signal_generated', {
                        'strategy_name': strategy_name,
                        'signal': signal,
                        'timestamp': datetime.now()
                    })
                    
                    # 检查是否有冲突信号
                    if len(pending_signals[symbol]) > 1:
                        # 触发冲突事件
                        self._trigger_event('signal_conflicted', {
                            'symbol': symbol,
                            'conflicted_signals': pending_signals[symbol],
                            'timestamp': datetime.now()
                        })
                        
                        # 解决冲突
                        resolved_signals = self.conflict_resolver.resolve_symbol_conflict(
                            pending_signals[symbol]
                        )
                        
                        # 触发解决事件
                        self._trigger_event('signal_resolved', {
                            'symbol': symbol,
                            'original_count': len(pending_signals[symbol]),
                            'resolved_count': len(resolved_signals),
                            'resolved_signals': resolved_signals,
                            'timestamp': datetime.now()
                        })
                        
                        # 执行解决后的信号
                        for strategy_name, resolved_signal in resolved_signals:
                            self._execute_signal(strategy_name, resolved_signal)
                        
                        # 清空该交易对的待处理信号
                        pending_signals[symbol].clear()
                    else:
                        # 单一信号，直接执行
                        self._execute_signal(strategy_name, signal)
                        pending_signals[symbol].clear()
                    
                    # 标记任务完成
                    self.signal_queue.task_done()
                    
                except Exception as e:
                    self.logger.error(f"信号处理异常: {e}")
            
            self.logger.info("信号处理器停止")
        
        self.signal_processor_thread = threading.Thread(
            target=process_signals,
            name="SignalProcessor",
            daemon=True
        )
        self.signal_processor_thread.start()
    
    def _execute_signal(self, strategy_name: str, signal: TradeSignal):
        """执行交易信号"""
        try:
            # 获取策略实例
            if strategy_name not in self.strategies:
                self.logger.error(f"策略不存在: {strategy_name}")
                return
            
            strategy_instance = self.strategies[strategy_name]
            
            # 检查资金分配
            capital_allocation = self.capital_allocator.allocations.get(strategy_name, 0)
            if capital_allocation <= 0:
                self.logger.warning(f"策略 {strategy_name} 未分配资金，跳过信号执行")
                return
            
            # 计算交易数量
            if signal.quantity is None:
                # 根据资金分配和信号置信度计算数量
                if signal.quantity_percent:
                    trade_amount = capital_allocation * signal.quantity_percent * signal.confidence
                else:
                    trade_amount = capital_allocation * 0.1 * signal.confidence  # 默认使用10%资金
                
                # 转换为数量（简化计算，实际应该根据当前价格）
                quantity = trade_amount / (signal.price or 50000)  # 假设默认价格
            else:
                quantity = signal.quantity
            
            # 执行订单
            order = self.broker.place_order(signal, quantity, signal.price)
            
            if order:
                # 更新策略指标
                strategy_instance.metrics.signals_generated += 1
                strategy_instance.metrics.last_signal_time = datetime.now()
                
                if order.status.value in ['FILLED', 'PARTIALLY_FILLED']:
                    strategy_instance.metrics.trades_executed += 1
                
                # 触发交易执行事件
                self._trigger_event('trade_executed', {
                    'strategy_name': strategy_name,
                    'signal': signal,
                    'order': order,
                    'timestamp': datetime.now()
                })
                
                self.logger.info(f"信号执行成功: {strategy_name} - {signal.symbol} {signal.signal_type.value}")
            else:
                self.logger.error(f"信号执行失败: {strategy_name} - {signal.symbol}")
                
        except Exception as e:
            self.logger.error(f"执行信号时发生异常: {strategy_name} - {e}")
    
    def _start_health_monitor(self):
        """启动健康监控器"""
        def monitor_health():
            """健康监控主循环"""
            self.logger.info("健康监控器启动")
            
            while not self.stop_event.is_set():
                try:
                    # 检查所有策略健康状态
                    for name, instance in self.strategies.items():
                        health_score = instance.update_health_score()
                        
                        # 如果健康度过低且启用自动恢复
                        if health_score < 0.3 and self.auto_recovery_enabled:
                            if instance.can_restart():
                                self.logger.warning(f"策略 {name} 健康度过低 ({health_score:.2f})，尝试重启")
                                if self._restart_strategy_internal(name):
                                    self._trigger_event('strategy_recovered', {
                                        'strategy_name': name,
                                        'old_health_score': health_score,
                                        'action': 'restart',
                                        'timestamp': datetime.now()
                                    })
                            else:
                                self.logger.error(f"策略 {name} 已达到最大重启次数，停用策略")
                                self._stop_strategy(name)
                    
                    # 检查资金分配是否需要重新平衡
                    new_allocations = self.capital_allocator.allocate_capital(self.strategies)
                    if new_allocations != self.capital_allocator.allocations:
                        self.logger.info("触发资金重新分配")
                        self._trigger_event('capital_rebalanced', {
                            'old_allocations': self.capital_allocator.allocations.copy(),
                            'new_allocations': new_allocations,
                            'timestamp': datetime.now()
                        })
                    
                    # 更新性能分析
                    self.performance_analyzer.calculate_portfolio_correlation()
                    
                    # 等待30秒后再次检查
                    time.sleep(30)
                    
                except Exception as e:
                    self.logger.error(f"健康监控异常: {e}")
                    time.sleep(10)
            
            self.logger.info("健康监控器停止")
        
        self.health_monitor_thread = threading.Thread(
            target=monitor_health,
            name="HealthMonitor",
            daemon=True
        )
        self.health_monitor_thread.start()
    
    def _restart_strategy_internal(self, name: str) -> bool:
        """内部重启策略方法"""
        if name not in self.strategies:
            return False
        
        instance = self.strategies[name]
        
        try:
            # 停止策略
            self._stop_strategy(name)
            time.sleep(2)  # 等待完全停止
            
            # 重新启动
            if self._start_strategy(name):
                instance.record_restart()
                instance.error_count = 0  # 重置错误计数
                instance.last_error = None
                self.logger.info(f"策略 {name} 重启成功")
                return True
            else:
                self.logger.error(f"策略 {name} 重启失败")
                return False
                
        except Exception as e:
            self.logger.error(f"重启策略 {name} 时发生异常: {e}")
            return False
    
    def _load_strategies_from_config(self):
        """从配置文件加载策略"""
        strategies_config = self.config.get('strategies', {})
        
        # 优先使用multi_strategies配置
        multi_strategies = strategies_config.get('multi_strategies', [])
        if multi_strategies:
            self.logger.info("使用multi_strategies配置加载策略")
            for strategy_config in multi_strategies:
                try:
                    self._create_strategy_from_config(strategy_config)
                except Exception as e:
                    self.logger.error(f"加载策略配置失败: {strategy_config.get('name', 'unknown')}, 错误: {e}")
        else:
            # 兼容原有配置格式
            self.logger.info("使用传统配置格式加载策略")
            # 如果strategies是字典格式，转换为列表格式
            if isinstance(strategies_config, dict):
                strategies_list = []
                for key, value in strategies_config.items():
                    if isinstance(value, dict) and key not in ['max_active_strategies', 'strategy_timeout']:
                        value['name'] = key
                        strategies_list.append(value)
                strategies_config = strategies_list
            
            # 如果strategies是列表格式，直接处理
            if isinstance(strategies_config, list):
                for strategy_config in strategies_config:
                    try:
                        self._create_strategy_from_config(strategy_config)
                    except Exception as e:
                        self.logger.error(f"加载策略配置失败: {strategy_config.get('name', 'unknown')}, 错误: {e}")
        
        self.logger.info(f"从配置加载了 {len(self.strategies)} 个策略")
    
    def _create_strategy_from_config(self, config: Dict):
        """根据配置创建策略实例"""
        from ..strategies.mean_reversion import MeanReversionStrategy
        
        # 策略类映射
        strategy_classes = {
            'MeanReversion': MeanReversionStrategy,
            'MeanReversionStrategy': MeanReversionStrategy
        }
        
        name = config.get('name')
        strategy_class_name = config.get('class', 'MeanReversion')
        
        if not name:
            raise ValueError("策略配置必须包含name字段")
        
        if strategy_class_name not in strategy_classes:
            raise ValueError(f"未知的策略类: {strategy_class_name}")
        
        # 合并参数配置
        parameters = config.get('parameters', {})
        risk_management = config.get('risk_management', {})
        
        # 将风险管理配置合并到parameters中
        if risk_management:
            parameters.update({
                'max_position_percent': risk_management.get('max_position_percent', 0.1),
                'stop_loss_percent': risk_management.get('stop_loss_percent', 0.02),
                'take_profit_percent': risk_management.get('take_profit_percent', 0.04)
            })
        
        # 创建策略配置
        strategy_config = StrategyConfig(
            name=name,
            symbol=config.get('symbol', 'BTCUSDT'),
            timeframe=config.get('interval', config.get('timeframe', '1h')),
            parameters=parameters
        )
        
        # 创建策略实例
        strategy_class = strategy_classes[strategy_class_name]
        strategy = strategy_class(strategy_config)
        
        # 创建策略实例信息
        instance = StrategyInstance(
            name=name,
            strategy=strategy,
            config=strategy_config,
            state="loaded"
        )
        
        self.strategies[name] = instance
        self.strategy_locks[name] = threading.Lock()
        
        self.logger.info(f"创建策略实例: {name} ({strategy_class_name}) - {config.get('symbol', 'BTCUSDT')}")
    
    def start(self) -> bool:
        """
        启动交易引擎
        
        Returns:
            bool: 启动是否成功
        """
        if self.state != EngineState.STOPPED:
            self.logger.warning(f"引擎已在运行中，当前状态: {self.state}")
            return False
        
        try:
            self.logger.info("正在启动TraderEngine...")
            self.state = EngineState.STARTING
            
            # 初始化组件
            if not self.initialize():
                self.state = EngineState.ERROR
                return False
            
            # 重置停止事件
            self.stop_event.clear()
            
            # 记录启动时间
            self.start_time = datetime.now()
            
            # 启动所有策略
            for name, instance in self.strategies.items():
                if self._start_strategy(name):
                    self.logger.info(f"策略启动成功: {name}")
                else:
                    self.logger.error(f"策略启动失败: {name}")
            
            # 启动增强功能组件
            self._start_signal_processor()
            self._start_health_monitor()
            self._start_monitoring()
            
            self.state = EngineState.RUNNING
            self.logger.info("StrategyOrchestrator启动成功")
            
            # 触发启动事件
            self._trigger_event('engine_started', {
                'timestamp': datetime.now(),
                'strategies_count': len(self.strategies)
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"TraderEngine启动失败: {e}")
            self.state = EngineState.ERROR
            return False
    
    def stop(self) -> bool:
        """
        停止交易引擎
        
        Returns:
            bool: 停止是否成功
        """
        if self.state not in [EngineState.RUNNING, EngineState.ERROR]:
            self.logger.warning(f"引擎未运行，当前状态: {self.state}")
            return False
        
        try:
            self.logger.info("正在停止TraderEngine...")
            self.state = EngineState.STOPPING
            
            # 设置停止事件
            self.stop_event.set()
            
            # 停止所有策略
            for name in list(self.strategies.keys()):
                self._stop_strategy(name)
            
            # 等待所有策略线程结束
            for name, thread in self.strategy_threads.items():
                if thread.is_alive():
                    thread.join(timeout=5)
                    if thread.is_alive():
                        self.logger.warning(f"策略线程未能正常结束: {name}")
            
            self.strategy_threads.clear()
            
            self.state = EngineState.STOPPED
            self.logger.info("TraderEngine停止成功")
            
            # 触发停止事件
            self._trigger_event('engine_stopped', {
                'timestamp': datetime.now(),
                'uptime': self.get_uptime()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"TraderEngine停止失败: {e}")
            self.state = EngineState.ERROR
            return False
    
    def _start_strategy(self, name: str) -> bool:
        """启动单个策略"""
        if name not in self.strategies:
            self.logger.error(f"策略不存在: {name}")
            return False
        
        instance = self.strategies[name]
        
        if instance.state == "running":
            self.logger.warning(f"策略已在运行: {name}")
            return True
        
        try:
            # 初始化策略
            instance.strategy.initialize()
            
            # 创建策略运行线程
            thread = threading.Thread(
                target=self._run_strategy,
                args=(name,),
                name=f"Strategy-{name}",
                daemon=True
            )
            
            self.strategy_threads[name] = thread
            thread.start()
            
            # 更新状态
            instance.state = "running"
            instance.last_update = datetime.now()
            
            self.logger.info(f"策略启动成功: {name}")
            
            # 触发策略启动事件
            self._trigger_event('strategy_started', {
                'name': name,
                'timestamp': datetime.now()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"策略启动失败: {name}, 错误: {e}")
            instance.state = "error"
            instance.error_count += 1
            instance.last_error = str(e)
            
            # 触发策略错误事件
            self._trigger_event('strategy_error', {
                'name': name,
                'error': str(e),
                'timestamp': datetime.now()
            })
            
            return False
    
    def _stop_strategy(self, name: str) -> bool:
        """停止单个策略"""
        if name not in self.strategies:
            self.logger.error(f"策略不存在: {name}")
            return False
        
        instance = self.strategies[name]
        
        if instance.state == "stopped":
            self.logger.warning(f"策略已停止: {name}")
            return True
        
        try:
            # 更新状态
            instance.state = "stopping"
            
            # 等待策略线程结束
            if name in self.strategy_threads:
                thread = self.strategy_threads[name]
                if thread.is_alive():
                    thread.join(timeout=3)
                    if thread.is_alive():
                        self.logger.warning(f"策略线程未能及时结束: {name}")
                
                del self.strategy_threads[name]
            
            # 更新状态
            instance.state = "stopped"
            instance.last_update = datetime.now()
            
            self.logger.info(f"策略停止成功: {name}")
            
            # 触发策略停止事件
            self._trigger_event('strategy_stopped', {
                'name': name,
                'timestamp': datetime.now()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"策略停止失败: {name}, 错误: {e}")
            instance.state = "error"
            instance.error_count += 1
            instance.last_error = str(e)
            return False
    
    def _run_strategy(self, name: str):
        """策略运行主循环"""
        instance = self.strategies[name]
        strategy = instance.strategy
        
        self.logger.info(f"策略开始运行: {name}")
        
        while not self.stop_event.is_set() and instance.state == "running":
            try:
                # 获取数据并处理
                # 这里应该根据策略配置获取对应的市场数据
                # 为了演示，我们先使用模拟数据
                
                # 模拟数据处理延迟
                time.sleep(1)
                
                # 更新最后更新时间
                instance.last_update = datetime.now()
                
                # 更新性能指标
                instance.performance = strategy.get_performance_metrics()
                
            except Exception as e:
                self.logger.error(f"策略运行异常: {name}, 错误: {e}")
                instance.state = "error"
                instance.error_count += 1
                instance.last_error = str(e)
                
                # 触发策略错误事件
                self._trigger_event('strategy_error', {
                    'name': name,
                    'error': str(e),
                    'timestamp': datetime.now()
                })
                break
        
        self.logger.info(f"策略运行结束: {name}")
    
    def _start_monitoring(self):
        """启动监控线程"""
        def monitor():
            while not self.stop_event.is_set() and self.state == EngineState.RUNNING:
                try:
                    self._update_metrics()
                    time.sleep(5)  # 每5秒更新一次指标
                except Exception as e:
                    self.logger.error(f"监控线程异常: {e}")
        
        monitor_thread = threading.Thread(
            target=monitor,
            name="EngineMonitor",
            daemon=True
        )
        monitor_thread.start()
    
    def _update_metrics(self):
        """更新引擎指标"""
        with self.metrics_lock:
            self.metrics.total_strategies = len(self.strategies)
            self.metrics.running_strategies = sum(1 for s in self.strategies.values() if s.state == "running")
            self.metrics.stopped_strategies = sum(1 for s in self.strategies.values() if s.state == "stopped")
            self.metrics.error_strategies = sum(1 for s in self.strategies.values() if s.state == "error")
            self.metrics.uptime = self.get_uptime()
            self.metrics.last_update = datetime.now()
    
    def _trigger_event(self, event_type: str, data: Dict[str, Any]):
        """触发事件回调"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"事件回调异常: {event_type}, 错误: {e}")
    
    def add_event_callback(self, event_type: str, callback: Callable):
        """添加事件回调"""
        if event_type not in self.event_callbacks:
            self.event_callbacks[event_type] = []
        self.event_callbacks[event_type].append(callback)
    
    def get_uptime(self) -> float:
        """获取运行时间（秒）"""
        if self.start_time:
            return (datetime.now() - self.start_time).total_seconds()
        return 0.0
    
    def get_strategies_status(self) -> Dict[str, Dict]:
        """获取所有策略状态"""
        status = {}
        for name, instance in self.strategies.items():
            status[name] = {
                'name': name,
                'state': instance.state,
                'symbol': instance.config.symbol,
                'timeframe': instance.config.timeframe,
                'last_update': instance.last_update.isoformat() if instance.last_update else None,
                'performance': instance.performance,
                'error_count': instance.error_count,
                'last_error': instance.last_error
            }
        return status
    
    def get_engine_status(self) -> Dict[str, Any]:
        """获取引擎状态"""
        return {
            'state': self.state.value,
            'uptime': self.get_uptime(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'metrics': {
                'total_strategies': self.metrics.total_strategies,
                'running_strategies': self.metrics.running_strategies,
                'stopped_strategies': self.metrics.stopped_strategies,
                'error_strategies': self.metrics.error_strategies,
                'total_signals': self.metrics.total_signals,
                'total_trades': self.metrics.total_trades,
                'last_update': self.metrics.last_update.isoformat() if self.metrics.last_update else None
            }
        }
    
    def restart_strategy(self, name: str) -> bool:
        """重启策略"""
        if name not in self.strategies:
            self.logger.error(f"策略不存在: {name}")
            return False
        
        self.logger.info(f"重启策略: {name}")
        
        # 先停止策略
        self._stop_strategy(name)
        
        # 等待一秒确保完全停止
        time.sleep(1)
        
        # 重新启动策略
        return self._start_strategy(name)
    
    def submit_signal(self, strategy_name: str, signal: TradeSignal) -> bool:
        """
        提交交易信号到信号队列
        
        Args:
            strategy_name: 策略名称
            signal: 交易信号
            
        Returns:
            bool: 是否成功提交
        """
        try:
            if strategy_name not in self.strategies:
                self.logger.error(f"策略不存在: {strategy_name}")
                return False
            
            self.signal_queue.put((strategy_name, signal))
            return True
            
        except Exception as e:
            self.logger.error(f"提交信号失败: {e}")
            return False
    
    def set_strategy_priority(self, strategy_name: str, priority: Priority) -> bool:
        """
        设置策略优先级
        
        Args:
            strategy_name: 策略名称
            priority: 优先级
            
        Returns:
            bool: 是否设置成功
        """
        if strategy_name not in self.strategies:
            self.logger.error(f"策略不存在: {strategy_name}")
            return False
        
        self.strategies[strategy_name].priority = priority
        self.conflict_resolver.set_strategy_priority(strategy_name, priority)
        
        self.logger.info(f"策略 {strategy_name} 优先级设置为 {priority.name}")
        return True
    
    def get_portfolio_correlation(self) -> Optional[np.ndarray]:
        """获取策略组合相关性矩阵"""
        return self.performance_analyzer.calculate_portfolio_correlation()
    
    def get_diversification_score(self) -> float:
        """获取组合多样化得分"""
        return self.performance_analyzer.get_diversification_score()
    
    def get_capital_allocations(self) -> Dict[str, float]:
        """获取当前资金分配"""
        return self.capital_allocator.allocations.copy()
    
    def force_capital_rebalance(self) -> Dict[str, float]:
        """强制重新分配资金"""
        self.capital_allocator.last_rebalance = datetime.now() - timedelta(hours=5)
        return self.capital_allocator.allocate_capital(self.strategies)
    
    def get_conflict_history(self) -> List[ConflictResolution]:
        """获取冲突解决历史"""
        return self.conflict_resolver.resolution_history.copy()
    
    def get_orchestrator_stats(self) -> Dict[str, Any]:
        """获取编排器详细统计信息"""
        stats = {
            'engine_status': self.get_engine_status(),
            'strategies_status': self.get_strategies_status(),
            'capital_allocations': self.get_capital_allocations(),
            'diversification_score': self.get_diversification_score(),
            'signal_queue_size': self.signal_queue.qsize(),
            'conflict_resolutions_count': len(self.conflict_resolver.resolution_history),
            'auto_recovery_enabled': self.auto_recovery_enabled,
            'total_capital': self.capital_allocator.total_capital
        }
        
        # 健康度统计
        health_scores = []
        for instance in self.strategies.values():
            if instance.health_score is not None:
                health_scores.append(instance.health_score)
        
        if health_scores:
            stats['average_health_score'] = sum(health_scores) / len(health_scores)
            stats['min_health_score'] = min(health_scores)
        else:
            stats['average_health_score'] = 1.0
            stats['min_health_score'] = 1.0
        
        # 策略优先级分布
        priority_distribution = {}
        for instance in self.strategies.values():
            priority = instance.priority.name
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        stats['priority_distribution'] = priority_distribution
        
        return stats
    
    def enable_auto_recovery(self, enabled: bool = True):
        """启用/禁用自动恢复"""
        self.auto_recovery_enabled = enabled
        self.logger.info(f"自动恢复已{'启用' if enabled else '禁用'}")
    
    def get_strategy_metrics(self, strategy_name: str) -> Optional[StrategyMetrics]:
        """获取指定策略的详细指标"""
        if strategy_name not in self.strategies:
            return None
        
        return self.strategies[strategy_name].metrics
    
    def update_strategy_performance(self, strategy_name: str, return_value: float):
        """更新策略性能数据"""
        if strategy_name in self.strategies:
            self.performance_analyzer.update_returns(strategy_name, return_value)


    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


# 保持向后兼容
TraderEngine = StrategyOrchestrator