"""
TraderEngine - 多策略交易引擎

统一管理多个币种和策略实例的核心调度器，支持：
- 多策略并行运行
- 资源协调和风险管理
- 动态策略管理
- 性能监控和状态管理
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import concurrent.futures

from .data import DataManager
from .broker import SimulatedBroker
from .account import AccountManager
from .risk import RiskManager
from ..strategies.base import Strategy, StrategyConfig
from ..utils.logger import get_logger
from ..utils.config import get_config


class EngineState(Enum):
    """引擎状态枚举"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class StrategyInstance:
    """策略实例信息"""
    name: str
    strategy: Strategy
    config: StrategyConfig
    state: str = "stopped"
    last_update: Optional[datetime] = None
    performance: Dict[str, Any] = None
    error_count: int = 0
    last_error: Optional[str] = None
    
    def __post_init__(self):
        if self.performance is None:
            self.performance = {}


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


class TraderEngine:
    """
    多策略交易引擎
    
    负责：
    - 策略生命周期管理
    - 数据分发和处理
    - 资源协调和冲突解决
    - 性能监控和异常处理
    - 统一的日志和报警
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化交易引擎
        
        Args:
            config: 引擎配置，如果为None则从配置文件加载
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
        
        # 性能指标
        self.metrics = EngineMetrics()
        self.metrics_lock = threading.Lock()
        
        # 事件回调
        self.event_callbacks: Dict[str, List[Callable]] = {
            'strategy_started': [],
            'strategy_stopped': [],
            'strategy_error': [],
            'signal_generated': [],
            'trade_executed': [],
            'engine_started': [],
            'engine_stopped': [],
            'engine_error': []
        }
        
        self.logger.info("TraderEngine初始化完成")
    
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
            
            self.logger.info("TraderEngine组件初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"TraderEngine初始化失败: {e}")
            self.state = EngineState.ERROR
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
            
            # 启动监控线程
            self._start_monitoring()
            
            self.state = EngineState.RUNNING
            self.logger.info("TraderEngine启动成功")
            
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
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()