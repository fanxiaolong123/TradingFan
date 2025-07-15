"""
经纪商模块 - 负责订单执行和交易管理

这个模块提供了统一的交易接口，支持：
- 订单下单、撤单、查询
- 仓位管理和账户信息
- 多种交易所支持（Binance、模拟交易等）
- 订单状态实时跟踪
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import asyncio
import time
import logging
from threading import Lock
import json

# 可选导入，避免缺少依赖时导入失败
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceOrderException
    BINANCE_AVAILABLE = True
except ImportError:
    Client = None
    BinanceAPIException = Exception
    BinanceOrderException = Exception
    BINANCE_AVAILABLE = False

from ..strategies.base import TradeSignal, OrderType, SignalType


class RateLimiter:
    """
    API限频器 - 控制API请求频率
    
    币安API限制：
    - 下单：10次/秒
    - 查询：20次/秒
    - WebSocket连接：5次/秒
    """
    
    def __init__(self, max_requests: int = 10, time_window: int = 1):
        """
        初始化限频器
        
        Args:
            max_requests: 时间窗口内最大请求数
            time_window: 时间窗口长度（秒）
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []  # 存储请求时间戳
        self.lock = Lock()
    
    def can_request(self) -> bool:
        """检查是否可以发送请求"""
        with self.lock:
            current_time = time.time()
            # 清理过期的请求记录
            self.requests = [req_time for req_time in self.requests 
                           if current_time - req_time < self.time_window]
            return len(self.requests) < self.max_requests
    
    def add_request(self) -> None:
        """添加请求记录"""
        with self.lock:
            self.requests.append(time.time())
    
    def wait_if_needed(self) -> float:
        """如果需要等待，返回等待时间"""
        with self.lock:
            if not self.can_request():
                current_time = time.time()
                oldest_request = min(self.requests)
                wait_time = self.time_window - (current_time - oldest_request)
                return max(0, wait_time)
            return 0


class RetryConfig:
    """重试配置"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, 
                 backoff_factor: float = 2.0, max_delay: float = 60.0):
        """
        初始化重试配置
        
        Args:
            max_retries: 最大重试次数
            retry_delay: 初始重试延迟（秒）
            backoff_factor: 退避因子
            max_delay: 最大延迟时间（秒）
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        """获取重试延迟时间"""
        delay = self.retry_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


class OrderConfirmation:
    """订单确认机制"""
    
    def __init__(self, broker_instance, order: 'Order'):
        """
        初始化订单确认
        
        Args:
            broker_instance: 经纪商实例
            order: 订单对象
        """
        self.broker = broker_instance
        self.order = order
        self.confirmed = False
        self.confirmation_time = None
        self.error_message = None
    
    def confirm(self, timeout: int = 30) -> bool:
        """
        确认订单状态
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            bool: 是否确认成功
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # 查询订单最新状态
                updated_order = self.broker.get_order(self.order.order_id)
                if updated_order:
                    self.order = updated_order
                    
                    # 检查订单状态
                    if updated_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                        self.confirmed = True
                        self.confirmation_time = datetime.now()
                        return True
                    elif updated_order.status in [OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                        self.error_message = f"订单状态异常: {updated_order.status.value}"
                        return False
                
                # 等待后重试
                time.sleep(1)
                
            except Exception as e:
                self.error_message = f"确认订单时发生错误: {str(e)}"
                return False
        
        self.error_message = "订单确认超时"
        return False


class AccountValidator:
    """账户状态校验器"""
    
    def __init__(self, broker_instance):
        """
        初始化账户校验器
        
        Args:
            broker_instance: 经纪商实例
        """
        self.broker = broker_instance
        self.last_validation_time = None
        self.validation_interval = 60  # 校验间隔（秒）
    
    def validate_account(self) -> Dict[str, Any]:
        """
        校验账户状态
        
        Returns:
            Dict[str, Any]: 校验结果
        """
        result = {
            'valid': False,
            'balance_sufficient': False,
            'api_connected': False,
            'trading_enabled': False,
            'errors': []
        }
        
        try:
            # 检查API连接
            if hasattr(self.broker, 'client'):
                self.broker.client.ping()
                result['api_connected'] = True
            
            # 检查账户信息
            if hasattr(self.broker, 'client'):
                account_info = self.broker.client.get_account()
                result['trading_enabled'] = account_info.get('canTrade', False)
                
                # 检查余额
                balances = self.broker.get_account_balance()
                result['balance_sufficient'] = len(balances) > 0
                
                result['valid'] = True
                self.last_validation_time = datetime.now()
                
        except Exception as e:
            result['errors'].append(f"账户校验失败: {str(e)}")
        
        return result
    
    def is_validation_needed(self) -> bool:
        """检查是否需要重新校验"""
        if self.last_validation_time is None:
            return True
        
        time_since_last = (datetime.now() - self.last_validation_time).total_seconds()
        return time_since_last > self.validation_interval


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "PENDING"          # 待处理
    NEW = "NEW"                  # 新建
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # 部分成交
    FILLED = "FILLED"            # 完全成交
    CANCELED = "CANCELED"        # 已取消
    REJECTED = "REJECTED"        # 被拒绝
    EXPIRED = "EXPIRED"          # 已过期


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "BUY"                  # 买入
    SELL = "SELL"                # 卖出


@dataclass
class Order:
    """订单数据结构"""
    order_id: str                # 订单ID
    symbol: str                  # 交易对符号
    side: OrderSide              # 买卖方向
    order_type: OrderType        # 订单类型
    quantity: float              # 委托数量
    price: Optional[float]       # 委托价格（市价单为None）
    
    # 订单状态
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0  # 已成交数量
    remaining_quantity: float = 0.0  # 剩余数量
    avg_price: float = 0.0       # 平均成交价格
    
    # 时间信息
    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    
    # 费用信息
    commission: float = 0.0      # 手续费
    commission_asset: str = ""   # 手续费币种
    
    # 额外信息
    client_order_id: str = ""    # 客户端订单ID
    strategy_name: str = ""      # 策略名称
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.client_order_id:
            self.client_order_id = f"auto_trader_{uuid.uuid4().hex[:8]}"
        self.remaining_quantity = self.quantity
    
    def update_fill(self, filled_qty: float, price: float, commission: float = 0.0) -> None:
        """
        更新订单成交信息
        
        Args:
            filled_qty: 成交数量
            price: 成交价格
            commission: 手续费
        """
        self.filled_quantity += filled_qty
        self.remaining_quantity = self.quantity - self.filled_quantity
        
        # 更新平均成交价格
        if self.filled_quantity > 0:
            total_cost = self.avg_price * (self.filled_quantity - filled_qty) + price * filled_qty
            self.avg_price = total_cost / self.filled_quantity
        
        # 更新手续费
        self.commission += commission
        
        # 更新状态
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
        
        self.update_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'avg_price': self.avg_price,
            'create_time': self.create_time.isoformat(),
            'update_time': self.update_time.isoformat(),
            'commission': self.commission,
            'commission_asset': self.commission_asset,
            'client_order_id': self.client_order_id,
            'strategy_name': self.strategy_name,
            'metadata': self.metadata
        }


@dataclass
class Position:
    """持仓数据结构"""
    symbol: str                  # 交易对符号
    quantity: float              # 持仓数量（正数为多仓，负数为空仓）
    avg_price: float             # 平均成本价
    unrealized_pnl: float = 0.0  # 未实现盈亏
    realized_pnl: float = 0.0    # 已实现盈亏
    
    # 时间信息
    open_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    
    # 额外信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_position(self, quantity_delta: float, price: float) -> None:
        """
        更新持仓
        
        Args:
            quantity_delta: 数量变化（正数为增加，负数为减少）
            price: 成交价格
        """
        if self.quantity == 0:
            # 新开仓
            self.quantity = quantity_delta
            self.avg_price = price
            self.open_time = datetime.now()
        else:
            # 更新仓位
            if (self.quantity > 0 and quantity_delta > 0) or (self.quantity < 0 and quantity_delta < 0):
                # 同向加仓
                total_cost = self.avg_price * abs(self.quantity) + price * abs(quantity_delta)
                self.quantity += quantity_delta
                self.avg_price = total_cost / abs(self.quantity)
            else:
                # 反向减仓或平仓
                if abs(quantity_delta) >= abs(self.quantity):
                    # 完全平仓或反向开仓
                    self.realized_pnl += (price - self.avg_price) * abs(self.quantity) * (1 if self.quantity > 0 else -1)
                    remaining_qty = abs(quantity_delta) - abs(self.quantity)
                    
                    if remaining_qty > 0:
                        # 反向开仓
                        self.quantity = remaining_qty * (-1 if self.quantity > 0 else 1)
                        self.avg_price = price
                        self.open_time = datetime.now()
                    else:
                        # 完全平仓
                        self.quantity = 0
                        self.avg_price = 0
                else:
                    # 部分平仓
                    self.realized_pnl += (price - self.avg_price) * abs(quantity_delta) * (1 if self.quantity > 0 else -1)
                    self.quantity += quantity_delta
        
        self.update_time = datetime.now()
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        计算未实现盈亏
        
        Args:
            current_price: 当前价格
            
        Returns:
            float: 未实现盈亏
        """
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
        else:
            self.unrealized_pnl = (current_price - self.avg_price) * self.quantity
        
        return self.unrealized_pnl
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'open_time': self.open_time.isoformat(),
            'update_time': self.update_time.isoformat(),
            'metadata': self.metadata
        }


class Broker(ABC):
    """经纪商抽象基类"""
    
    @abstractmethod
    def place_order(self, signal: TradeSignal, quantity: float, price: Optional[float] = None) -> Optional[Order]:
        """
        下单
        
        Args:
            signal: 交易信号
            quantity: 委托数量
            price: 委托价格（市价单为None）
            
        Returns:
            Order: 订单对象
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        撤单
        
        Args:
            order_id: 订单ID
            
        Returns:
            bool: 是否成功
        """
        pass
    
    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """
        查询订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            Order: 订单对象
        """
        pass
    
    @abstractmethod
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        获取未成交订单
        
        Args:
            symbol: 交易对符号，None表示所有
            
        Returns:
            List[Order]: 订单列表
        """
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        获取持仓
        
        Args:
            symbol: 交易对符号
            
        Returns:
            Position: 持仓对象
        """
        pass
    
    @abstractmethod
    def get_account_balance(self) -> Dict[str, float]:
        """
        获取账户余额
        
        Returns:
            Dict[str, float]: 币种余额字典
        """
        pass


class BinanceBroker(Broker):
    """币安经纪商实现 - 增强版，支持重试、限频、订单确认和账户校验"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False,
                 enable_rate_limiting: bool = True, enable_order_confirmation: bool = True,
                 retry_config: Optional[RetryConfig] = None):
        """
        初始化币安经纪商
        
        Args:
            api_key: API密钥
            api_secret: API密钥
            testnet: 是否使用测试网
            enable_rate_limiting: 是否启用限频
            enable_order_confirmation: 是否启用订单确认
            retry_config: 重试配置
        """
        if not BINANCE_AVAILABLE:
            raise ImportError("python-binance库未安装，无法使用BinanceBroker")
            
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_order_confirmation = enable_order_confirmation
        
        # 初始化币安客户端
        self.client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)
        
        # 内部数据
        self.orders: Dict[str, Order] = {}              # 订单缓存
        self.positions: Dict[str, Position] = {}        # 持仓缓存
        self.order_id_map: Dict[str, str] = {}          # 订单ID映射（内部ID -> 交易所ID）
        
        # 增强功能模块
        self.rate_limiter = RateLimiter(max_requests=8, time_window=1) if enable_rate_limiting else None
        self.query_rate_limiter = RateLimiter(max_requests=15, time_window=1) if enable_rate_limiting else None
        self.retry_config = retry_config or RetryConfig()
        self.account_validator = AccountValidator(self)
        
        # 日志配置
        self.logger = logging.getLogger(f"{__name__}.BinanceBroker")
        
        # 测试连接
        try:
            self.client.ping()
            self.logger.info(f"币安经纪商初始化成功 (testnet: {testnet})")
            print(f"币安经纪商初始化成功 (testnet: {testnet})")
            
            # 初始账户校验
            validation_result = self.account_validator.validate_account()
            if not validation_result['valid']:
                self.logger.warning(f"账户校验警告: {validation_result['errors']}")
            
        except Exception as e:
            self.logger.error(f"币安经纪商初始化失败: {e}")
            print(f"币安经纪商初始化失败: {e}")
            raise
    
    def _wait_for_rate_limit(self, limiter: Optional[RateLimiter]) -> None:
        """等待限频"""
        if limiter:
            wait_time = limiter.wait_if_needed()
            if wait_time > 0:
                self.logger.info(f"触发限频，等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
            limiter.add_request()
    
    def _execute_with_retry(self, func, *args, **kwargs):
        """
        带重试机制的执行函数
        
        Args:
            func: 要执行的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            函数执行结果
        """
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return func(*args, **kwargs)
                
            except (BinanceAPIException, BinanceOrderException) as e:
                last_exception = e
                
                # 某些错误不需要重试
                if hasattr(e, 'code'):
                    # 账户余额不足、参数错误等不需要重试
                    if e.code in [-2010, -1013, -1021, -1022]:
                        self.logger.error(f"不可重试的错误: {e}")
                        raise e
                
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    self.logger.warning(f"第 {attempt + 1} 次尝试失败: {e}, {delay:.2f}秒后重试")
                    time.sleep(delay)
                else:
                    self.logger.error(f"达到最大重试次数，最后错误: {e}")
                    
            except Exception as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.get_delay(attempt)
                    self.logger.warning(f"第 {attempt + 1} 次尝试失败: {e}, {delay:.2f}秒后重试")
                    time.sleep(delay)
                else:
                    self.logger.error(f"达到最大重试次数，最后错误: {e}")
        
        raise last_exception
    
    def _validate_order_params(self, signal: TradeSignal, quantity: float, price: Optional[float]) -> bool:
        """
        校验订单参数
        
        Args:
            signal: 交易信号
            quantity: 数量
            price: 价格
            
        Returns:
            bool: 参数是否有效
        """
        # 基本参数校验
        if quantity <= 0:
            self.logger.error("订单数量必须大于0")
            return False
        
        if signal.order_type == OrderType.LIMIT and (price is None or price <= 0):
            self.logger.error("限价单必须指定有效价格")
            return False
        
        # 账户状态校验
        if self.account_validator.is_validation_needed():
            validation_result = self.account_validator.validate_account()
            if not validation_result['valid']:
                self.logger.error(f"账户状态校验失败: {validation_result['errors']}")
                return False
            
            if not validation_result['trading_enabled']:
                self.logger.error("账户交易功能未启用")
                return False
        
        return True
    
    def place_order(self, signal: TradeSignal, quantity: float, price: Optional[float] = None) -> Optional[Order]:
        """增强版下单方法 - 支持重试、限频、参数校验和订单确认"""
        
        # 参数校验
        if not self._validate_order_params(signal, quantity, price):
            return None
        
        # 等待限频
        self._wait_for_rate_limit(self.rate_limiter)
        
        try:
            # 创建订单对象
            order = Order(
                order_id=str(uuid.uuid4()),
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                order_type=signal.order_type,
                quantity=quantity,
                price=price,
                strategy_name=signal.strategy_name,
                metadata=signal.metadata
            )
            
            # 准备币安API参数
            binance_params = {
                'symbol': signal.symbol,
                'side': order.side.value,
                'quantity': quantity,
                'newClientOrderId': order.client_order_id
            }
            
            # 根据订单类型设置参数
            if signal.order_type == OrderType.MARKET:
                binance_params['type'] = 'MARKET'
            elif signal.order_type == OrderType.LIMIT:
                binance_params['type'] = 'LIMIT'
                binance_params['price'] = str(price)
                binance_params['timeInForce'] = 'GTC'  # Good Till Canceled
            elif signal.order_type == OrderType.STOP_LOSS:
                binance_params['type'] = 'STOP_LOSS_LIMIT'
                binance_params['price'] = str(price)
                binance_params['stopPrice'] = str(signal.stop_loss)
                binance_params['timeInForce'] = 'GTC'
            elif signal.order_type == OrderType.TAKE_PROFIT:
                binance_params['type'] = 'TAKE_PROFIT_LIMIT'
                binance_params['price'] = str(price)
                binance_params['stopPrice'] = str(signal.take_profit)
                binance_params['timeInForce'] = 'GTC'
            
            # 使用重试机制发送订单
            def _place_order_internal():
                return self.client.create_order(**binance_params)
            
            result = self._execute_with_retry(_place_order_internal)
            
            # 更新订单信息
            order.order_id = result['orderId']
            order.status = OrderStatus(result['status'])
            order.filled_quantity = float(result.get('executedQty', 0))
            order.remaining_quantity = quantity - order.filled_quantity
            
            # 如果有成交，更新平均价格
            if order.filled_quantity > 0:
                order.avg_price = float(result.get('cummulativeQuoteQty', 0)) / order.filled_quantity
            
            # 缓存订单
            self.orders[order.order_id] = order
            self.order_id_map[order.order_id] = result['orderId']
            
            # 更新持仓
            self._update_position_from_order(order)
            
            self.logger.info(f"订单已提交: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
            print(f"订单已提交: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
            
            # 订单确认机制
            if self.enable_order_confirmation and signal.order_type == OrderType.MARKET:
                confirmation = OrderConfirmation(self, order)
                if confirmation.confirm(timeout=30):
                    self.logger.info(f"订单确认成功: {order.order_id}")
                else:
                    self.logger.warning(f"订单确认失败: {order.order_id}, {confirmation.error_message}")
            
            return order
            
        except (BinanceAPIException, BinanceOrderException) as e:
            self.logger.error(f"币安API错误: {e}")
            print(f"币安API错误: {e}")
            return None
        except Exception as e:
            self.logger.error(f"下单失败: {e}")
            print(f"下单失败: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """增强版撤单方法 - 支持重试和限频"""
        
        # 等待限频
        self._wait_for_rate_limit(self.rate_limiter)
        
        try:
            order = self.orders.get(order_id)
            if not order:
                self.logger.error(f"订单不存在: {order_id}")
                print(f"订单不存在: {order_id}")
                return False
            
            # 使用重试机制发送撤单请求
            def _cancel_order_internal():
                return self.client.cancel_order(
                    symbol=order.symbol,
                    orderId=self.order_id_map[order_id]
                )
            
            result = self._execute_with_retry(_cancel_order_internal)
            
            # 更新订单状态
            order.status = OrderStatus.CANCELED
            order.update_time = datetime.now()
            
            self.logger.info(f"订单已撤销: {order_id}")
            print(f"订单已撤销: {order_id}")
            return True
            
        except (BinanceAPIException, BinanceOrderException) as e:
            self.logger.error(f"撤单失败: {e}")
            print(f"撤单失败: {e}")
            return False
        except Exception as e:
            self.logger.error(f"撤单失败: {e}")
            print(f"撤单失败: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """增强版查询订单方法 - 支持重试和限频"""
        
        # 等待查询限频
        self._wait_for_rate_limit(self.query_rate_limiter)
        
        try:
            order = self.orders.get(order_id)
            if not order:
                return None
            
            # 使用重试机制查询最新状态
            def _get_order_internal():
                return self.client.get_order(
                    symbol=order.symbol,
                    orderId=self.order_id_map[order_id]
                )
            
            result = self._execute_with_retry(_get_order_internal)
            
            # 更新订单状态
            order.status = OrderStatus(result['status'])
            order.filled_quantity = float(result['executedQty'])
            order.remaining_quantity = order.quantity - order.filled_quantity
            order.update_time = datetime.now()
            
            if order.filled_quantity > 0:
                order.avg_price = float(result['cummulativeQuoteQty']) / order.filled_quantity
            
            return order
            
        except Exception as e:
            self.logger.error(f"查询订单失败: {e}")
            print(f"查询订单失败: {e}")
            return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """增强版获取未成交订单方法 - 支持重试和限频"""
        
        # 等待查询限频
        self._wait_for_rate_limit(self.query_rate_limiter)
        
        try:
            # 使用重试机制获取未成交订单
            def _get_open_orders_internal():
                if symbol:
                    return self.client.get_open_orders(symbol=symbol)
                else:
                    return self.client.get_open_orders()
            
            results = self._execute_with_retry(_get_open_orders_internal)
            
            orders = []
            for result in results:
                order_id = result['orderId']
                
                # 检查是否在缓存中
                cached_order = None
                for cached_id, cached_order_obj in self.orders.items():
                    if self.order_id_map.get(cached_id) == order_id:
                        cached_order = cached_order_obj
                        break
                
                if cached_order:
                    # 更新缓存订单
                    cached_order.status = OrderStatus(result['status'])
                    cached_order.filled_quantity = float(result['executedQty'])
                    cached_order.remaining_quantity = cached_order.quantity - cached_order.filled_quantity
                    orders.append(cached_order)
                else:
                    # 创建新订单对象
                    order = Order(
                        order_id=order_id,
                        symbol=result['symbol'],
                        side=OrderSide(result['side']),
                        order_type=OrderType.MARKET if result['type'] == 'MARKET' else OrderType.LIMIT,
                        quantity=float(result['origQty']),
                        price=float(result['price']) if result['price'] != '0.00000000' else None,
                        status=OrderStatus(result['status']),
                        filled_quantity=float(result['executedQty']),
                        create_time=datetime.fromtimestamp(result['time'] / 1000),
                        update_time=datetime.fromtimestamp(result['updateTime'] / 1000)
                    )
                    order.remaining_quantity = order.quantity - order.filled_quantity
                    orders.append(order)
            
            return orders
            
        except Exception as e:
            self.logger.error(f"获取未成交订单失败: {e}")
            print(f"获取未成交订单失败: {e}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)
    
    def get_account_balance(self) -> Dict[str, float]:
        """增强版获取账户余额方法 - 支持重试和限频"""
        
        # 等待查询限频
        self._wait_for_rate_limit(self.query_rate_limiter)
        
        try:
            # 使用重试机制获取账户信息
            def _get_account_internal():
                return self.client.get_account()
            
            account_info = self._execute_with_retry(_get_account_internal)
            balances = {}
            
            for balance in account_info['balances']:
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:
                    balances[balance['asset']] = {
                        'free': free,
                        'locked': locked,
                        'total': total
                    }
            
            return balances
            
        except Exception as e:
            self.logger.error(f"获取账户余额失败: {e}")
            print(f"获取账户余额失败: {e}")
            return {}
    
    def _update_position_from_order(self, order: Order) -> None:
        """从订单更新持仓"""
        if order.filled_quantity <= 0:
            return
        
        symbol = order.symbol
        quantity_delta = order.filled_quantity
        
        # 卖出时数量为负
        if order.side == OrderSide.SELL:
            quantity_delta = -quantity_delta
        
        # 获取或创建持仓
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, quantity=0, avg_price=0)
        
        position = self.positions[symbol]
        position.update_position(quantity_delta, order.avg_price)
        
        # 如果持仓为0，清除持仓记录
        if position.quantity == 0:
            del self.positions[symbol]
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """
        获取限频器状态信息
        
        Returns:
            Dict[str, Any]: 限频器状态
        """
        status = {}
        
        if self.rate_limiter:
            with self.rate_limiter.lock:
                current_time = time.time()
                active_requests = [req_time for req_time in self.rate_limiter.requests 
                                 if current_time - req_time < self.rate_limiter.time_window]
                status['order_rate_limiter'] = {
                    'max_requests': self.rate_limiter.max_requests,
                    'time_window': self.rate_limiter.time_window,
                    'current_requests': len(active_requests),
                    'available_requests': self.rate_limiter.max_requests - len(active_requests),
                    'can_request': len(active_requests) < self.rate_limiter.max_requests
                }
        
        if self.query_rate_limiter:
            with self.query_rate_limiter.lock:
                current_time = time.time()
                active_requests = [req_time for req_time in self.query_rate_limiter.requests 
                                 if current_time - req_time < self.query_rate_limiter.time_window]
                status['query_rate_limiter'] = {
                    'max_requests': self.query_rate_limiter.max_requests,
                    'time_window': self.query_rate_limiter.time_window,
                    'current_requests': len(active_requests),
                    'available_requests': self.query_rate_limiter.max_requests - len(active_requests),
                    'can_request': len(active_requests) < self.query_rate_limiter.max_requests
                }
        
        return status
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        获取连接状态
        
        Returns:
            Dict[str, Any]: 连接状态信息
        """
        status = {
            'connected': False,
            'api_key_valid': False,
            'trading_enabled': False,
            'server_time_diff': None,
            'last_validation': self.account_validator.last_validation_time,
            'error': None
        }
        
        try:
            # 测试连接
            self.client.ping()
            status['connected'] = True
            
            # 获取服务器时间
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            status['server_time_diff'] = server_time['serverTime'] - local_time
            
            # 测试API密钥
            account_info = self.client.get_account()
            status['api_key_valid'] = True
            status['trading_enabled'] = account_info.get('canTrade', False)
            
        except Exception as e:
            status['error'] = str(e)
            self.logger.error(f"连接状态检查失败: {e}")
        
        return status
    
    def get_broker_stats(self) -> Dict[str, Any]:
        """
        获取经纪商统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'total_orders': len(self.orders),
            'active_positions': len(self.positions),
            'connection_status': self.get_connection_status(),
            'rate_limit_status': self.get_rate_limit_status(),
            'configuration': {
                'testnet': self.testnet,
                'rate_limiting_enabled': self.enable_rate_limiting,
                'order_confirmation_enabled': self.enable_order_confirmation,
                'retry_config': {
                    'max_retries': self.retry_config.max_retries,
                    'retry_delay': self.retry_config.retry_delay,
                    'backoff_factor': self.retry_config.backoff_factor,
                    'max_delay': self.retry_config.max_delay
                }
            }
        }
        
        # 订单状态统计
        order_status_count = {}
        for order in self.orders.values():
            status = order.status.value
            order_status_count[status] = order_status_count.get(status, 0) + 1
        stats['order_status_distribution'] = order_status_count
        
        # 持仓统计
        position_stats = {}
        total_unrealized_pnl = 0
        total_realized_pnl = 0
        
        for position in self.positions.values():
            total_unrealized_pnl += position.unrealized_pnl
            total_realized_pnl += position.realized_pnl
        
        position_stats['total_unrealized_pnl'] = total_unrealized_pnl
        position_stats['total_realized_pnl'] = total_realized_pnl
        stats['position_stats'] = position_stats
        
        return stats


class SimulatedBroker(Broker):
    """模拟经纪商实现（用于回测和模拟交易）"""
    
    def __init__(self, initial_balance: Dict[str, float], commission_rate: float = 0.001):
        """
        初始化模拟经纪商
        
        Args:
            initial_balance: 初始余额
            commission_rate: 手续费率
        """
        self.initial_balance = initial_balance.copy()
        self.balance = initial_balance.copy()
        self.commission_rate = commission_rate
        
        # 内部数据
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_counter = 0
        
        print(f"模拟经纪商初始化完成，初始余额: {initial_balance}")
    
    def place_order(self, signal: TradeSignal, quantity: float, price: Optional[float] = None) -> Optional[Order]:
        """模拟下单"""
        try:
            # 生成订单ID
            self.order_counter += 1
            order_id = f"sim_order_{self.order_counter}"
            
            # 创建订单
            order = Order(
                order_id=order_id,
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                order_type=signal.order_type,
                quantity=quantity,
                price=price,
                strategy_name=signal.strategy_name,
                metadata=signal.metadata
            )
            
            # 模拟即时成交（市价单）
            if signal.order_type == OrderType.MARKET:
                # 使用当前价格成交
                fill_price = price if price else signal.price if signal.price else 100.0  # 默认价格
                self._fill_order(order, quantity, fill_price)
            else:
                # 限价单等待成交
                order.status = OrderStatus.NEW
            
            # 保存订单
            self.orders[order_id] = order
            
            print(f"模拟下单: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
            
            return order
            
        except Exception as e:
            print(f"模拟下单失败: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """模拟撤单"""
        order = self.orders.get(order_id)
        if not order:
            return False
        
        if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            order.status = OrderStatus.CANCELED
            order.update_time = datetime.now()
            print(f"模拟撤单: {order_id}")
            return True
        
        return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未成交订单"""
        open_orders = []
        for order in self.orders.values():
            if order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                if symbol is None or order.symbol == symbol:
                    open_orders.append(order)
        return open_orders
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)
    
    def get_account_balance(self) -> Dict[str, float]:
        """获取账户余额"""
        return self.balance.copy()
    
    def _fill_order(self, order: Order, fill_qty: float, fill_price: float) -> None:
        """模拟订单成交"""
        # 计算手续费
        commission = fill_qty * fill_price * self.commission_rate
        
        # 更新订单
        order.update_fill(fill_qty, fill_price, commission)
        
        # 更新余额
        base_asset = order.symbol.replace('USDT', '').replace('BUSD', '').replace('BTC', '')
        quote_asset = order.symbol.replace(base_asset, '')
        
        if order.side == OrderSide.BUY:
            # 买入：扣除报价货币，增加基础货币
            cost = fill_qty * fill_price + commission
            self.balance[quote_asset] = self.balance.get(quote_asset, 0) - cost
            self.balance[base_asset] = self.balance.get(base_asset, 0) + fill_qty
        else:
            # 卖出：扣除基础货币，增加报价货币
            revenue = fill_qty * fill_price - commission
            self.balance[base_asset] = self.balance.get(base_asset, 0) - fill_qty
            self.balance[quote_asset] = self.balance.get(quote_asset, 0) + revenue
        
        # 更新持仓
        self._update_position_from_order(order)
    
    def _update_position_from_order(self, order: Order) -> None:
        """从订单更新持仓"""
        if order.filled_quantity <= 0:
            return
        
        symbol = order.symbol
        quantity_delta = order.filled_quantity
        
        if order.side == OrderSide.SELL:
            quantity_delta = -quantity_delta
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol, quantity=0, avg_price=0)
        
        position = self.positions[symbol]
        position.update_position(quantity_delta, order.avg_price)
        
        if position.quantity == 0:
            del self.positions[symbol]
    
    def reset(self) -> None:
        """重置模拟经纪商"""
        self.balance = self.initial_balance.copy()
        self.orders.clear()
        self.positions.clear()
        self.order_counter = 0
        print("模拟经纪商已重置")