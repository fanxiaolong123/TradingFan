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

# 可选导入，避免缺少依赖时导入失败
try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
    BINANCE_AVAILABLE = True
except ImportError:
    Client = None
    BinanceAPIException = Exception
    BINANCE_AVAILABLE = False

from ..strategies.base import TradeSignal, OrderType, SignalType


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
    """币安经纪商实现"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        """
        初始化币安经纪商
        
        Args:
            api_key: API密钥
            api_secret: API密钥
            testnet: 是否使用测试网
        """
        if not BINANCE_AVAILABLE:
            raise ImportError("python-binance库未安装，无法使用BinanceBroker")
            
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # 初始化币安客户端
        self.client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)
        
        # 内部数据
        self.orders: Dict[str, Order] = {}              # 订单缓存
        self.positions: Dict[str, Position] = {}        # 持仓缓存
        self.order_id_map: Dict[str, str] = {}          # 订单ID映射（内部ID -> 交易所ID）
        
        # 测试连接
        try:
            self.client.ping()
            print(f"币安经纪商初始化成功 (testnet: {testnet})")
        except Exception as e:
            print(f"币安经纪商初始化失败: {e}")
            raise
    
    def place_order(self, signal: TradeSignal, quantity: float, price: Optional[float] = None) -> Optional[Order]:
        """下单"""
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
            
            # 发送订单到币安
            result = self.client.create_order(**binance_params)
            
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
            
            print(f"订单已提交: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
            
            return order
            
        except BinanceAPIException as e:
            print(f"币安API错误: {e}")
            return None
        except Exception as e:
            print(f"下单失败: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        try:
            order = self.orders.get(order_id)
            if not order:
                print(f"订单不存在: {order_id}")
                return False
            
            # 发送撤单请求
            result = self.client.cancel_order(
                symbol=order.symbol,
                orderId=self.order_id_map[order_id]
            )
            
            # 更新订单状态
            order.status = OrderStatus.CANCELED
            order.update_time = datetime.now()
            
            print(f"订单已撤销: {order_id}")
            return True
            
        except BinanceAPIException as e:
            print(f"撤单失败: {e}")
            return False
        except Exception as e:
            print(f"撤单失败: {e}")
            return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """查询订单"""
        try:
            order = self.orders.get(order_id)
            if not order:
                return None
            
            # 查询最新状态
            result = self.client.get_order(
                symbol=order.symbol,
                orderId=self.order_id_map[order_id]
            )
            
            # 更新订单状态
            order.status = OrderStatus(result['status'])
            order.filled_quantity = float(result['executedQty'])
            order.remaining_quantity = order.quantity - order.filled_quantity
            order.update_time = datetime.now()
            
            if order.filled_quantity > 0:
                order.avg_price = float(result['cummulativeQuoteQty']) / order.filled_quantity
            
            return order
            
        except Exception as e:
            print(f"查询订单失败: {e}")
            return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未成交订单"""
        try:
            if symbol:
                results = self.client.get_open_orders(symbol=symbol)
            else:
                results = self.client.get_open_orders()
            
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
            print(f"获取未成交订单失败: {e}")
            return []
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""
        return self.positions.get(symbol)
    
    def get_account_balance(self) -> Dict[str, float]:
        """获取账户余额"""
        try:
            account_info = self.client.get_account()
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