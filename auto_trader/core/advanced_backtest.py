"""
高级回测引擎模块 - 提供更真实的市场模拟

这个模块在原有回测基础上增加了：
- 真实交易延迟模型：模拟网络延迟、订单处理延迟
- 高级滑点模型：基于成交量和市场深度的动态滑点
- 盘口深度撮合：模拟真实的买卖盘口，按价格优先时间优先原则撮合
- 市场冲击模型：大单对市场价格的影响
"""

import random
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import asyncio
from collections import deque, OrderedDict
import threading

from .backtest import BacktestEngine, BacktestConfig, BacktestResult, BacktestStatus
from .broker import Order, OrderStatus, OrderSide, SimulatedBroker
from .data import DataManager
from .account import AccountManager
from .risk import RiskManager, RiskLimits
from ..strategies.base import Strategy, StrategyConfig, TradeSignal, SignalType, OrderType, OrderFillEvent


class LatencyProfile(Enum):
    """延迟配置枚举"""
    EXCELLENT = "EXCELLENT"      # 优秀网络条件 (5-15ms)
    GOOD = "GOOD"               # 良好网络条件 (15-50ms)
    AVERAGE = "AVERAGE"         # 平均网络条件 (50-150ms)
    POOR = "POOR"               # 较差网络条件 (150-500ms)
    CUSTOM = "CUSTOM"           # 自定义延迟


@dataclass
class LatencyConfig:
    """延迟配置"""
    # 网络延迟范围 (毫秒)
    network_latency_min: float = 10.0      # 最小网络延迟
    network_latency_max: float = 50.0      # 最大网络延迟
    
    # 订单处理延迟范围 (毫秒)
    order_processing_min: float = 5.0      # 最小订单处理延迟
    order_processing_max: float = 20.0     # 最大订单处理延迟
    
    # 市场数据延迟范围 (毫秒)
    market_data_delay_min: float = 0.0     # 最小市场数据延迟
    market_data_delay_max: float = 10.0    # 最大市场数据延迟
    
    # 服务器拥堵时的额外延迟 (毫秒)
    congestion_penalty_min: float = 0.0    # 最小拥堵延迟
    congestion_penalty_max: float = 100.0  # 最大拥堵延迟
    congestion_probability: float = 0.05   # 拥堵概率
    
    @classmethod
    def from_profile(cls, profile: LatencyProfile) -> 'LatencyConfig':
        """根据延迟配置创建延迟配置对象"""
        if profile == LatencyProfile.EXCELLENT:
            return cls(
                network_latency_min=5.0,
                network_latency_max=15.0,
                order_processing_min=2.0,
                order_processing_max=8.0,
                market_data_delay_min=0.0,
                market_data_delay_max=5.0,
                congestion_penalty_min=0.0,
                congestion_penalty_max=20.0,
                congestion_probability=0.01
            )
        elif profile == LatencyProfile.GOOD:
            return cls(
                network_latency_min=15.0,
                network_latency_max=50.0,
                order_processing_min=5.0,
                order_processing_max=20.0,
                market_data_delay_min=0.0,
                market_data_delay_max=10.0,
                congestion_penalty_min=0.0,
                congestion_penalty_max=50.0,
                congestion_probability=0.03
            )
        elif profile == LatencyProfile.AVERAGE:
            return cls(
                network_latency_min=50.0,
                network_latency_max=150.0,
                order_processing_min=10.0,
                order_processing_max=40.0,
                market_data_delay_min=5.0,
                market_data_delay_max=20.0,
                congestion_penalty_min=20.0,
                congestion_penalty_max=100.0,
                congestion_probability=0.05
            )
        elif profile == LatencyProfile.POOR:
            return cls(
                network_latency_min=150.0,
                network_latency_max=500.0,
                order_processing_min=20.0,
                order_processing_max=100.0,
                market_data_delay_min=10.0,
                market_data_delay_max=50.0,
                congestion_penalty_min=50.0,
                congestion_penalty_max=200.0,
                congestion_probability=0.10
            )
        else:
            return cls()


@dataclass
class SlippageConfig:
    """滑点配置"""
    # 基础滑点 (固定滑点率)
    base_slippage_rate: float = 0.0001      # 0.01% 基础滑点
    
    # 成交量相关滑点
    volume_impact_factor: float = 0.00001   # 成交量影响因子
    volume_threshold: float = 1000000.0     # 成交量阈值 (USDT)
    
    # 市场深度相关滑点
    depth_impact_factor: float = 0.00005    # 深度影响因子
    min_depth_ratio: float = 0.1            # 最小深度比例
    
    # 价格波动相关滑点
    volatility_impact_factor: float = 0.0001  # 波动性影响因子
    volatility_lookback_periods: int = 20    # 波动性计算回望周期
    
    # 市场时间相关滑点 (某些时段流动性较差)
    time_based_multiplier: Dict[int, float] = field(default_factory=lambda: {
        0: 1.5,   # 00:00-01:00 流动性较差
        1: 1.3,   # 01:00-02:00
        2: 1.2,   # 02:00-03:00
        3: 1.2,   # 03:00-04:00
        4: 1.1,   # 04:00-05:00
        5: 1.0,   # 05:00-06:00 正常
        6: 0.9,   # 06:00-07:00 流动性较好
        7: 0.8,   # 07:00-08:00 流动性好
        8: 0.8,   # 08:00-09:00 流动性好
        9: 0.7,   # 09:00-10:00 流动性最好
        10: 0.7,  # 10:00-11:00 流动性最好
        11: 0.8,  # 11:00-12:00 流动性好
        12: 0.8,  # 12:00-13:00 流动性好
        13: 0.9,  # 13:00-14:00 流动性较好
        14: 1.0,  # 14:00-15:00 正常
        15: 1.0,  # 15:00-16:00 正常
        16: 0.9,  # 16:00-17:00 流动性较好
        17: 0.8,  # 17:00-18:00 流动性好
        18: 0.8,  # 18:00-19:00 流动性好
        19: 0.9,  # 19:00-20:00 流动性较好
        20: 1.0,  # 20:00-21:00 正常
        21: 1.1,  # 21:00-22:00 流动性稍差
        22: 1.2,  # 22:00-23:00 流动性较差
        23: 1.3   # 23:00-00:00 流动性较差
    })
    
    # 最大滑点限制
    max_slippage_rate: float = 0.01         # 最大滑点率 1%


@dataclass
class OrderBookEntry:
    """订单簿条目"""
    price: float           # 价格
    quantity: float        # 数量
    timestamp: datetime    # 时间戳
    order_count: int = 1   # 订单数量 (该价位的订单个数)


@dataclass
class OrderBook:
    """订单簿"""
    symbol: str                                    # 交易对
    bids: OrderedDict[float, OrderBookEntry]      # 买盘 (价格降序)
    asks: OrderedDict[float, OrderBookEntry]      # 卖盘 (价格升序)
    last_update: datetime                          # 最后更新时间
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.bids:
            self.bids = OrderedDict()
        if not self.asks:
            self.asks = OrderedDict()
    
    def get_best_bid(self) -> Optional[OrderBookEntry]:
        """获取最佳买价"""
        if self.bids:
            price = next(iter(self.bids))
            return self.bids[price]
        return None
    
    def get_best_ask(self) -> Optional[OrderBookEntry]:
        """获取最佳卖价"""
        if self.asks:
            price = next(iter(self.asks))
            return self.asks[price]
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """获取中间价"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        elif best_bid:
            return best_bid.price
        elif best_ask:
            return best_ask.price
        else:
            return None
    
    def get_spread(self) -> Optional[float]:
        """获取买卖价差"""
        best_bid = self.get_best_bid()
        best_ask = self.get_best_ask()
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """获取市场深度"""
        bids_list = []
        asks_list = []
        
        # 获取买盘深度
        for i, (price, entry) in enumerate(self.bids.items()):
            if i >= levels:
                break
            bids_list.append((price, entry.quantity))
        
        # 获取卖盘深度
        for i, (price, entry) in enumerate(self.asks.items()):
            if i >= levels:
                break
            asks_list.append((price, entry.quantity))
        
        return {
            'bids': bids_list,
            'asks': asks_list
        }
    
    def calculate_total_depth(self, side: str, max_levels: int = 10) -> float:
        """计算总深度 (指定方向的总流动性)"""
        total_quantity = 0.0
        
        if side.upper() == 'BID':
            for i, (price, entry) in enumerate(self.bids.items()):
                if i >= max_levels:
                    break
                total_quantity += entry.quantity
        elif side.upper() == 'ASK':
            for i, (price, entry) in enumerate(self.asks.items()):
                if i >= max_levels:
                    break
                total_quantity += entry.quantity
        
        return total_quantity


class OrderBookSimulator:
    """订单簿模拟器"""
    
    def __init__(self, symbol: str, initial_price: float = 50000.0):
        """
        初始化订单簿模拟器
        
        Args:
            symbol: 交易对符号
            initial_price: 初始价格
        """
        self.symbol = symbol
        self.current_price = initial_price
        self.order_book = OrderBook(
            symbol=symbol,
            bids=OrderedDict(),
            asks=OrderedDict(),
            last_update=datetime.now()
        )
        
        # 生成初始订单簿
        self._generate_initial_orderbook(initial_price)
    
    def _generate_initial_orderbook(self, base_price: float, levels: int = 20):
        """生成初始订单簿"""
        now = datetime.now()
        
        # 生成买盘 (价格递减)
        for i in range(levels):
            price_offset = (i + 1) * 0.001  # 0.1% 递减
            price = base_price * (1 - price_offset)
            quantity = random.uniform(0.1, 5.0) * (1 + i * 0.1)  # 距离越远数量越多
            
            self.order_book.bids[price] = OrderBookEntry(
                price=price,
                quantity=quantity,
                timestamp=now,
                order_count=random.randint(1, 5)
            )
        
        # 按价格降序排序买盘
        self.order_book.bids = OrderedDict(
            sorted(self.order_book.bids.items(), key=lambda x: x[0], reverse=True)
        )
        
        # 生成卖盘 (价格递增)
        for i in range(levels):
            price_offset = (i + 1) * 0.001  # 0.1% 递增
            price = base_price * (1 + price_offset)
            quantity = random.uniform(0.1, 5.0) * (1 + i * 0.1)  # 距离越远数量越多
            
            self.order_book.asks[price] = OrderBookEntry(
                price=price,
                quantity=quantity,
                timestamp=now,
                order_count=random.randint(1, 5)
            )
        
        # 按价格升序排序卖盘
        self.order_book.asks = OrderedDict(
            sorted(self.order_book.asks.items(), key=lambda x: x[0])
        )
    
    def update_price(self, new_price: float, volume: float = 0.0):
        """
        更新价格并调整订单簿
        
        Args:
            new_price: 新价格
            volume: 成交量
        """
        price_change_rate = (new_price - self.current_price) / self.current_price
        self.current_price = new_price
        
        # 根据价格变化调整订单簿
        self._adjust_orderbook_for_price_change(price_change_rate, volume)
        
        # 添加一些随机性来模拟市场动态
        self._add_random_liquidity()
    
    def _adjust_orderbook_for_price_change(self, price_change_rate: float, volume: float):
        """根据价格变化调整订单簿"""
        now = datetime.now()
        
        # 清除过时的订单簿条目
        if abs(price_change_rate) > 0.005:  # 如果价格变化超过0.5%，重新生成订单簿
            self._generate_initial_orderbook(self.current_price)
        else:
            # 微调现有订单簿
            adjustment_factor = 1 + price_change_rate * 0.1  # 价格影响因子
            
            # 调整买盘
            new_bids = OrderedDict()
            for price, entry in self.order_book.bids.items():
                new_price = price * adjustment_factor
                new_quantity = entry.quantity * random.uniform(0.8, 1.2)  # 添加随机性
                new_bids[new_price] = OrderBookEntry(
                    price=new_price,
                    quantity=new_quantity,
                    timestamp=now,
                    order_count=entry.order_count
                )
            
            # 调整卖盘
            new_asks = OrderedDict()
            for price, entry in self.order_book.asks.items():
                new_price = price * adjustment_factor
                new_quantity = entry.quantity * random.uniform(0.8, 1.2)  # 添加随机性
                new_asks[new_price] = OrderBookEntry(
                    price=new_price,
                    quantity=new_quantity,
                    timestamp=now,
                    order_count=entry.order_count
                )
            
            self.order_book.bids = new_bids
            self.order_book.asks = new_asks
        
        self.order_book.last_update = now
    
    def _add_random_liquidity(self):
        """添加随机流动性"""
        now = datetime.now()
        
        # 有小概率添加新的订单簿条目
        if random.random() < 0.3:  # 30% 概率添加新流动性
            # 添加买盘
            if random.random() < 0.5:
                best_bid = self.order_book.get_best_bid()
                if best_bid:
                    new_price = best_bid.price * random.uniform(0.995, 1.001)
                    new_quantity = random.uniform(0.1, 2.0)
                    self.order_book.bids[new_price] = OrderBookEntry(
                        price=new_price,
                        quantity=new_quantity,
                        timestamp=now
                    )
            
            # 添加卖盘
            else:
                best_ask = self.order_book.get_best_ask()
                if best_ask:
                    new_price = best_ask.price * random.uniform(0.999, 1.005)
                    new_quantity = random.uniform(0.1, 2.0)
                    self.order_book.asks[new_price] = OrderBookEntry(
                        price=new_price,
                        quantity=new_quantity,
                        timestamp=now
                    )
        
        # 重新排序订单簿
        self.order_book.bids = OrderedDict(
            sorted(self.order_book.bids.items(), key=lambda x: x[0], reverse=True)
        )
        self.order_book.asks = OrderedDict(
            sorted(self.order_book.asks.items(), key=lambda x: x[0])
        )
    
    def simulate_market_order_execution(self, side: OrderSide, quantity: float) -> Tuple[float, float, List[Tuple[float, float]]]:
        """
        模拟市价单执行
        
        Args:
            side: 订单方向
            quantity: 委托数量
            
        Returns:
            Tuple[平均成交价, 实际成交数量, 成交明细]
        """
        fills = []  # 成交明细 [(价格, 数量), ...]
        remaining_quantity = quantity
        total_cost = 0.0
        total_filled = 0.0
        
        if side == OrderSide.BUY:
            # 买单：从最佳卖价开始匹配
            asks_copy = self.order_book.asks.copy()
            for price, entry in asks_copy.items():
                if remaining_quantity <= 0:
                    break
                
                fill_quantity = min(remaining_quantity, entry.quantity)
                fills.append((price, fill_quantity))
                
                total_cost += price * fill_quantity
                total_filled += fill_quantity
                remaining_quantity -= fill_quantity
                
                # 更新订单簿
                if fill_quantity >= entry.quantity:
                    # 完全消耗该价位
                    del self.order_book.asks[price]
                else:
                    # 部分消耗
                    self.order_book.asks[price].quantity -= fill_quantity
        
        else:  # OrderSide.SELL
            # 卖单：从最佳买价开始匹配
            bids_copy = self.order_book.bids.copy()
            for price, entry in bids_copy.items():
                if remaining_quantity <= 0:
                    break
                
                fill_quantity = min(remaining_quantity, entry.quantity)
                fills.append((price, fill_quantity))
                
                total_cost += price * fill_quantity
                total_filled += fill_quantity
                remaining_quantity -= fill_quantity
                
                # 更新订单簿
                if fill_quantity >= entry.quantity:
                    # 完全消耗该价位
                    del self.order_book.bids[price]
                else:
                    # 部分消耗
                    self.order_book.bids[price].quantity -= fill_quantity
        
        # 计算平均成交价
        avg_price = total_cost / total_filled if total_filled > 0 else 0.0
        
        return avg_price, total_filled, fills


class AdvancedMarketSimulator:
    """高级市场模拟器"""
    
    def __init__(self, latency_config: LatencyConfig, slippage_config: SlippageConfig):
        """
        初始化高级市场模拟器
        
        Args:
            latency_config: 延迟配置
            slippage_config: 滑点配置
        """
        self.latency_config = latency_config
        self.slippage_config = slippage_config
        
        # 订单簿模拟器
        self.orderbook_simulators: Dict[str, OrderBookSimulator] = {}
        
        # 市场状态
        self.market_volatility_cache: Dict[str, deque] = {}  # 波动性缓存
        self.recent_volumes: Dict[str, deque] = {}           # 近期成交量缓存
        
        # 延迟队列
        self.pending_orders: List[Tuple[datetime, Order, float]] = []  # (执行时间, 订单, 期望价格)
        
        print("高级市场模拟器初始化完成")
    
    def initialize_symbol(self, symbol: str, initial_price: float):
        """初始化交易对"""
        if symbol not in self.orderbook_simulators:
            self.orderbook_simulators[symbol] = OrderBookSimulator(symbol, initial_price)
            self.market_volatility_cache[symbol] = deque(maxlen=self.slippage_config.volatility_lookback_periods)
            self.recent_volumes[symbol] = deque(maxlen=24)  # 保存24小时成交量
    
    def update_market_data(self, symbol: str, price: float, volume: float = 0.0):
        """更新市场数据"""
        if symbol in self.orderbook_simulators:
            # 更新订单簿
            self.orderbook_simulators[symbol].update_price(price, volume)
            
            # 更新波动性缓存
            volatility_cache = self.market_volatility_cache[symbol]
            if len(volatility_cache) > 0:
                price_change = abs(price - volatility_cache[-1]) / volatility_cache[-1]
                volatility_cache.append(price_change)
            else:
                volatility_cache.append(0.0)
            
            # 更新成交量缓存
            self.recent_volumes[symbol].append(volume)
    
    def calculate_execution_delay(self) -> float:
        """计算执行延迟 (毫秒)"""
        # 基础网络延迟
        network_delay = random.uniform(
            self.latency_config.network_latency_min,
            self.latency_config.network_latency_max
        )
        
        # 订单处理延迟
        processing_delay = random.uniform(
            self.latency_config.order_processing_min,
            self.latency_config.order_processing_max
        )
        
        # 拥堵延迟 (小概率)
        congestion_delay = 0.0
        if random.random() < self.latency_config.congestion_probability:
            congestion_delay = random.uniform(
                self.latency_config.congestion_penalty_min,
                self.latency_config.congestion_penalty_max
            )
        
        total_delay = network_delay + processing_delay + congestion_delay
        return total_delay
    
    def calculate_slippage(self, symbol: str, side: OrderSide, quantity: float, 
                          market_price: float, timestamp: datetime) -> float:
        """
        计算滑点
        
        Args:
            symbol: 交易对
            side: 买卖方向
            quantity: 交易数量
            market_price: 市场价格
            timestamp: 时间戳
            
        Returns:
            float: 滑点率
        """
        # 基础滑点
        base_slippage = self.slippage_config.base_slippage_rate
        
        # 成交量影响
        volume_impact = 0.0
        order_value = quantity * market_price
        if order_value > self.slippage_config.volume_threshold:
            volume_ratio = order_value / self.slippage_config.volume_threshold
            volume_impact = self.slippage_config.volume_impact_factor * volume_ratio
        
        # 市场深度影响
        depth_impact = 0.0
        if symbol in self.orderbook_simulators:
            orderbook = self.orderbook_simulators[symbol].order_book
            side_str = 'ask' if side == OrderSide.BUY else 'bid'
            total_depth = orderbook.calculate_total_depth(side_str)
            
            if total_depth > 0:
                depth_ratio = quantity / total_depth
                if depth_ratio > self.slippage_config.min_depth_ratio:
                    depth_impact = self.slippage_config.depth_impact_factor * depth_ratio
        
        # 波动性影响
        volatility_impact = 0.0
        if symbol in self.market_volatility_cache:
            volatility_cache = self.market_volatility_cache[symbol]
            if len(volatility_cache) > 1:
                avg_volatility = np.mean(list(volatility_cache))
                volatility_impact = self.slippage_config.volatility_impact_factor * avg_volatility
        
        # 时间影响 (某些时段流动性较差)
        time_impact = 0.0
        hour = timestamp.hour
        time_multiplier = self.slippage_config.time_based_multiplier.get(hour, 1.0)
        time_impact = base_slippage * (time_multiplier - 1.0)
        
        # 计算总滑点
        total_slippage = base_slippage + volume_impact + depth_impact + volatility_impact + time_impact
        
        # 限制最大滑点
        total_slippage = min(total_slippage, self.slippage_config.max_slippage_rate)
        
        return total_slippage
    
    def simulate_order_execution(self, order: Order, expected_price: float, 
                                timestamp: datetime) -> Tuple[float, float, List[Dict]]:
        """
        模拟订单执行
        
        Args:
            order: 订单对象
            expected_price: 期望价格
            timestamp: 时间戳
            
        Returns:
            Tuple[实际成交价, 实际成交数量, 执行详情]
        """
        symbol = order.symbol
        side = order.side
        quantity = order.quantity
        
        # 确保订单簿已初始化
        if symbol not in self.orderbook_simulators:
            self.initialize_symbol(symbol, expected_price)
        
        orderbook_sim = self.orderbook_simulators[symbol]
        
        # 更新订单簿价格
        orderbook_sim.update_price(expected_price)
        
        # 计算滑点
        slippage_rate = self.calculate_slippage(symbol, side, quantity, expected_price, timestamp)
        
        execution_details = []
        
        if order.order_type == OrderType.MARKET:
            # 市价单：使用订单簿撮合
            avg_price, filled_quantity, fills = orderbook_sim.simulate_market_order_execution(side, quantity)
            
            # 应用滑点
            if side == OrderSide.BUY:
                # 买单滑点：价格上涨
                final_price = avg_price * (1 + slippage_rate)
            else:
                # 卖单滑点：价格下跌
                final_price = avg_price * (1 - slippage_rate)
            
            execution_details = [
                {
                    'type': 'market_execution',
                    'original_price': avg_price,
                    'slippage_rate': slippage_rate,
                    'final_price': final_price,
                    'fills': fills
                }
            ]
            
            return final_price, filled_quantity, execution_details
        
        else:
            # 限价单：检查是否能立即成交
            if side == OrderSide.BUY:
                best_ask = orderbook_sim.order_book.get_best_ask()
                if best_ask and order.price >= best_ask.price:
                    # 可以立即成交
                    fill_price = min(order.price, best_ask.price)
                    fill_quantity = min(quantity, best_ask.quantity)
                    
                    # 应用轻微滑点
                    final_price = fill_price * (1 + slippage_rate * 0.1)  # 限价单滑点较小
                    
                    execution_details = [
                        {
                            'type': 'limit_immediate_execution',
                            'fill_price': fill_price,
                            'slippage_rate': slippage_rate * 0.1,
                            'final_price': final_price
                        }
                    ]
                    
                    return final_price, fill_quantity, execution_details
            
            else:  # OrderSide.SELL
                best_bid = orderbook_sim.order_book.get_best_bid()
                if best_bid and order.price <= best_bid.price:
                    # 可以立即成交
                    fill_price = max(order.price, best_bid.price)
                    fill_quantity = min(quantity, best_bid.quantity)
                    
                    # 应用轻微滑点
                    final_price = fill_price * (1 - slippage_rate * 0.1)  # 限价单滑点较小
                    
                    execution_details = [
                        {
                            'type': 'limit_immediate_execution',
                            'fill_price': fill_price,
                            'slippage_rate': slippage_rate * 0.1,
                            'final_price': final_price
                        }
                    ]
                    
                    return final_price, fill_quantity, execution_details
            
            # 限价单不能立即成交，进入等待队列
            return 0.0, 0.0, [{'type': 'limit_pending', 'message': '限价单等待成交'}]
    
    def process_delayed_order(self, order: Order, expected_price: float) -> Optional[Order]:
        """
        处理延迟订单
        
        Args:
            order: 原始订单
            expected_price: 期望价格
            
        Returns:
            Order: 更新后的订单，如果执行失败返回None
        """
        # 计算执行延迟
        delay_ms = self.calculate_execution_delay()
        execution_time = datetime.now() + timedelta(milliseconds=delay_ms)
        
        # 将订单加入延迟队列
        self.pending_orders.append((execution_time, order, expected_price))
        
        # 返回待处理状态的订单
        order.status = OrderStatus.PENDING
        return order
    
    def process_pending_orders(self, current_time: datetime) -> List[Order]:
        """
        处理待执行的订单
        
        Args:
            current_time: 当前时间
            
        Returns:
            List[Order]: 已执行的订单列表
        """
        executed_orders = []
        remaining_orders = []
        
        for execution_time, order, expected_price in self.pending_orders:
            if current_time >= execution_time:
                # 到达执行时间，执行订单
                try:
                    final_price, filled_quantity, details = self.simulate_order_execution(
                        order, expected_price, current_time
                    )
                    
                    if filled_quantity > 0:
                        # 更新订单状态
                        order.update_fill(filled_quantity, final_price)
                        order.metadata['execution_details'] = details
                        executed_orders.append(order)
                    else:
                        # 未能成交，继续等待
                        remaining_orders.append((execution_time, order, expected_price))
                
                except Exception as e:
                    # 执行失败
                    order.status = OrderStatus.REJECTED
                    order.metadata['error'] = str(e)
                    executed_orders.append(order)
            else:
                # 未到执行时间，继续等待
                remaining_orders.append((execution_time, order, expected_price))
        
        # 更新待处理订单列表
        self.pending_orders = remaining_orders
        
        return executed_orders
    
    def get_market_impact_estimate(self, symbol: str, side: OrderSide, quantity: float, price: float) -> Dict[str, float]:
        """
        估算市场冲击
        
        Args:
            symbol: 交易对
            side: 买卖方向
            quantity: 交易数量
            price: 价格
            
        Returns:
            Dict: 市场冲击估算
        """
        if symbol not in self.orderbook_simulators:
            return {
                'price_impact': 0.0,
                'slippage_estimate': 0.0,
                'liquidity_consumed': 0.0
            }
        
        orderbook = self.orderbook_simulators[symbol].order_book
        order_value = quantity * price
        
        # 计算流动性消耗
        side_str = 'ask' if side == OrderSide.BUY else 'bid'
        total_depth = orderbook.calculate_total_depth(side_str)
        liquidity_consumed = quantity / total_depth if total_depth > 0 else 1.0
        
        # 估算价格冲击
        price_impact = 0.0
        if liquidity_consumed > 0.1:  # 如果消耗超过10%的流动性
            price_impact = liquidity_consumed * 0.001  # 0.1% per 10% liquidity
        
        # 估算滑点
        slippage_estimate = self.calculate_slippage(symbol, side, quantity, price, datetime.now())
        
        return {
            'price_impact': price_impact,
            'slippage_estimate': slippage_estimate,
            'liquidity_consumed': liquidity_consumed,
            'order_value': order_value,
            'total_depth': total_depth
        }


class AdvancedSimulatedBroker(SimulatedBroker):
    """高级模拟经纪商 - 集成高级市场模拟器"""
    
    def __init__(self, initial_balance: Dict[str, float], commission_rate: float = 0.001,
                 latency_profile: LatencyProfile = LatencyProfile.GOOD,
                 latency_config: Optional[LatencyConfig] = None,
                 slippage_config: Optional[SlippageConfig] = None):
        """
        初始化高级模拟经纪商
        
        Args:
            initial_balance: 初始余额
            commission_rate: 手续费率
            latency_profile: 延迟配置档案
            latency_config: 自定义延迟配置
            slippage_config: 滑点配置
        """
        super().__init__(initial_balance, commission_rate)
        
        # 延迟配置
        if latency_config:
            self.latency_config = latency_config
        else:
            self.latency_config = LatencyConfig.from_profile(latency_profile)
        
        # 滑点配置
        self.slippage_config = slippage_config or SlippageConfig()
        
        # 高级市场模拟器
        self.market_simulator = AdvancedMarketSimulator(self.latency_config, self.slippage_config)
        
        # 价格缓存
        self.price_cache: Dict[str, float] = {}
        
        print(f"高级模拟经纪商初始化完成，延迟档案: {latency_profile}")
    
    def update_market_price(self, symbol: str, price: float, volume: float = 0.0):
        """更新市场价格和数据"""
        self.price_cache[symbol] = price
        self.market_simulator.update_market_data(symbol, price, volume)
    
    def place_order(self, signal: TradeSignal, quantity: float, price: Optional[float] = None) -> Optional[Order]:
        """下单（带延迟和滑点模拟）"""
        try:
            # 生成订单ID
            self.order_counter += 1
            order_id = f"adv_sim_order_{self.order_counter}"
            
            # 创建订单
            order = Order(
                order_id=order_id,
                symbol=signal.symbol,
                side=OrderSide.BUY if signal.signal_type == SignalType.BUY else OrderSide.SELL,
                order_type=signal.order_type,
                quantity=quantity,
                price=price,
                strategy_name=signal.strategy_name,
                metadata=signal.metadata.copy()
            )
            
            # 获取期望价格
            expected_price = price if price else signal.price if signal.price else self.price_cache.get(signal.symbol, 100.0)
            
            # 市场冲击评估
            impact_estimate = self.market_simulator.get_market_impact_estimate(
                signal.symbol, order.side, quantity, expected_price
            )
            order.metadata['market_impact_estimate'] = impact_estimate
            
            if signal.order_type == OrderType.MARKET:
                # 市价单：通过延迟处理模拟真实交易
                delayed_order = self.market_simulator.process_delayed_order(order, expected_price)
                if delayed_order:
                    self.orders[order_id] = delayed_order
                    print(f"高级模拟下单 (延迟处理): {order.symbol} {order.side.value} {order.quantity}")
                    return delayed_order
            else:
                # 限价单：立即检查是否能成交
                final_price, filled_quantity, details = self.market_simulator.simulate_order_execution(
                    order, expected_price, datetime.now()
                )
                
                if filled_quantity > 0:
                    # 立即成交
                    commission = filled_quantity * final_price * self.commission_rate
                    order.update_fill(filled_quantity, final_price, commission)
                    order.metadata['execution_details'] = details
                    
                    # 更新余额和持仓
                    self._update_balance_and_position(order)
                else:
                    # 未成交，设为等待状态
                    order.status = OrderStatus.NEW
                
                self.orders[order_id] = order
                print(f"高级模拟下单: {order.symbol} {order.side.value} {order.quantity} @ {order.price}")
                return order
            
        except Exception as e:
            print(f"高级模拟下单失败: {e}")
            return None
        
        return None
    
    def process_pending_orders(self, current_time: Optional[datetime] = None) -> List[Order]:
        """处理延迟订单"""
        if current_time is None:
            current_time = datetime.now()
        
        executed_orders = self.market_simulator.process_pending_orders(current_time)
        
        # 更新已执行订单的余额和持仓
        for order in executed_orders:
            if order.status == OrderStatus.FILLED and order.filled_quantity > 0:
                self._update_balance_and_position(order)
                print(f"延迟订单执行完成: {order.symbol} {order.side.value} {order.filled_quantity} @ {order.avg_price}")
        
        return executed_orders
    
    def _update_balance_and_position(self, order: Order):
        """更新余额和持仓"""
        if order.filled_quantity <= 0:
            return
        
        # 解析货币对
        symbol = order.symbol
        base_asset = symbol.replace('USDT', '').replace('BUSD', '')
        quote_asset = 'USDT' if 'USDT' in symbol else 'BUSD'
        
        # 计算手续费
        commission = order.commission
        
        if order.side == OrderSide.BUY:
            # 买入：扣除报价货币，增加基础货币
            cost = order.filled_quantity * order.avg_price + commission
            self.balance[quote_asset] = self.balance.get(quote_asset, 0) - cost
            self.balance[base_asset] = self.balance.get(base_asset, 0) + order.filled_quantity
        else:
            # 卖出：扣除基础货币，增加报价货币
            revenue = order.filled_quantity * order.avg_price - commission
            self.balance[base_asset] = self.balance.get(base_asset, 0) - order.filled_quantity
            self.balance[quote_asset] = self.balance.get(quote_asset, 0) + revenue
        
        # 更新持仓
        self._update_position_from_order(order)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计信息"""
        total_orders = len(self.orders)
        filled_orders = sum(1 for order in self.orders.values() if order.status == OrderStatus.FILLED)
        pending_orders = len(self.market_simulator.pending_orders)
        
        # 计算平均滑点
        total_slippage = 0.0
        slippage_count = 0
        
        for order in self.orders.values():
            if 'execution_details' in order.metadata:
                for detail in order.metadata['execution_details']:
                    if 'slippage_rate' in detail:
                        total_slippage += detail['slippage_rate']
                        slippage_count += 1
        
        avg_slippage = total_slippage / slippage_count if slippage_count > 0 else 0.0
        
        return {
            'total_orders': total_orders,
            'filled_orders': filled_orders,
            'pending_orders': pending_orders,
            'fill_rate': filled_orders / total_orders if total_orders > 0 else 0.0,
            'average_slippage': avg_slippage,
            'latency_config': {
                'network_latency_range': f"{self.latency_config.network_latency_min}-{self.latency_config.network_latency_max}ms",
                'processing_delay_range': f"{self.latency_config.order_processing_min}-{self.latency_config.order_processing_max}ms"
            }
        }


class AdvancedBacktestEngine(BacktestEngine):
    """高级回测引擎 - 集成真实市场模拟"""
    
    def __init__(self, data_manager: DataManager, 
                 latency_profile: LatencyProfile = LatencyProfile.GOOD,
                 latency_config: Optional[LatencyConfig] = None,
                 slippage_config: Optional[SlippageConfig] = None):
        """
        初始化高级回测引擎
        
        Args:
            data_manager: 数据管理器
            latency_profile: 延迟配置档案
            latency_config: 自定义延迟配置
            slippage_config: 滑点配置
        """
        super().__init__(data_manager)
        
        self.latency_profile = latency_profile
        self.latency_config = latency_config
        self.slippage_config = slippage_config
        
        print("高级回测引擎初始化完成")
    
    def _initialize_backtest_components(self, config: BacktestConfig) -> None:
        """初始化回测组件（使用高级模拟经纪商）"""
        # 创建高级模拟经纪商
        self.broker = AdvancedSimulatedBroker(
            initial_balance=config.initial_balance,
            commission_rate=config.commission_rate,
            latency_profile=self.latency_profile,
            latency_config=self.latency_config,
            slippage_config=self.slippage_config
        )
        
        # 创建账户管理器
        self.account_manager = AccountManager()
        self.account_manager.set_initial_balance(config.initial_balance)
        
        # 创建风险管理器
        if config.enable_risk_management:
            risk_limits = config.risk_limits or RiskLimits()
            self.risk_manager = RiskManager(risk_limits)
    
    def _process_timestamp(self, timestamp: datetime, config: BacktestConfig) -> None:
        """处理单个时间点（增强版）"""
        # 处理延迟订单
        if isinstance(self.broker, AdvancedSimulatedBroker):
            executed_orders = self.broker.process_pending_orders(timestamp)
            
            # 处理执行完成的订单
            for order in executed_orders:
                if order.status == OrderStatus.FILLED:
                    # 更新账户状态
                    self.account_manager.add_order(order)
                    
                    # 更新持仓
                    position = self.broker.get_position(order.symbol)
                    if position:
                        self.account_manager.update_position(order.symbol, position)
                    
                    # 通知策略订单成交
                    strategy_name = order.strategy_name
                    if strategy_name in self.strategies:
                        strategy = self.strategies[strategy_name]
                        order_event = OrderFillEvent(
                            order_id=order.order_id,
                            symbol=order.symbol,
                            side=order.side.value,
                            quantity=order.filled_quantity,
                            price=order.avg_price,
                            timestamp=order.update_time,
                            commission=order.commission,
                            commission_asset='USDT'
                        )
                        strategy.on_order_fill(order_event)
        
        # 继续原有的时间点处理逻辑
        signals_to_process = []
        
        # 为每个策略处理数据
        for name, strategy in self.strategies.items():
            strategy_config = self.strategy_configs[name]
            symbol = strategy_config.symbol
            
            # 获取该策略的数据
            if symbol not in self.market_data:
                continue
            
            df = self.market_data[symbol]
            
            # 获取到当前时间点的数据
            current_data = df[df['timestamp'] <= timestamp].copy()
            
            if current_data.empty:
                continue
            
            # 更新价格缓存和市场数据
            latest_price = current_data['close'].iloc[-1]
            latest_volume = current_data['volume'].iloc[-1] if 'volume' in current_data.columns else 0.0
            
            if isinstance(self.broker, AdvancedSimulatedBroker):
                self.broker.update_market_price(symbol, latest_price, latest_volume)
            
            if self.risk_manager:
                self.risk_manager.update_price_cache(symbol, latest_price)
            self.account_manager.update_price_cache(symbol, latest_price)
            
            # 策略处理数据
            try:
                signals = strategy.on_data(current_data)
                
                # 收集有效信号
                for signal in signals:
                    if strategy.validate_signal(signal):
                        signals_to_process.append((name, signal, latest_price))
            
            except Exception as e:
                print(f"策略 {name} 处理数据失败: {e}")
                strategy.on_error(e)
        
        # 处理所有信号
        for strategy_name, signal, current_price in signals_to_process:
            self._process_signal(strategy_name, signal, current_price, config)
    
    def _calculate_final_results(self, result: BacktestResult) -> None:
        """计算最终结果（增强版）"""
        # 调用原有的结果计算
        super()._calculate_final_results(result)
        
        # 添加高级回测特有的统计信息
        if isinstance(self.broker, AdvancedSimulatedBroker):
            execution_stats = self.broker.get_execution_statistics()
            # 将高级回测信息存储在chart_paths中，因为BacktestResult没有metadata字段
            result.chart_paths.update({
                'advanced_backtest_stats': 'advanced_backtest_statistics.json',
                'execution_statistics': execution_stats,
                'latency_profile': self.latency_profile.value,
                'slippage_config': self.slippage_config.__dict__ if self.slippage_config else None
            })
        
        print("高级回测结果计算完成")