"""
账户管理模块 - 负责账户状态管理和资金管理

这个模块提供了统一的账户管理接口，支持：
- 账户余额管理
- 资金使用率计算
- 收益统计和分析
- 风险指标计算
- 交易记录管理
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
from enum import Enum
import numpy as np

from .broker import Order, Position, OrderStatus, OrderSide


class AccountType(Enum):
    """账户类型枚举"""
    SPOT = "SPOT"                # 现货账户
    MARGIN = "MARGIN"            # 杠杆账户
    FUTURES = "FUTURES"          # 合约账户
    SIMULATED = "SIMULATED"      # 模拟账户


@dataclass
class AccountBalance:
    """账户余额数据结构"""
    asset: str                   # 资产符号
    free: float                  # 可用余额
    locked: float                # 冻结余额
    total: float                 # 总余额
    value_in_usdt: float = 0.0   # USDT估值
    
    def __post_init__(self):
        """初始化后处理"""
        if self.total == 0:
            self.total = self.free + self.locked


@dataclass
class TradeRecord:
    """交易记录数据结构"""
    trade_id: str                # 交易ID
    symbol: str                  # 交易对
    side: OrderSide              # 买卖方向
    quantity: float              # 交易数量
    price: float                 # 成交价格
    commission: float            # 手续费
    commission_asset: str        # 手续费币种
    realized_pnl: float          # 已实现盈亏
    timestamp: datetime          # 交易时间
    
    # 策略信息
    strategy_name: str = ""      # 策略名称
    order_id: str = ""           # 订单ID
    
    # 额外信息
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'commission_asset': self.commission_asset,
            'realized_pnl': self.realized_pnl,
            'timestamp': self.timestamp.isoformat(),
            'strategy_name': self.strategy_name,
            'order_id': self.order_id,
            'metadata': self.metadata
        }


@dataclass
class PerformanceMetrics:
    """绩效指标数据结构"""
    # 基本指标
    total_trades: int = 0        # 总交易次数
    winning_trades: int = 0      # 盈利交易次数
    losing_trades: int = 0       # 亏损交易次数
    
    # 收益指标
    total_pnl: float = 0.0       # 总盈亏
    total_return: float = 0.0    # 总收益率
    avg_return: float = 0.0      # 平均收益率
    
    # 风险指标
    max_drawdown: float = 0.0    # 最大回撤
    volatility: float = 0.0      # 波动率
    sharpe_ratio: float = 0.0    # 夏普比率
    
    # 胜率指标
    win_rate: float = 0.0        # 胜率
    avg_win: float = 0.0         # 平均盈利
    avg_loss: float = 0.0        # 平均亏损
    profit_factor: float = 0.0   # 盈利因子
    
    # 时间指标
    avg_trade_duration: float = 0.0  # 平均交易持续时间（小时）
    
    # 资金使用指标
    avg_position_size: float = 0.0   # 平均仓位大小
    max_position_size: float = 0.0   # 最大仓位大小
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'total_return': self.total_return,
            'avg_return': self.avg_return,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'avg_trade_duration': self.avg_trade_duration,
            'avg_position_size': self.avg_position_size,
            'max_position_size': self.max_position_size
        }


class AccountManager:
    """账户管理器"""
    
    def __init__(self, account_type: AccountType = AccountType.SPOT):
        """
        初始化账户管理器
        
        Args:
            account_type: 账户类型
        """
        self.account_type = account_type
        
        # 账户数据
        self.balances: Dict[str, AccountBalance] = {}        # 账户余额
        self.positions: Dict[str, Position] = {}             # 持仓信息
        self.orders: Dict[str, Order] = {}                   # 订单记录
        self.trades: List[TradeRecord] = []                  # 交易记录
        
        # 初始资金
        self.initial_balance: Dict[str, float] = {}          # 初始余额
        self.initial_value_usdt: float = 0.0                 # 初始USDT估值
        
        # 绩效数据
        self.daily_values: List[Dict[str, Any]] = []         # 每日账户价值
        self.performance_metrics: Optional[PerformanceMetrics] = None
        
        # 价格缓存（用于计算估值）
        self.price_cache: Dict[str, float] = {}              # 价格缓存
        
        print(f"账户管理器初始化完成，账户类型: {account_type.value}")
    
    def set_initial_balance(self, balances: Dict[str, float]) -> None:
        """
        设置初始余额
        
        Args:
            balances: 初始余额字典
        """
        self.initial_balance = balances.copy()
        
        # 创建账户余额对象
        for asset, amount in balances.items():
            self.balances[asset] = AccountBalance(
                asset=asset,
                free=amount,
                locked=0.0,
                total=amount
            )
        
        # 计算初始USDT估值
        self.initial_value_usdt = self._calculate_total_value_usdt()
        
        print(f"设置初始余额: {balances}")
        print(f"初始USDT估值: {self.initial_value_usdt:.2f}")
    
    def update_balance(self, asset: str, free: float, locked: float) -> None:
        """
        更新账户余额
        
        Args:
            asset: 资产符号
            free: 可用余额
            locked: 冻结余额
        """
        if asset not in self.balances:
            self.balances[asset] = AccountBalance(asset=asset, free=0.0, locked=0.0, total=0.0)
        
        balance = self.balances[asset]
        balance.free = free
        balance.locked = locked
        balance.total = free + locked
        
        # 更新USDT估值
        balance.value_in_usdt = self._calculate_asset_value_usdt(asset, balance.total)
    
    def update_position(self, symbol: str, position: Position) -> None:
        """
        更新持仓信息
        
        Args:
            symbol: 交易对符号
            position: 持仓对象
        """
        if position.quantity == 0:
            # 平仓，删除持仓记录
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            # 更新持仓
            self.positions[symbol] = position
    
    def add_order(self, order: Order) -> None:
        """
        添加订单记录
        
        Args:
            order: 订单对象
        """
        self.orders[order.order_id] = order
        
        # 如果订单已成交，创建交易记录
        if order.status == OrderStatus.FILLED and order.filled_quantity > 0:
            self._create_trade_record(order)
    
    def update_order(self, order: Order) -> None:
        """
        更新订单状态
        
        Args:
            order: 订单对象
        """
        if order.order_id in self.orders:
            old_order = self.orders[order.order_id]
            old_filled_qty = old_order.filled_quantity
            
            # 更新订单
            self.orders[order.order_id] = order
            
            # 如果有新的成交，创建交易记录
            if order.filled_quantity > old_filled_qty:
                new_filled_qty = order.filled_quantity - old_filled_qty
                self._create_trade_record_partial(order, new_filled_qty)
    
    def _create_trade_record(self, order: Order) -> None:
        """创建交易记录（完整成交）"""
        trade = TradeRecord(
            trade_id=f"trade_{order.order_id}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.filled_quantity,
            price=order.avg_price,
            commission=order.commission,
            commission_asset=order.commission_asset,
            realized_pnl=0.0,  # 需要根据持仓计算
            timestamp=order.update_time,
            strategy_name=order.strategy_name,
            order_id=order.order_id,
            metadata=order.metadata
        )
        
        # 计算已实现盈亏
        trade.realized_pnl = self._calculate_realized_pnl(trade)
        
        self.trades.append(trade)
        print(f"创建交易记录: {trade.symbol} {trade.side.value} {trade.quantity} @ {trade.price}")
    
    def _create_trade_record_partial(self, order: Order, filled_qty: float) -> None:
        """创建交易记录（部分成交）"""
        trade = TradeRecord(
            trade_id=f"trade_{order.order_id}_{len(self.trades)}",
            symbol=order.symbol,
            side=order.side,
            quantity=filled_qty,
            price=order.avg_price,
            commission=order.commission * (filled_qty / order.filled_quantity),
            commission_asset=order.commission_asset,
            realized_pnl=0.0,
            timestamp=order.update_time,
            strategy_name=order.strategy_name,
            order_id=order.order_id,
            metadata=order.metadata
        )
        
        # 计算已实现盈亏
        trade.realized_pnl = self._calculate_realized_pnl(trade)
        
        self.trades.append(trade)
    
    def _calculate_realized_pnl(self, trade: TradeRecord) -> float:
        """
        计算已实现盈亏
        
        Args:
            trade: 交易记录
            
        Returns:
            float: 已实现盈亏
        """
        # 简化版本：基于持仓计算
        position = self.positions.get(trade.symbol)
        if position:
            return position.realized_pnl
        return 0.0
    
    def update_price_cache(self, symbol: str, price: float) -> None:
        """
        更新价格缓存
        
        Args:
            symbol: 交易对符号
            price: 当前价格
        """
        self.price_cache[symbol] = price
        
        # 更新相关资产的USDT估值
        base_asset = symbol.replace('USDT', '').replace('BUSD', '')
        if base_asset in self.balances:
            balance = self.balances[base_asset]
            balance.value_in_usdt = self._calculate_asset_value_usdt(base_asset, balance.total)
    
    def _calculate_asset_value_usdt(self, asset: str, amount: float) -> float:
        """
        计算资产的USDT估值
        
        Args:
            asset: 资产符号
            amount: 数量
            
        Returns:
            float: USDT估值
        """
        if asset == 'USDT' or asset == 'BUSD':
            return amount
        
        # 查找价格
        price_symbol = f"{asset}USDT"
        if price_symbol in self.price_cache:
            return amount * self.price_cache[price_symbol]
        
        # 如果没有直接的USDT价格，尝试通过BTC估值
        btc_symbol = f"{asset}BTC"
        if btc_symbol in self.price_cache and "BTCUSDT" in self.price_cache:
            btc_price = self.price_cache[btc_symbol]
            btc_usdt_price = self.price_cache["BTCUSDT"]
            return amount * btc_price * btc_usdt_price
        
        # 默认返回0
        return 0.0
    
    def _calculate_total_value_usdt(self) -> float:
        """
        计算总资产的USDT估值
        
        Returns:
            float: 总USDT估值
        """
        total_value = 0.0
        
        for balance in self.balances.values():
            total_value += balance.value_in_usdt
        
        return total_value
    
    def get_account_summary(self) -> Dict[str, Any]:
        """
        获取账户摘要
        
        Returns:
            Dict[str, Any]: 账户摘要信息
        """
        total_value = self._calculate_total_value_usdt()
        total_pnl = total_value - self.initial_value_usdt
        total_return = (total_pnl / self.initial_value_usdt) if self.initial_value_usdt > 0 else 0.0
        
        return {
            'account_type': self.account_type.value,
            'total_value_usdt': total_value,
            'initial_value_usdt': self.initial_value_usdt,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'active_positions': len(self.positions),
            'open_orders': len([o for o in self.orders.values() if o.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]]),
            'balances': {asset: balance.total for asset, balance in self.balances.items() if balance.total > 0},
            'positions': {symbol: pos.quantity for symbol, pos in self.positions.items()}
        }
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """
        计算绩效指标
        
        Returns:
            PerformanceMetrics: 绩效指标对象
        """
        if not self.trades:
            return PerformanceMetrics()
        
        metrics = PerformanceMetrics()
        
        # 基本统计
        metrics.total_trades = len(self.trades)
        
        # 计算盈亏
        total_pnl = sum(trade.realized_pnl for trade in self.trades)
        winning_trades = [trade for trade in self.trades if trade.realized_pnl > 0]
        losing_trades = [trade for trade in self.trades if trade.realized_pnl < 0]
        
        metrics.winning_trades = len(winning_trades)
        metrics.losing_trades = len(losing_trades)
        metrics.total_pnl = total_pnl
        
        # 收益率
        if self.initial_value_usdt > 0:
            metrics.total_return = total_pnl / self.initial_value_usdt
            metrics.avg_return = metrics.total_return / len(self.trades) if len(self.trades) > 0 else 0
        
        # 胜率和平均盈亏
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
            
            if winning_trades:
                metrics.avg_win = sum(trade.realized_pnl for trade in winning_trades) / len(winning_trades)
            
            if losing_trades:
                metrics.avg_loss = sum(trade.realized_pnl for trade in losing_trades) / len(losing_trades)
            
            # 盈利因子
            total_profit = sum(trade.realized_pnl for trade in winning_trades)
            total_loss = abs(sum(trade.realized_pnl for trade in losing_trades))
            metrics.profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # 最大回撤（简化计算）
        if len(self.daily_values) > 1:
            values = [day['total_value'] for day in self.daily_values]
            peak = values[0]
            max_drawdown = 0
            
            for value in values[1:]:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            metrics.max_drawdown = max_drawdown
        
        # 波动率（基于每日收益率）
        if len(self.daily_values) > 1:
            daily_returns = []
            for i in range(1, len(self.daily_values)):
                prev_value = self.daily_values[i-1]['total_value']
                curr_value = self.daily_values[i]['total_value']
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    daily_returns.append(daily_return)
            
            if daily_returns:
                metrics.volatility = np.std(daily_returns) * np.sqrt(365)  # 年化波动率
                
                # 夏普比率（假设无风险利率为3%）
                avg_daily_return = np.mean(daily_returns)
                risk_free_rate = 0.03 / 365  # 每日无风险利率
                if metrics.volatility > 0:
                    metrics.sharpe_ratio = (avg_daily_return - risk_free_rate) / (metrics.volatility / np.sqrt(365))
        
        # 仓位统计
        position_sizes = []
        for trade in self.trades:
            position_value = trade.quantity * trade.price
            position_sizes.append(position_value)
        
        if position_sizes:
            metrics.avg_position_size = np.mean(position_sizes)
            metrics.max_position_size = max(position_sizes)
        
        self.performance_metrics = metrics
        return metrics
    
    def record_daily_value(self) -> None:
        """记录每日账户价值"""
        total_value = self._calculate_total_value_usdt()
        
        # 使用isoformat()确保date可以被JSON序列化
        today = datetime.now().date()
        daily_record = {
            'date': today.isoformat(),  # 转换为字符串格式
            'total_value': total_value,
            'total_pnl': total_value - self.initial_value_usdt,
            'positions': len(self.positions),
            'trades_count': len(self.trades)
        }
        
        # 检查是否已有今日记录（比较字符串格式的日期）
        today_str = today.isoformat()
        existing_record = next((record for record in self.daily_values if record['date'] == today_str), None)
        
        if existing_record:
            # 更新今日记录
            existing_record.update(daily_record)
        else:
            # 添加新记录
            self.daily_values.append(daily_record)
    
    def get_trades_df(self) -> pd.DataFrame:
        """
        获取交易记录DataFrame
        
        Returns:
            pd.DataFrame: 交易记录DataFrame
        """
        if not self.trades:
            return pd.DataFrame()
        
        data = [trade.to_dict() for trade in self.trades]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def get_daily_values_df(self) -> pd.DataFrame:
        """
        获取每日价值DataFrame
        
        Returns:
            pd.DataFrame: 每日价值DataFrame
        """
        if not self.daily_values:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.daily_values)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def export_summary_report(self) -> Dict[str, Any]:
        """
        导出详细报告
        
        Returns:
            Dict[str, Any]: 完整的账户报告
        """
        summary = self.get_account_summary()
        metrics = self.calculate_performance_metrics()
        
        return {
            'account_summary': summary,
            'performance_metrics': metrics.to_dict(),
            'recent_trades': [trade.to_dict() for trade in self.trades[-10:]],  # 最近10笔交易
            'positions': {symbol: pos.to_dict() for symbol, pos in self.positions.items()},
            'daily_values': self.daily_values[-30:] if len(self.daily_values) > 30 else self.daily_values,  # 最近30天
            'report_time': datetime.now().isoformat()
        }
    
    def reset(self) -> None:
        """重置账户管理器"""
        self.balances.clear()
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.daily_values.clear()
        self.price_cache.clear()
        self.performance_metrics = None
        self.initial_value_usdt = 0.0
        print("账户管理器已重置")