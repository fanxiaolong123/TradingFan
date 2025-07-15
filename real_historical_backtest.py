#!/usr/bin/env python3
"""
真实历史数据回测引擎

使用从Binance官方下载的大量历史数据进行回测
支持多年期的真实数据回测，提供准确的性能分析
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
from pathlib import Path
import logging

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_trader.strategies.momentum import MomentumStrategy
from auto_trader.strategies.mean_reversion import MeanReversionStrategy
from auto_trader.strategies.base import StrategyConfig, SignalType

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """交易记录"""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float
    signal_type: str = ""
    pnl: float = 0.0
    cumulative_pnl: float = 0.0

@dataclass
class Position:
    """持仓记录"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    total_cost: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_position(self, trade: Trade) -> None:
        """更新持仓"""
        if trade.side == 'BUY':
            # 买入
            new_quantity = self.quantity + trade.quantity
            if new_quantity > 0:
                self.total_cost += trade.quantity * trade.price + trade.commission
                self.avg_price = self.total_cost / new_quantity
            self.quantity = new_quantity
        elif trade.side == 'SELL':
            # 卖出
            if self.quantity > 0:
                sell_quantity = min(trade.quantity, self.quantity)
                # 计算实现盈亏
                realized_pnl = sell_quantity * (trade.price - self.avg_price) - trade.commission
                trade.pnl = realized_pnl
                
                # 更新持仓
                self.quantity -= sell_quantity
                if self.quantity > 0:
                    self.total_cost = self.quantity * self.avg_price
                else:
                    self.total_cost = 0.0
                    self.avg_price = 0.0
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """计算未实现盈亏"""
        if self.quantity > 0:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
        else:
            self.unrealized_pnl = 0.0
        return self.unrealized_pnl

@dataclass
class PerformanceMetrics:
    """性能指标"""
    # 基本指标
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # 风险指标
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0  # 95% VaR
    cvar_95: float = 0.0  # 95% CVaR
    
    # 交易指标
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # 其他指标
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # 资金曲线
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    
    # 时间相关
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0

class RealHistoricalBacktester:
    """真实历史数据回测引擎"""
    
    def __init__(self, data_dir: str = "binance_historical_data/processed", 
                 initial_capital: float = 100000.0, commission_rate: float = 0.001):
        """
        初始化回测引擎
        
        Args:
            data_dir: 历史数据目录
            initial_capital: 初始资金
            commission_rate: 手续费率
        """
        self.data_dir = Path(data_dir)
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        
        # 交易记录
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}
        self.portfolio_values: List[float] = []
        self.timestamps: List[datetime] = []
        
        # 当前状态
        self.current_capital = initial_capital
        self.current_timestamp = None
        
        # 历史数据缓存
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"✅ 真实历史数据回测引擎初始化完成")
        logger.info(f"📁 数据目录: {self.data_dir}")
        logger.info(f"💰 初始资金: {self.initial_capital:,.0f} USDT")
        logger.info(f"📊 手续费率: {self.commission_rate*100:.3f}%")
    
    def load_historical_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """
        加载历史数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            
        Returns:
            pd.DataFrame: 历史数据
        """
        cache_key = f"{symbol}_{interval}"
        
        # 检查缓存
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        # 构建文件路径
        filename = f"{symbol}_{interval}_combined.csv"
        file_path = self.data_dir / filename
        
        if not file_path.exists():
            logger.error(f"❌ 历史数据文件不存在: {file_path}")
            return None
        
        try:
            # 加载数据
            df = pd.read_csv(file_path)
            
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 验证数据
            if len(df) == 0:
                logger.error(f"❌ 空数据文件: {file_path}")
                return None
            
            # 缓存数据
            self.data_cache[cache_key] = df
            
            logger.info(f"✅ 加载历史数据: {symbol} {interval}")
            logger.info(f"📊 数据量: {len(df):,} 条记录")
            logger.info(f"📅 时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
            logger.info(f"🕐 时间跨度: {(df['timestamp'].max() - df['timestamp'].min()).days} 天")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ 加载数据失败: {file_path} - {str(e)}")
            return None
    
    def get_data_subset(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        获取指定时间范围的数据子集
        
        Args:
            df: 原始数据
            start_date: 开始时间
            end_date: 结束时间
            
        Returns:
            pd.DataFrame: 数据子集
        """
        # 过滤时间范围
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        subset = df[mask].copy()
        
        if len(subset) == 0:
            logger.warning(f"⚠️ 指定时间范围内没有数据: {start_date} 到 {end_date}")
            return pd.DataFrame()
        
        # 重置索引
        subset = subset.reset_index(drop=True)
        
        logger.info(f"📊 筛选数据: {len(subset):,} 条记录")
        logger.info(f"📅 实际范围: {subset['timestamp'].min()} 到 {subset['timestamp'].max()}")
        
        return subset
    
    def run_backtest(self, strategy_config: StrategyConfig, 
                    start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """
        运行回测
        
        Args:
            strategy_config: 策略配置
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            PerformanceMetrics: 性能指标
        """
        logger.info(f"🚀 开始真实历史数据回测")
        logger.info(f"📊 策略: {strategy_config.name}")
        logger.info(f"🪙 交易对: {strategy_config.symbol}")
        logger.info(f"⏱️ 时间周期: {strategy_config.timeframe}")
        logger.info(f"📅 回测时间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        
        # 重置状态
        self._reset()
        
        # 加载历史数据
        df = self.load_historical_data(strategy_config.symbol, strategy_config.timeframe)
        
        if df is None:
            logger.error("❌ 无法加载历史数据")
            return PerformanceMetrics()
        
        # 获取指定时间范围的数据
        data = self.get_data_subset(df, start_date, end_date)
        
        if data.empty:
            logger.error("❌ 指定时间范围内无数据")
            return PerformanceMetrics()
        
        # 创建策略
        strategy = self._create_strategy(strategy_config)
        strategy.initialize()
        
        # 运行回测
        self._run_strategy(strategy, data)
        
        # 计算性能指标
        metrics = self._calculate_metrics(start_date, end_date)
        
        return metrics
    
    def _reset(self) -> None:
        """重置回测状态"""
        self.trades.clear()
        self.positions.clear()
        self.portfolio_values.clear()
        self.timestamps.clear()
        self.current_capital = self.initial_capital
        self.current_timestamp = None
    
    def _create_strategy(self, config: StrategyConfig):
        """创建策略实例"""
        if 'momentum' in config.name.lower():
            return MomentumStrategy(config)
        elif 'mean_reversion' in config.name.lower():
            return MeanReversionStrategy(config)
        else:
            # 默认使用动量策略
            return MomentumStrategy(config)
    
    def _run_strategy(self, strategy, data: pd.DataFrame) -> None:
        """运行策略"""
        logger.info(f"🔄 开始策略运行...")
        
        # 设置时间戳索引
        data_indexed = data.set_index('timestamp')
        
        total_bars = len(data_indexed)
        processed_bars = 0
        
        for current_time, row in data_indexed.iterrows():
            self.current_timestamp = current_time
            processed_bars += 1
            
            # 显示进度
            if processed_bars % max(1, total_bars // 20) == 0:
                progress = processed_bars / total_bars * 100
                logger.info(f"   进度: {progress:.1f}% ({processed_bars}/{total_bars})")
            
            # 获取到当前时间的历史数据
            historical_data = data_indexed.loc[:current_time].copy()
            
            # 确保有足够的数据
            if len(historical_data) < 50:
                self._update_portfolio_value(row['close'])
                continue
            
            try:
                # 生成交易信号
                signals = strategy.on_data(historical_data)
                
                # 执行交易
                for signal in signals:
                    if strategy.validate_signal(signal):
                        self._execute_signal(signal, row['close'])
                
                # 更新投资组合价值
                self._update_portfolio_value(row['close'])
                
            except Exception as e:
                logger.warning(f"   ⚠️ 策略运行异常: {e}")
                continue
        
        logger.info(f"✅ 策略运行完成，共处理 {processed_bars:,} 个数据点")
    
    def _execute_signal(self, signal, current_price: float) -> None:
        """执行交易信号"""
        symbol = signal.symbol
        
        # 确保持仓记录存在
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        position = self.positions[symbol]
        
        # 计算交易数量
        if signal.quantity_percent:
            # 按百分比计算
            portfolio_value = self.current_capital + sum(
                pos.calculate_unrealized_pnl(current_price) 
                for pos in self.positions.values()
            )
            trade_value = portfolio_value * signal.quantity_percent
            quantity = trade_value / current_price
        else:
            # 固定数量
            quantity = signal.quantity or 0.1
        
        # 根据信号类型执行交易
        if signal.signal_type == SignalType.BUY:
            # 买入
            cost = quantity * current_price
            commission = cost * self.commission_rate
            total_cost = cost + commission
            
            if self.current_capital >= total_cost:
                trade = Trade(
                    timestamp=self.current_timestamp,
                    symbol=symbol,
                    side='BUY',
                    quantity=quantity,
                    price=current_price,
                    commission=commission,
                    signal_type=signal.signal_type.value
                )
                
                # 更新持仓
                position.update_position(trade)
                
                # 更新资金
                self.current_capital -= total_cost
                
                # 记录交易
                self.trades.append(trade)
        
        elif signal.signal_type == SignalType.SELL:
            # 卖出
            if position.quantity > 0:
                sell_quantity = min(quantity, position.quantity)
                revenue = sell_quantity * current_price
                commission = revenue * self.commission_rate
                net_revenue = revenue - commission
                
                trade = Trade(
                    timestamp=self.current_timestamp,
                    symbol=symbol,
                    side='SELL',
                    quantity=sell_quantity,
                    price=current_price,
                    commission=commission,
                    signal_type=signal.signal_type.value
                )
                
                # 更新持仓
                position.update_position(trade)
                
                # 更新资金
                self.current_capital += net_revenue
                
                # 记录交易
                self.trades.append(trade)
    
    def _update_portfolio_value(self, current_price: float) -> None:
        """更新投资组合价值"""
        total_value = self.current_capital
        
        # 加上所有持仓的市值
        for position in self.positions.values():
            unrealized_pnl = position.calculate_unrealized_pnl(current_price)
            total_value += position.quantity * current_price
        
        self.portfolio_values.append(total_value)
        self.timestamps.append(self.current_timestamp)
    
    def _calculate_metrics(self, start_date: datetime, end_date: datetime) -> PerformanceMetrics:
        """计算性能指标"""
        logger.info(f"📊 计算性能指标...")
        
        metrics = PerformanceMetrics()
        
        if not self.portfolio_values:
            return metrics
        
        # 基本信息
        metrics.start_date = start_date
        metrics.end_date = end_date
        metrics.trading_days = (end_date - start_date).days
        metrics.equity_curve = self.portfolio_values.copy()
        
        # 计算收益率
        initial_value = self.portfolio_values[0]
        final_value = self.portfolio_values[-1]
        
        metrics.total_return = (final_value - initial_value) / initial_value
        
        # 年化收益率
        years = metrics.trading_days / 365.25
        if years > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (1/years) - 1
        
        # 计算日收益率
        daily_returns = []
        for i in range(1, len(self.portfolio_values)):
            daily_return = (self.portfolio_values[i] - self.portfolio_values[i-1]) / self.portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        metrics.daily_returns = daily_returns
        
        if daily_returns:
            # 波动率
            metrics.volatility = np.std(daily_returns) * np.sqrt(252)  # 年化波动率
            
            # 夏普比率
            if metrics.volatility > 0:
                metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility
            
            # 索提诺比率
            negative_returns = [r for r in daily_returns if r < 0]
            if negative_returns:
                downside_deviation = np.std(negative_returns) * np.sqrt(252)
                if downside_deviation > 0:
                    metrics.sortino_ratio = metrics.annualized_return / downside_deviation
        
        # 最大回撤
        metrics.max_drawdown = self._calculate_max_drawdown()
        
        # 卡尔马比率
        if abs(metrics.max_drawdown) > 0:
            metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
        
        # 风险指标
        if daily_returns:
            metrics.var_95 = np.percentile(daily_returns, 5)
            metrics.cvar_95 = np.mean([r for r in daily_returns if r <= metrics.var_95])
        
        # 交易指标
        self._calculate_trade_metrics(metrics)
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.portfolio_values:
            return 0.0
        
        equity_curve = np.array(self.portfolio_values)
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        
        return np.min(drawdowns)
    
    def _calculate_trade_metrics(self, metrics: PerformanceMetrics) -> None:
        """计算交易指标"""
        if not self.trades:
            return
        
        metrics.total_trades = len(self.trades)
        
        # 计算交易盈亏
        trade_pnls = []
        for trade in self.trades:
            if trade.side == 'SELL' and trade.pnl != 0:
                trade_pnls.append(trade.pnl)
        
        if trade_pnls:
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            # 胜率
            metrics.win_rate = len(winning_trades) / len(trade_pnls)
            
            # 平均盈利和亏损
            if winning_trades:
                metrics.avg_win = np.mean(winning_trades)
            if losing_trades:
                metrics.avg_loss = np.mean(losing_trades)
            
            # 盈亏比
            if abs(metrics.avg_loss) > 0:
                metrics.profit_factor = abs(sum(winning_trades) / sum(losing_trades))
    
    def generate_report(self, metrics: PerformanceMetrics, strategy_name: str) -> str:
        """生成详细报告"""
        report = f"""
📊 真实历史数据回测报告 - {strategy_name}
{'='*80}

📅 回测周期: {metrics.start_date.strftime('%Y-%m-%d')} 到 {metrics.end_date.strftime('%Y-%m-%d')}
📈 交易天数: {metrics.trading_days} 天
💰 初始资金: {self.initial_capital:,.0f} USDT
💵 最终资金: {metrics.equity_curve[-1]:,.0f} USDT

🎯 收益指标
{'-'*50}
📈 总收益率: {metrics.total_return*100:.2f}%
📊 年化收益率: {metrics.annualized_return*100:.2f}%
📉 波动率: {metrics.volatility*100:.2f}%
⚡ 夏普比率: {metrics.sharpe_ratio:.3f}
📊 索提诺比率: {metrics.sortino_ratio:.3f}
📉 卡尔马比率: {metrics.calmar_ratio:.3f}

⚠️ 风险指标  
{'-'*50}
📉 最大回撤: {metrics.max_drawdown*100:.2f}%
📊 95% VaR: {metrics.var_95*100:.2f}%
📈 95% CVaR: {metrics.cvar_95*100:.2f}%

🔄 交易指标
{'-'*50}
📊 总交易数: {metrics.total_trades}
🎯 胜率: {metrics.win_rate*100:.1f}%
💰 盈亏比: {metrics.profit_factor:.2f}
📈 平均盈利: {metrics.avg_win:.2f} USDT
📉 平均亏损: {metrics.avg_loss:.2f} USDT

💡 评估建议
{'-'*50}
"""
        
        # 添加评估建议
        if metrics.annualized_return > 0.15:
            report += "✅ 策略表现优秀，年化收益率超过15%\n"
        elif metrics.annualized_return > 0.08:
            report += "✅ 策略表现良好，年化收益率超过8%\n"
        else:
            report += "⚠️ 策略表现一般，建议优化参数\n"
        
        if metrics.sharpe_ratio > 1.0:
            report += "✅ 夏普比率优秀，风险调整后收益良好\n"
        elif metrics.sharpe_ratio > 0.5:
            report += "✅ 夏普比率合理\n"
        else:
            report += "⚠️ 夏普比率偏低，风险相对较高\n"
        
        if abs(metrics.max_drawdown) < 0.1:
            report += "✅ 最大回撤控制良好\n"
        elif abs(metrics.max_drawdown) < 0.2:
            report += "✅ 最大回撤可接受\n"
        else:
            report += "⚠️ 最大回撤较大，需要加强风险控制\n"
        
        return report

def main():
    """主函数"""
    print("🚀 真实历史数据回测系统")
    print("=" * 80)
    
    # 创建回测引擎
    backtest_engine = RealHistoricalBacktester()
    
    # 定义回测时间范围（使用真实的多年数据）
    end_date = datetime(2024, 6, 30)  # 2024年6月30日
    start_date = datetime(2023, 1, 1)  # 2023年1月1日（18个月的数据）
    
    logger.info(f"🎯 回测时间范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"📊 回测时长: {(end_date - start_date).days} 天")
    
    # 策略配置
    strategies = [
        {
            "name": "BTC长期动量策略",
            "config": StrategyConfig(
                name="btc_long_term_momentum",
                symbol="BTCUSDT",
                timeframe="1h",
                parameters={
                    'short_ma_period': 24,   # 24小时均线
                    'long_ma_period': 72,    # 72小时均线
                    'rsi_period': 14,
                    'rsi_overbought': 70,
                    'rsi_oversold': 30,
                    'momentum_period': 48,   # 48小时动量
                    'momentum_threshold': 0.02,
                    'volume_threshold': 1.5,
                    'position_size': 0.3,
                    'stop_loss_pct': 0.03,
                    'take_profit_pct': 0.06
                }
            )
        },
        {
            "name": "ETH均值回归策略",
            "config": StrategyConfig(
                name="eth_mean_reversion",
                symbol="ETHUSDT",
                timeframe="1h",
                parameters={
                    'ma_period': 48,         # 48小时均线
                    'deviation_threshold': 0.02,  # 2%偏离阈值
                    'min_volume': 10,
                    'position_size': 0.25,
                    'stop_loss_pct': 0.025,
                    'take_profit_pct': 0.05
                }
            )
        }
    ]
    
    # 运行回测
    results = []
    
    for strategy_info in strategies:
        print(f"\n{'='*80}")
        print(f"🎯 回测策略: {strategy_info['name']}")
        print(f"{'='*80}")
        
        try:
            metrics = backtest_engine.run_backtest(
                strategy_config=strategy_info['config'],
                start_date=start_date,
                end_date=end_date
            )
            
            # 生成报告
            report = backtest_engine.generate_report(metrics, strategy_info['name'])
            print(report)
            
            results.append({
                'name': strategy_info['name'],
                'metrics': metrics,
                'report': report
            })
            
        except Exception as e:
            logger.error(f"❌ 策略回测失败: {e}")
            continue
    
    # 总结比较
    if results:
        print(f"\n{'='*80}")
        print(f"📊 策略比较总结")
        print(f"{'='*80}")
        
        print(f"{'策略名称':<30} {'总收益率':<12} {'年化收益率':<12} {'夏普比率':<10} {'最大回撤':<10}")
        print(f"{'-'*80}")
        
        for result in results:
            metrics = result['metrics']
            print(f"{result['name']:<30} {metrics.total_return*100:>10.2f}% {metrics.annualized_return*100:>10.2f}% "
                  f"{metrics.sharpe_ratio:>8.3f} {metrics.max_drawdown*100:>8.2f}%")
        
        # 推荐最佳策略
        if len(results) > 1:
            best_strategy = max(results, key=lambda x: x['metrics'].annualized_return)
            print(f"\n🏆 最佳策略: {best_strategy['name']}")
            print(f"📈 年化收益率: {best_strategy['metrics'].annualized_return*100:.2f}%")
            print(f"⚡ 夏普比率: {best_strategy['metrics'].sharpe_ratio:.3f}")
            print(f"📉 最大回撤: {best_strategy['metrics'].max_drawdown*100:.2f}%")
    
    print(f"\n🎉 真实历史数据回测完成!")
    return results

if __name__ == "__main__":
    results = main()