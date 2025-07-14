"""
回测引擎模块 - 负责策略回测和性能分析

这个模块提供了完整的回测功能，包括：
- 历史数据回测
- 多策略组合回测
- 详细的性能分析
- 可视化图表生成
- 回测结果导出
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from enum import Enum
import json
import os

from ..strategies.base import Strategy, StrategyConfig, TradeSignal, SignalType, OrderFillEvent
from .data import DataManager, KlineData
from .broker import SimulatedBroker, Order, OrderSide, OrderStatus
from .account import AccountManager, TradeRecord, PerformanceMetrics
from .risk import RiskManager, RiskLimits


class BacktestStatus(Enum):
    """回测状态枚举"""
    PREPARING = "PREPARING"      # 准备中
    RUNNING = "RUNNING"          # 运行中
    COMPLETED = "COMPLETED"      # 已完成
    FAILED = "FAILED"            # 失败
    STOPPED = "STOPPED"          # 已停止


@dataclass
class BacktestConfig:
    """回测配置"""
    # 时间范围
    start_date: datetime             # 开始时间
    end_date: datetime               # 结束时间
    
    # 初始资金
    initial_balance: Dict[str, float] = field(default_factory=lambda: {'USDT': 10000.0})
    
    # 交易费用
    commission_rate: float = 0.001   # 手续费率
    slippage: float = 0.0001         # 滑点
    
    # 数据配置
    data_provider: str = "binance"   # 数据提供者
    timeframe: str = "1h"            # 时间周期
    
    # 风险控制
    enable_risk_management: bool = True  # 是否启用风险管理
    risk_limits: Optional[RiskLimits] = None  # 风险限制
    
    # 输出配置
    output_dir: str = "backtest_results"  # 输出目录
    save_trades: bool = True         # 是否保存交易记录
    save_charts: bool = True         # 是否保存图表
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_balance': self.initial_balance,
            'commission_rate': self.commission_rate,
            'slippage': self.slippage,
            'data_provider': self.data_provider,
            'timeframe': self.timeframe,
            'enable_risk_management': self.enable_risk_management,
            'output_dir': self.output_dir,
            'save_trades': self.save_trades,
            'save_charts': self.save_charts
        }


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    config: BacktestConfig           # 回测配置
    status: BacktestStatus           # 回测状态
    start_time: datetime             # 开始时间
    end_time: Optional[datetime] = None  # 结束时间
    
    # 性能指标
    performance_metrics: Optional[PerformanceMetrics] = None
    
    # 详细数据
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)     # 权益曲线
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)           # 交易记录
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)        # 持仓记录
    
    # 策略数据
    strategy_signals: Dict[str, pd.DataFrame] = field(default_factory=dict)  # 策略信号
    
    # 图表路径
    chart_paths: Dict[str, str] = field(default_factory=dict)
    
    # 错误信息
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'config': self.config.to_dict(),
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'performance_metrics': self.performance_metrics.to_dict() if self.performance_metrics else None,
            'trades_count': len(self.trades),
            'positions_count': len(self.positions),
            'chart_paths': self.chart_paths,
            'error_message': self.error_message
        }


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, data_manager: DataManager):
        """
        初始化回测引擎
        
        Args:
            data_manager: 数据管理器
        """
        self.data_manager = data_manager
        
        # 回测组件
        self.broker: Optional[SimulatedBroker] = None
        self.account_manager: Optional[AccountManager] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # 策略管理
        self.strategies: Dict[str, Strategy] = {}
        self.strategy_configs: Dict[str, StrategyConfig] = {}
        
        # 回测状态
        self.current_result: Optional[BacktestResult] = None
        self.is_running = False
        
        # 数据缓存
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        print("回测引擎初始化完成")
    
    def add_strategy(self, name: str, strategy: Strategy, config: StrategyConfig) -> None:
        """
        添加策略
        
        Args:
            name: 策略名称
            strategy: 策略实例
            config: 策略配置
        """
        self.strategies[name] = strategy
        self.strategy_configs[name] = config
        print(f"添加策略: {name}")
    
    def remove_strategy(self, name: str) -> None:
        """
        移除策略
        
        Args:
            name: 策略名称
        """
        if name in self.strategies:
            del self.strategies[name]
            del self.strategy_configs[name]
            print(f"移除策略: {name}")
    
    def run_backtest(self, config: BacktestConfig) -> BacktestResult:
        """
        运行回测
        
        Args:
            config: 回测配置
            
        Returns:
            BacktestResult: 回测结果
        """
        # 创建回测结果对象
        result = BacktestResult(
            config=config,
            status=BacktestStatus.PREPARING,
            start_time=datetime.now()
        )
        
        self.current_result = result
        self.is_running = True
        
        try:
            print(f"开始回测: {config.start_date} - {config.end_date}")
            
            # 1. 初始化回测组件
            self._initialize_backtest_components(config)
            
            # 2. 加载历史数据
            self._load_historical_data(config)
            
            # 3. 初始化策略
            self._initialize_strategies()
            
            # 4. 运行回测循环
            result.status = BacktestStatus.RUNNING
            self._run_backtest_loop(config)
            
            # 5. 计算最终结果
            self._calculate_final_results(result)
            
            # 6. 生成图表
            if config.save_charts:
                self._generate_charts(result)
            
            # 7. 保存结果
            self._save_results(result)
            
            result.status = BacktestStatus.COMPLETED
            result.end_time = datetime.now()
            
            print(f"回测完成，用时: {result.end_time - result.start_time}")
            
        except Exception as e:
            result.status = BacktestStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            print(f"回测失败: {e}")
        
        finally:
            self.is_running = False
        
        return result
    
    def _initialize_backtest_components(self, config: BacktestConfig) -> None:
        """初始化回测组件"""
        # 创建模拟经纪商
        self.broker = SimulatedBroker(
            initial_balance=config.initial_balance,
            commission_rate=config.commission_rate
        )
        
        # 创建账户管理器
        self.account_manager = AccountManager()
        self.account_manager.set_initial_balance(config.initial_balance)
        
        # 创建风险管理器
        if config.enable_risk_management:
            risk_limits = config.risk_limits or RiskLimits()
            self.risk_manager = RiskManager(risk_limits)
    
    def _load_historical_data(self, config: BacktestConfig) -> None:
        """加载历史数据"""
        print("加载历史数据...")
        
        # 获取所有需要的交易对
        symbols = set()
        for strategy_config in self.strategy_configs.values():
            symbols.add(strategy_config.symbol)
        
        # 加载每个交易对的数据
        for symbol in symbols:
            print(f"加载 {symbol} 数据...")
            
            df = self.data_manager.get_historical_klines(
                symbol=symbol,
                interval=config.timeframe,
                start_time=config.start_date,
                end_time=config.end_date,
                provider=config.data_provider
            )
            
            if df.empty:
                raise ValueError(f"无法获取 {symbol} 的历史数据")
            
            self.market_data[symbol] = df
            print(f"加载 {symbol} 数据完成，共 {len(df)} 条记录")
    
    def _initialize_strategies(self) -> None:
        """初始化策略"""
        print("初始化策略...")
        
        for name, strategy in self.strategies.items():
            strategy.initialize()
            print(f"策略 {name} 初始化完成")
    
    def _run_backtest_loop(self, config: BacktestConfig) -> None:
        """运行回测循环"""
        print("开始回测循环...")
        
        # 获取所有数据的时间范围
        all_timestamps = set()
        for df in self.market_data.values():
            all_timestamps.update(df['timestamp'].tolist())
        
        # 按时间排序
        sorted_timestamps = sorted(all_timestamps)
        
        # 逐时间点处理
        for i, timestamp in enumerate(sorted_timestamps):
            # 显示进度
            if i % 100 == 0:
                progress = (i + 1) / len(sorted_timestamps) * 100
                print(f"回测进度: {progress:.1f}% ({i+1}/{len(sorted_timestamps)})")
            
            # 处理当前时间点
            self._process_timestamp(timestamp, config)
            
            # 记录每日账户价值
            if timestamp.hour == 0:  # 每日零点记录
                self.account_manager.record_daily_value()
    
    def _process_timestamp(self, timestamp: datetime, config: BacktestConfig) -> None:
        """处理单个时间点"""
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
            
            # 更新价格缓存
            latest_price = current_data['close'].iloc[-1]
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
    
    def _process_signal(self, strategy_name: str, signal: TradeSignal, current_price: float, config: BacktestConfig) -> None:
        """处理交易信号"""
        try:
            # 计算交易数量
            quantity = self._calculate_quantity(signal, current_price)
            
            if quantity <= 0:
                return
            
            # 风险检查
            if self.risk_manager:
                risk_report = self.risk_manager.check_signal_risk(
                    signal, self.account_manager, quantity, current_price
                )
                
                if risk_report.result.value != "PASS":
                    print(f"信号被风控拒绝: {risk_report.message}")
                    return
            
            # 下单
            order = self.broker.place_order(signal, quantity, signal.price)
            
            if order and order.status == OrderStatus.FILLED:
                # 更新账户状态
                self.account_manager.add_order(order)
                
                # 更新持仓
                position = self.broker.get_position(signal.symbol)
                if position:
                    self.account_manager.update_position(signal.symbol, position)
                
                # 记录交易（用于风控）
                if self.risk_manager:
                    self.risk_manager.record_trade(
                        signal.symbol, order.side, order.filled_quantity, order.avg_price, strategy_name
                    )
                
                # 通知策略订单成交
                strategy = self.strategies.get(strategy_name)
                if strategy:
                    order_event = OrderFillEvent(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side.value,
                        quantity=order.filled_quantity,
                        price=order.avg_price,
                        timestamp=order.update_time,
                        commission=order.commission,
                        commission_asset=order.commission_asset
                    )
                    strategy.on_order_fill(order_event)
                
                print(f"订单成交: {strategy_name} - {order.symbol} {order.side.value} {order.filled_quantity} @ {order.avg_price}")
        
        except Exception as e:
            print(f"处理信号失败: {e}")
    
    def _calculate_quantity(self, signal: TradeSignal, current_price: float) -> float:
        """计算交易数量"""
        if signal.quantity is not None:
            return signal.quantity
        
        if signal.quantity_percent is not None:
            # 根据百分比计算数量
            account_summary = self.account_manager.get_account_summary()
            total_value = account_summary.get('total_value_usdt', 0)
            
            target_value = total_value * signal.quantity_percent
            quantity = target_value / current_price
            
            return quantity
        
        # 默认使用最小交易数量
        return 0.001
    
    def _calculate_final_results(self, result: BacktestResult) -> None:
        """计算最终结果"""
        print("计算最终结果...")
        
        # 计算性能指标
        result.performance_metrics = self.account_manager.calculate_performance_metrics()
        
        # 生成权益曲线
        result.equity_curve = self.account_manager.get_daily_values_df()
        
        # 生成交易记录
        result.trades = self.account_manager.get_trades_df()
        
        # 生成持仓记录
        positions_data = []
        for symbol, position in self.account_manager.positions.items():
            positions_data.append(position.to_dict())
        
        if positions_data:
            result.positions = pd.DataFrame(positions_data)
        
        # 生成策略信号记录
        for name, strategy in self.strategies.items():
            if strategy.signal_history:
                signals_data = []
                for signal in strategy.signal_history:
                    signals_data.append({
                        'timestamp': signal.timestamp,
                        'symbol': signal.symbol,
                        'signal_type': signal.signal_type.value,
                        'price': signal.price,
                        'quantity': signal.quantity,
                        'confidence': signal.confidence,
                        'strategy_name': signal.strategy_name
                    })
                
                result.strategy_signals[name] = pd.DataFrame(signals_data)
        
        print("最终结果计算完成")
    
    def _generate_charts(self, result: BacktestResult) -> None:
        """生成图表"""
        print("生成图表...")
        
        # 确保输出目录存在
        output_dir = result.config.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 生成权益曲线图
        self._generate_equity_curve_chart(result, output_dir)
        
        # 生成交易图
        self._generate_trades_chart(result, output_dir)
        
        # 生成绩效分析图
        self._generate_performance_charts(result, output_dir)
        
        print("图表生成完成")
    
    def _generate_equity_curve_chart(self, result: BacktestResult, output_dir: str) -> None:
        """生成权益曲线图"""
        if result.equity_curve.empty:
            return
        
        # 使用Plotly生成交互式图表
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=result.equity_curve['date'],
            y=result.equity_curve['total_value'],
            mode='lines',
            name='账户价值',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=result.equity_curve['date'],
            y=result.equity_curve['total_pnl'],
            mode='lines',
            name='盈亏',
            line=dict(color='green', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            title='权益曲线',
            xaxis_title='日期',
            yaxis_title='账户价值 (USDT)',
            yaxis2=dict(
                title='盈亏 (USDT)',
                overlaying='y',
                side='right'
            ),
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        # 保存图表
        chart_path = os.path.join(output_dir, 'equity_curve.html')
        fig.write_html(chart_path)
        result.chart_paths['equity_curve'] = chart_path
        
        print(f"权益曲线图已保存: {chart_path}")
    
    def _generate_trades_chart(self, result: BacktestResult, output_dir: str) -> None:
        """生成交易图"""
        if result.trades.empty:
            return
        
        # 按策略分组统计
        strategy_stats = result.trades.groupby('strategy_name').agg({
            'trade_id': 'count',
            'realized_pnl': ['sum', 'mean']
        }).round(2)
        
        strategy_stats.columns = ['交易次数', '总盈亏', '平均盈亏']
        
        # 生成柱状图
        fig = go.Figure()
        
        strategies = strategy_stats.index.tolist()
        
        fig.add_trace(go.Bar(
            x=strategies,
            y=strategy_stats['总盈亏'],
            name='总盈亏',
            marker_color=['green' if x > 0 else 'red' for x in strategy_stats['总盈亏']]
        ))
        
        fig.update_layout(
            title='各策略盈亏统计',
            xaxis_title='策略名称',
            yaxis_title='盈亏 (USDT)',
            template='plotly_white',
            height=500
        )
        
        # 保存图表
        chart_path = os.path.join(output_dir, 'trades_by_strategy.html')
        fig.write_html(chart_path)
        result.chart_paths['trades_by_strategy'] = chart_path
        
        print(f"交易统计图已保存: {chart_path}")
    
    def _generate_performance_charts(self, result: BacktestResult, output_dir: str) -> None:
        """生成绩效分析图"""
        if not result.performance_metrics:
            return
        
        metrics = result.performance_metrics
        
        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('胜率分析', '盈亏分布', '交易统计', '风险指标'),
            specs=[[{'type': 'pie'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 胜率饼图
        fig.add_trace(go.Pie(
            labels=['盈利交易', '亏损交易'],
            values=[metrics.winning_trades, metrics.losing_trades],
            name="胜率"
        ), row=1, col=1)
        
        # 盈亏分布
        fig.add_trace(go.Bar(
            x=['平均盈利', '平均亏损'],
            y=[metrics.avg_win, metrics.avg_loss],
            name="盈亏分布",
            marker_color=['green', 'red']
        ), row=1, col=2)
        
        # 交易统计
        fig.add_trace(go.Bar(
            x=['总交易次数', '盈利交易', '亏损交易'],
            y=[metrics.total_trades, metrics.winning_trades, metrics.losing_trades],
            name="交易统计",
            marker_color=['blue', 'green', 'red']
        ), row=2, col=1)
        
        # 风险指标
        fig.add_trace(go.Bar(
            x=['总收益率', '最大回撤', '夏普比率'],
            y=[metrics.total_return, metrics.max_drawdown, metrics.sharpe_ratio],
            name="风险指标",
            marker_color=['blue', 'orange', 'purple']
        ), row=2, col=2)
        
        fig.update_layout(
            title_text="绩效分析dashboard",
            showlegend=False,
            height=800
        )
        
        # 保存图表
        chart_path = os.path.join(output_dir, 'performance_analysis.html')
        fig.write_html(chart_path)
        result.chart_paths['performance_analysis'] = chart_path
        
        print(f"绩效分析图已保存: {chart_path}")
    
    def _save_results(self, result: BacktestResult) -> None:
        """保存结果"""
        output_dir = result.config.output_dir
        
        # 保存回测配置和结果摘要
        summary_path = os.path.join(output_dir, 'backtest_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        
        # 保存详细交易记录
        if result.config.save_trades and not result.trades.empty:
            trades_path = os.path.join(output_dir, 'trades.csv')
            result.trades.to_csv(trades_path, index=False, encoding='utf-8')
        
        # 保存权益曲线数据
        if not result.equity_curve.empty:
            equity_path = os.path.join(output_dir, 'equity_curve.csv')
            result.equity_curve.to_csv(equity_path, index=False, encoding='utf-8')
        
        # 保存账户报告
        if self.account_manager:
            account_report = self.account_manager.export_summary_report()
            account_path = os.path.join(output_dir, 'account_report.json')
            with open(account_path, 'w', encoding='utf-8') as f:
                json.dump(account_report, f, ensure_ascii=False, indent=2)
        
        # 保存风险报告
        if self.risk_manager:
            risk_report = self.risk_manager.export_risk_report()
            risk_path = os.path.join(output_dir, 'risk_report.json')
            with open(risk_path, 'w', encoding='utf-8') as f:
                json.dump(risk_report, f, ensure_ascii=False, indent=2)
        
        print(f"回测结果已保存至: {output_dir}")
    
    def stop_backtest(self) -> None:
        """停止回测"""
        self.is_running = False
        if self.current_result:
            self.current_result.status = BacktestStatus.STOPPED
            self.current_result.end_time = datetime.now()
        print("回测已停止")
    
    def get_current_status(self) -> Optional[Dict[str, Any]]:
        """获取当前状态"""
        if not self.current_result:
            return None
        
        return {
            'status': self.current_result.status.value,
            'is_running': self.is_running,
            'start_time': self.current_result.start_time.isoformat(),
            'elapsed_time': str(datetime.now() - self.current_result.start_time),
            'strategies_count': len(self.strategies),
            'error_message': self.current_result.error_message
        }
    
    def clear_strategies(self) -> None:
        """清除所有策略"""
        self.strategies.clear()
        self.strategy_configs.clear()
        print("所有策略已清除")
    
    def reset(self) -> None:
        """重置回测引擎"""
        self.strategies.clear()
        self.strategy_configs.clear()
        self.market_data.clear()
        self.current_result = None
        self.is_running = False
        
        if self.broker:
            self.broker.reset()
        
        if self.account_manager:
            self.account_manager.reset()
        
        print("回测引擎已重置")