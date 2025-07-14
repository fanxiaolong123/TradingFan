#!/usr/bin/env python3
"""
量化自动交易系统主入口文件

这是AutoTrader量化交易系统的主要启动文件，提供了：
- 命令行界面和参数解析
- 系统初始化和配置加载
- 多种运行模式（回测、实盘、模拟等）
- 交互式管理和监控界面
"""

import sys
import os
import argparse
import signal
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入项目模块
from auto_trader.utils import (
    get_config, setup_logging, get_logger, 
    now, format_duration
)
from auto_trader.core.data import DataManager, BinanceDataProvider
from auto_trader.core.broker import BinanceBroker, SimulatedBroker
from auto_trader.core.account import AccountManager, AccountType
from auto_trader.core.risk import RiskManager, RiskLimits
from auto_trader.core.backtest import BacktestEngine, BacktestConfig
from auto_trader.strategies.base import StrategyConfig
from auto_trader.strategies.mean_reversion import MeanReversionStrategy


class TradingSystem:
    """交易系统主类"""
    
    def __init__(self):
        """初始化交易系统"""
        self.config = None
        self.logger = None
        self.data_manager = None
        self.broker = None
        self.account_manager = None
        self.risk_manager = None
        self.backtest_engine = None
        
        # 系统状态
        self.is_running = False
        self.current_mode = None
        
        # 策略管理
        self.active_strategies = {}
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize(self, config_file: str = "config.yml") -> None:
        """
        初始化系统
        
        Args:
            config_file: 配置文件路径
        """
        try:
            # 加载配置
            self.config = get_config()
            
            # 设置日志
            logging_config = self.config.get_logging_config()
            setup_logging(logging_config)
            self.logger = get_logger(__name__)
            
            self.logger.info("=" * 60)
            self.logger.info(f"🚀 {self.config.get('system.name', 'AutoTrader')} 系统启动")
            self.logger.info(f"版本: {self.config.get('system.version', '1.0.0')}")
            self.logger.info(f"环境: {self.config.get('system.environment', 'development')}")
            self.logger.info(f"启动时间: {now().strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info("=" * 60)
            
            # 初始化数据管理器
            self._initialize_data_manager()
            
            # 初始化账户管理器
            self._initialize_account_manager()
            
            # 初始化风险管理器
            self._initialize_risk_manager()
            
            # 初始化回测引擎
            self._initialize_backtest_engine()
            
            self.logger.info("✅ 系统初始化完成")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ 系统初始化失败: {e}")
            else:
                print(f"❌ 系统初始化失败: {e}")
            raise
    
    def _initialize_data_manager(self) -> None:
        """初始化数据管理器"""
        self.logger.info("🔧 初始化数据管理器...")
        
        self.data_manager = DataManager()
        
        # 添加Binance数据提供者
        binance_config = self.config.get_data_source_config('binance')
        if binance_config.get('enabled', True):
            binance_provider = BinanceDataProvider(
                api_key=binance_config.get('api_key'),
                api_secret=binance_config.get('api_secret')
            )
            self.data_manager.add_provider('binance', binance_provider, is_default=True)
            self.logger.info("✅ Binance数据提供者已添加")
    
    def _initialize_account_manager(self) -> None:
        """初始化账户管理器"""
        self.logger.info("🔧 初始化账户管理器...")
        
        account_config = self.config.get_account_config()
        account_type_str = account_config.get('account_type', 'SPOT')
        account_type = AccountType(account_type_str)
        
        self.account_manager = AccountManager(account_type)
        
        # 设置初始余额
        initial_balance = account_config.get('initial_balance', {'USDT': 10000.0})
        self.account_manager.set_initial_balance(initial_balance)
        
        self.logger.info(f"✅ 账户管理器已初始化 (类型: {account_type.value})")
    
    def _initialize_risk_manager(self) -> None:
        """初始化风险管理器"""
        self.logger.info("🔧 初始化风险管理器...")
        
        risk_config = self.config.get_risk_management_config()
        
        if risk_config.get('enabled', True):
            # 创建风险限制
            risk_limits = RiskLimits(
                max_position_percent=risk_config.get('position_limits', {}).get('max_position_percent', 0.1),
                max_total_position_percent=risk_config.get('position_limits', {}).get('max_total_position_percent', 0.8),
                max_daily_loss_percent=risk_config.get('loss_limits', {}).get('max_daily_loss_percent', 0.05),
                max_total_loss_percent=risk_config.get('loss_limits', {}).get('max_total_loss_percent', 0.20),
                max_drawdown_percent=risk_config.get('loss_limits', {}).get('max_drawdown_percent', 0.15),
                max_trades_per_hour=risk_config.get('frequency_limits', {}).get('max_trades_per_hour', 10),
                max_trades_per_day=risk_config.get('frequency_limits', {}).get('max_trades_per_day', 100),
            )
            
            self.risk_manager = RiskManager(risk_limits)
            self.logger.info("✅ 风险管理器已启用")
        else:
            self.logger.warning("⚠️ 风险管理器已禁用")
    
    def _initialize_backtest_engine(self) -> None:
        """初始化回测引擎"""
        self.logger.info("🔧 初始化回测引擎...")
        
        self.backtest_engine = BacktestEngine(self.data_manager)
        self.logger.info("✅ 回测引擎已初始化")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        signal_name = signal.Signals(signum).name
        self.logger.info(f"📶 接收到信号: {signal_name}")
        self.shutdown()
    
    def run_backtest(self, 
                    strategy_name: str,
                    symbol: str,
                    start_date: str,
                    end_date: str,
                    timeframe: str = '1h',
                    initial_balance: float = 10000.0) -> None:
        """
        运行回测
        
        Args:
            strategy_name: 策略名称
            symbol: 交易对
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            timeframe: 时间周期
            initial_balance: 初始资金
        """
        self.logger.info("🎯 开始运行回测")
        self.current_mode = "backtest"
        
        try:
            # 解析日期
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 创建回测配置
            backtest_config = BacktestConfig(
                start_date=start_dt,
                end_date=end_dt,
                initial_balance={'USDT': initial_balance},
                timeframe=timeframe,
                data_provider='binance',
                enable_risk_management=True,
                risk_limits=self.risk_manager.risk_limits if self.risk_manager else None
            )
            
            # 创建策略实例
            strategy_config = StrategyConfig(
                name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                parameters=self.config.get(f'strategies.{strategy_name}.parameters', {})
            )
            
            if strategy_name == 'mean_reversion':
                strategy = MeanReversionStrategy(strategy_config)
            else:
                raise ValueError(f"未知策略: {strategy_name}")
            
            # 添加策略到回测引擎
            self.backtest_engine.add_strategy(strategy_name, strategy, strategy_config)
            
            # 运行回测
            result = self.backtest_engine.run_backtest(backtest_config)
            
            # 显示结果
            self._display_backtest_results(result)
            
        except Exception as e:
            self.logger.error(f"❌ 回测失败: {e}")
            raise
        finally:
            self.current_mode = None
    
    def _display_backtest_results(self, result) -> None:
        """显示回测结果"""
        self.logger.info("📊 回测结果摘要:")
        self.logger.info("-" * 40)
        
        if result.performance_metrics:
            metrics = result.performance_metrics
            self.logger.info(f"总交易次数: {metrics.total_trades}")
            self.logger.info(f"盈利交易: {metrics.winning_trades}")
            self.logger.info(f"亏损交易: {metrics.losing_trades}")
            self.logger.info(f"胜率: {metrics.win_rate:.2%}")
            self.logger.info(f"总收益率: {metrics.total_return:.2%}")
            self.logger.info(f"最大回撤: {metrics.max_drawdown:.2%}")
            self.logger.info(f"夏普比率: {metrics.sharpe_ratio:.4f}")
        
        self.logger.info(f"状态: {result.status.value}")
        if result.error_message:
            self.logger.error(f"错误: {result.error_message}")
        
        # 显示图表路径
        if result.chart_paths:
            self.logger.info("📈 生成的图表:")
            for chart_name, path in result.chart_paths.items():
                self.logger.info(f"  - {chart_name}: {path}")
    
    def run_live_trading(self, 
                        strategy_name: str,
                        symbol: str,
                        timeframe: str = '1h',
                        dry_run: bool = True) -> None:
        """
        运行实盘交易
        
        Args:
            strategy_name: 策略名称
            symbol: 交易对
            timeframe: 时间周期
            dry_run: 是否为模拟模式
        """
        self.logger.info("🔴 开始实盘交易" if not dry_run else "🟡 开始模拟交易")
        self.current_mode = "live" if not dry_run else "simulation"
        self.is_running = True
        
        try:
            # 初始化经纪商
            if dry_run:
                self.broker = SimulatedBroker(
                    initial_balance=self.config.get('account.initial_balance', {'USDT': 10000.0}),
                    commission_rate=self.config.get('trading.default_commission_rate', 0.001)
                )
            else:
                binance_config = self.config.get_data_source_config('binance')
                self.broker = BinanceBroker(
                    api_key=binance_config.get('api_key'),
                    api_secret=binance_config.get('api_secret'),
                    testnet=binance_config.get('testnet', False)
                )
            
            # 创建策略实例
            strategy_config = StrategyConfig(
                name=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                parameters=self.config.get(f'strategies.{strategy_name}.parameters', {}),
                dry_run=dry_run
            )
            
            if strategy_name == 'mean_reversion':
                strategy = MeanReversionStrategy(strategy_config)
            else:
                raise ValueError(f"未知策略: {strategy_name}")
            
            # 初始化策略
            strategy.initialize()
            self.active_strategies[strategy_name] = strategy
            
            # 获取历史数据
            self.logger.info(f"📥 获取 {symbol} 历史数据...")
            historical_data = self.data_manager.get_historical_klines(
                symbol=symbol,
                interval=timeframe,
                limit=1000
            )
            
            if historical_data.empty:
                raise ValueError(f"无法获取 {symbol} 的历史数据")
            
            # 处理历史数据
            signals = strategy.on_data(historical_data)
            
            # 订阅实时数据
            self.logger.info(f"📡 订阅 {symbol} 实时数据...")
            
            def on_kline_data(kline_data):
                """处理实时K线数据"""
                try:
                    if not self.is_running:
                        return
                    
                    # 将K线数据转换为DataFrame
                    new_data = pd.DataFrame([kline_data.to_dict()])
                    
                    # 更新价格缓存
                    if self.risk_manager:
                        self.risk_manager.update_price_cache(symbol, kline_data.close)
                    self.account_manager.update_price_cache(symbol, kline_data.close)
                    
                    # 策略处理数据
                    signals = strategy.on_data(new_data)
                    
                    # 处理信号
                    for signal in signals:
                        self._process_signal(strategy_name, signal, kline_data.close)
                
                except Exception as e:
                    self.logger.error(f"处理实时数据失败: {e}")
            
            # 开始订阅
            self.data_manager.subscribe_kline(symbol, timeframe, on_kline_data)
            
            self.logger.info("✅ 交易系统运行中... (按 Ctrl+C 停止)")
            
            # 保持运行
            import time
            while self.is_running:
                time.sleep(1)
                
                # 每分钟记录一次状态
                if datetime.now().second == 0:
                    self._log_system_status()
        
        except Exception as e:
            self.logger.error(f"❌ 交易运行失败: {e}")
            raise
        finally:
            self.is_running = False
            self.current_mode = None
    
    def _process_signal(self, strategy_name: str, signal, current_price: float) -> None:
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
                    self.logger.warning(f"⚠️ 信号被风控拒绝: {risk_report.message}")
                    return
            
            # 下单
            order = self.broker.place_order(signal, quantity, signal.price)
            
            if order:
                self.logger.info(f"📋 订单已提交: {order.symbol} {order.side.value} {order.quantity}")
                
                # 更新账户状态
                self.account_manager.add_order(order)
                
                # 记录交易
                if self.risk_manager:
                    self.risk_manager.record_trade(
                        signal.symbol, order.side, order.filled_quantity, 
                        order.avg_price, strategy_name
                    )
        
        except Exception as e:
            self.logger.error(f"处理信号失败: {e}")
    
    def _calculate_quantity(self, signal, current_price: float) -> float:
        """计算交易数量"""
        if signal.quantity is not None:
            return signal.quantity
        
        if signal.quantity_percent is not None:
            account_summary = self.account_manager.get_account_summary()
            total_value = account_summary.get('total_value_usdt', 0)
            target_value = total_value * signal.quantity_percent
            return target_value / current_price
        
        return 0.001  # 默认最小数量
    
    def _log_system_status(self) -> None:
        """记录系统状态"""
        try:
            account_summary = self.account_manager.get_account_summary()
            self.logger.info(f"💰 账户价值: {account_summary.get('total_value_usdt', 0):.2f} USDT")
            self.logger.info(f"📈 活跃策略: {len(self.active_strategies)}")
            self.logger.info(f"📊 活跃仓位: {account_summary.get('active_positions', 0)}")
        except Exception as e:
            self.logger.error(f"记录系统状态失败: {e}")
    
    def list_strategies(self) -> None:
        """列出可用策略"""
        self.logger.info("📋 可用策略:")
        self.logger.info("- mean_reversion: 均值回归策略")
        # 可以在这里添加更多策略
    
    def show_status(self) -> None:
        """显示系统状态"""
        self.logger.info("📊 系统状态:")
        self.logger.info(f"运行状态: {'运行中' if self.is_running else '已停止'}")
        self.logger.info(f"当前模式: {self.current_mode or '无'}")
        self.logger.info(f"活跃策略: {len(self.active_strategies)}")
        
        if self.account_manager:
            account_summary = self.account_manager.get_account_summary()
            self.logger.info(f"账户价值: {account_summary.get('total_value_usdt', 0):.2f} USDT")
            self.logger.info(f"活跃仓位: {account_summary.get('active_positions', 0)}")
    
    def shutdown(self) -> None:
        """关闭系统"""
        self.logger.info("🛑 正在关闭系统...")
        
        self.is_running = False
        
        # 清理资源
        if self.data_manager:
            # 关闭数据流
            pass
        
        if self.broker:
            # 关闭连接
            pass
        
        self.logger.info("✅ 系统已安全关闭")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="AutoTrader 量化自动交易系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行回测
  python main.py backtest --strategy mean_reversion --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31
  
  # 运行模拟交易
  python main.py live --strategy mean_reversion --symbol BTCUSDT --dry-run
  
  # 运行实盘交易
  python main.py live --strategy mean_reversion --symbol BTCUSDT
  
  # 显示系统状态
  python main.py status
        """
    )
    
    # 添加子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 回测命令
    backtest_parser = subparsers.add_parser('backtest', help='运行回测')
    backtest_parser.add_argument('--strategy', required=True, help='策略名称')
    backtest_parser.add_argument('--symbol', required=True, help='交易对符号')
    backtest_parser.add_argument('--start', required=True, help='开始日期 (YYYY-MM-DD)')
    backtest_parser.add_argument('--end', required=True, help='结束日期 (YYYY-MM-DD)')
    backtest_parser.add_argument('--timeframe', default='1h', help='时间周期 (默认: 1h)')
    backtest_parser.add_argument('--balance', type=float, default=10000.0, help='初始资金 (默认: 10000)')
    
    # 实盘交易命令
    live_parser = subparsers.add_parser('live', help='运行实盘交易')
    live_parser.add_argument('--strategy', required=True, help='策略名称')
    live_parser.add_argument('--symbol', required=True, help='交易对符号')
    live_parser.add_argument('--timeframe', default='1h', help='时间周期 (默认: 1h)')
    live_parser.add_argument('--dry-run', action='store_true', help='模拟模式')
    
    # 其他命令
    subparsers.add_parser('strategies', help='列出可用策略')
    subparsers.add_parser('status', help='显示系统状态')
    
    # 全局参数
    parser.add_argument('--config', default='config.yml', help='配置文件路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    
    return parser


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    # 如果没有提供命令，显示帮助
    if not args.command:
        parser.print_help()
        return
    
    # 创建交易系统实例
    trading_system = TradingSystem()
    
    try:
        # 初始化系统
        trading_system.initialize(args.config)
        
        # 根据命令执行相应操作
        if args.command == 'backtest':
            trading_system.run_backtest(
                strategy_name=args.strategy,
                symbol=args.symbol,
                start_date=args.start,
                end_date=args.end,
                timeframe=args.timeframe,
                initial_balance=args.balance
            )
        
        elif args.command == 'live':
            trading_system.run_live_trading(
                strategy_name=args.strategy,
                symbol=args.symbol,
                timeframe=args.timeframe,
                dry_run=args.dry_run
            )
        
        elif args.command == 'strategies':
            trading_system.list_strategies()
        
        elif args.command == 'status':
            trading_system.show_status()
    
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断")
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    finally:
        trading_system.shutdown()


if __name__ == '__main__':
    # 设置事件循环策略（Windows兼容性）
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # 运行主函数
    main()