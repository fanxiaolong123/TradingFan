#!/usr/bin/env python3
"""
多币种多时间框架回测引擎
支持BTC、ETH、BNB、SOL、DOGE、PEPE等币种的批量回测
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_historical_backtest import RealHistoricalBacktester, PerformanceMetrics
from auto_trader.strategies.base import StrategyConfig

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """回测配置"""
    symbol: str
    timeframe: str
    strategy_name: str
    parameters: Dict[str, Any]

@dataclass
class BacktestResult:
    """回测结果"""
    config: BacktestConfig
    metrics: PerformanceMetrics
    equity_curve: List[float]
    trades: List[Any]

class MultiCryptoBacktester:
    """多币种批量回测器"""
    
    def __init__(self, data_dir: str = "binance_historical_data/processed"):
        self.data_dir = Path(data_dir)
        self.results: List[BacktestResult] = []
        
    def get_strategy_configs(self) -> List[BacktestConfig]:
        """获取所有策略配置"""
        configs = []
        
        # 定义币种列表
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'DOGEUSDT', 'PEPEUSDT']
        
        # 定义时间框架
        timeframes = ['15m', '1h', '4h', '1d']
        
        # 为每个币种和时间框架创建策略配置
        for symbol in symbols:
            for timeframe in timeframes:
                # 动量策略配置
                momentum_config = BacktestConfig(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=f"{symbol}_{timeframe}_momentum",
                    parameters={
                        'short_ma_period': self._get_ma_period(timeframe, 'short'),
                        'long_ma_period': self._get_ma_period(timeframe, 'long'),
                        'rsi_period': 14,
                        'rsi_overbought': 70,
                        'rsi_oversold': 30,
                        'momentum_period': self._get_momentum_period(timeframe),
                        'momentum_threshold': 0.02,
                        'volume_threshold': 1.5,
                        'position_size': 0.3,
                        'stop_loss_pct': self._get_stop_loss(symbol),
                        'take_profit_pct': self._get_take_profit(symbol)
                    }
                )
                configs.append(momentum_config)
                
                # 均值回归策略配置
                mean_reversion_config = BacktestConfig(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy_name=f"{symbol}_{timeframe}_mean_reversion",
                    parameters={
                        'ma_period': self._get_ma_period(timeframe, 'medium'),
                        'deviation_threshold': self._get_deviation_threshold(symbol),
                        'min_volume': 10,
                        'position_size': 0.25,
                        'stop_loss_pct': self._get_stop_loss(symbol) * 0.8,
                        'take_profit_pct': self._get_take_profit(symbol) * 0.8
                    }
                )
                configs.append(mean_reversion_config)
        
        return configs
    
    def _get_ma_period(self, timeframe: str, type: str) -> int:
        """根据时间框架获取均线周期"""
        periods = {
            '15m': {'short': 20, 'medium': 50, 'long': 100},
            '1h': {'short': 24, 'medium': 48, 'long': 72},
            '4h': {'short': 12, 'medium': 24, 'long': 48},
            '1d': {'short': 7, 'medium': 14, 'long': 30}
        }
        return periods.get(timeframe, {'short': 20, 'medium': 50, 'long': 100})[type]
    
    def _get_momentum_period(self, timeframe: str) -> int:
        """根据时间框架获取动量周期"""
        periods = {
            '15m': 30,
            '1h': 48,
            '4h': 24,
            '1d': 14
        }
        return periods.get(timeframe, 30)
    
    def _get_stop_loss(self, symbol: str) -> float:
        """根据币种获取止损百分比"""
        # 高波动币种使用更大的止损
        stop_losses = {
            'BTCUSDT': 0.03,
            'ETHUSDT': 0.035,
            'BNBUSDT': 0.04,
            'SOLUSDT': 0.05,
            'DOGEUSDT': 0.06,
            'PEPEUSDT': 0.08
        }
        return stop_losses.get(symbol, 0.05)
    
    def _get_take_profit(self, symbol: str) -> float:
        """根据币种获取止盈百分比"""
        # 高波动币种使用更大的止盈
        take_profits = {
            'BTCUSDT': 0.06,
            'ETHUSDT': 0.07,
            'BNBUSDT': 0.08,
            'SOLUSDT': 0.10,
            'DOGEUSDT': 0.12,
            'PEPEUSDT': 0.15
        }
        return take_profits.get(symbol, 0.10)
    
    def _get_deviation_threshold(self, symbol: str) -> float:
        """根据币种获取偏离阈值"""
        thresholds = {
            'BTCUSDT': 0.02,
            'ETHUSDT': 0.025,
            'BNBUSDT': 0.03,
            'SOLUSDT': 0.04,
            'DOGEUSDT': 0.05,
            'PEPEUSDT': 0.06
        }
        return thresholds.get(symbol, 0.03)
    
    def run_single_backtest(self, config: BacktestConfig, 
                           start_date: datetime, end_date: datetime) -> Optional[BacktestResult]:
        """运行单个回测"""
        try:
            logger.info(f"🚀 开始回测: {config.strategy_name}")
            
            # 创建回测引擎
            engine = RealHistoricalBacktester()
            
            # 创建策略配置
            strategy_type = "momentum" if "momentum" in config.strategy_name else "mean_reversion"
            strategy_config = StrategyConfig(
                name=f"{strategy_type}_{config.symbol}_{config.timeframe}",
                symbol=config.symbol,
                timeframe=config.timeframe,
                parameters=config.parameters
            )
            
            # 运行回测
            metrics = engine.run_backtest(strategy_config, start_date, end_date)
            
            # 创建结果
            result = BacktestResult(
                config=config,
                metrics=metrics,
                equity_curve=metrics.equity_curve,
                trades=engine.trades
            )
            
            logger.info(f"✅ 完成回测: {config.strategy_name} - 收益率: {metrics.total_return*100:.2f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 回测失败 {config.strategy_name}: {str(e)}")
            return None
    
    def run_parallel_backtests(self, configs: List[BacktestConfig], 
                              start_date: datetime, end_date: datetime,
                              max_workers: int = 4) -> List[BacktestResult]:
        """并行运行多个回测"""
        results = []
        
        logger.info(f"🔄 开始并行回测 {len(configs)} 个策略配置")
        
        # 使用进程池并行执行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_config = {
                executor.submit(self.run_single_backtest, config, start_date, end_date): config 
                for config in configs
            }
            
            # 收集结果
            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"❌ 处理结果失败 {config.strategy_name}: {str(e)}")
        
        logger.info(f"✅ 完成 {len(results)} 个策略回测")
        
        return results
    
    def generate_comprehensive_report(self, results: List[BacktestResult]) -> Dict[str, Any]:
        """生成综合报告"""
        report = {
            'summary': {},
            'by_symbol': {},
            'by_timeframe': {},
            'by_strategy': {},
            'top_performers': [],
            'recommendations': []
        }
        
        # 按币种分组
        symbol_groups = {}
        for result in results:
            symbol = result.config.symbol
            if symbol not in symbol_groups:
                symbol_groups[symbol] = []
            symbol_groups[symbol].append(result)
        
        # 分析每个币种
        for symbol, group_results in symbol_groups.items():
            best_result = max(group_results, key=lambda x: x.metrics.annualized_return)
            
            report['by_symbol'][symbol] = {
                'best_strategy': best_result.config.strategy_name,
                'best_return': best_result.metrics.total_return,
                'best_sharpe': best_result.metrics.sharpe_ratio,
                'best_drawdown': best_result.metrics.max_drawdown,
                'total_strategies': len(group_results),
                'avg_return': np.mean([r.metrics.total_return for r in group_results])
            }
        
        # 按时间框架分组
        timeframe_groups = {}
        for result in results:
            timeframe = result.config.timeframe
            if timeframe not in timeframe_groups:
                timeframe_groups[timeframe] = []
            timeframe_groups[timeframe].append(result)
        
        # 分析每个时间框架
        for timeframe, group_results in timeframe_groups.items():
            report['by_timeframe'][timeframe] = {
                'avg_return': np.mean([r.metrics.total_return for r in group_results]),
                'avg_sharpe': np.mean([r.metrics.sharpe_ratio for r in group_results]),
                'avg_drawdown': np.mean([r.metrics.max_drawdown for r in group_results]),
                'total_strategies': len(group_results)
            }
        
        # 找出表现最好的策略
        sorted_results = sorted(results, key=lambda x: x.metrics.annualized_return, reverse=True)
        
        for i, result in enumerate(sorted_results[:10]):
            report['top_performers'].append({
                'rank': i + 1,
                'strategy': result.config.strategy_name,
                'symbol': result.config.symbol,
                'timeframe': result.config.timeframe,
                'total_return': result.metrics.total_return,
                'annualized_return': result.metrics.annualized_return,
                'sharpe_ratio': result.metrics.sharpe_ratio,
                'max_drawdown': result.metrics.max_drawdown,
                'win_rate': result.metrics.win_rate
            })
        
        # 生成推荐
        self._generate_recommendations(report, results)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any], results: List[BacktestResult]):
        """生成投资建议"""
        recommendations = []
        
        # 推荐最佳币种
        best_symbol = max(report['by_symbol'].items(), 
                         key=lambda x: x[1]['best_return'])[0]
        recommendations.append(f"最佳交易币种: {best_symbol}")
        
        # 推荐最佳时间框架
        best_timeframe = max(report['by_timeframe'].items(), 
                            key=lambda x: x[1]['avg_return'])[0]
        recommendations.append(f"最佳时间框架: {best_timeframe}")
        
        # 风险提示
        high_risk_symbols = [symbol for symbol, data in report['by_symbol'].items() 
                            if abs(data['best_drawdown']) > 0.2]
        if high_risk_symbols:
            recommendations.append(f"高风险币种 (回撤>20%): {', '.join(high_risk_symbols)}")
        
        # 稳健选择
        stable_results = [r for r in results if r.metrics.sharpe_ratio > 1.0 
                         and abs(r.metrics.max_drawdown) < 0.15]
        if stable_results:
            best_stable = max(stable_results, key=lambda x: x.metrics.annualized_return)
            recommendations.append(f"稳健策略推荐: {best_stable.config.strategy_name}")
        
        report['recommendations'] = recommendations
    
    def visualize_results(self, results: List[BacktestResult], output_dir: str = "backtest_results"):
        """可视化回测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 收益率对比图
        self._plot_returns_comparison(results, output_dir)
        
        # 2. 风险收益散点图
        self._plot_risk_return_scatter(results, output_dir)
        
        # 3. 资金曲线对比
        self._plot_equity_curves(results, output_dir)
        
        # 4. 热力图
        self._plot_heatmap(results, output_dir)
        
        logger.info(f"📊 可视化结果已保存到: {output_dir}")
    
    def _plot_returns_comparison(self, results: List[BacktestResult], output_dir: str):
        """绘制收益率对比图"""
        plt.figure(figsize=(15, 8))
        
        # 按币种分组
        symbol_returns = {}
        for result in results:
            symbol = result.config.symbol.replace('USDT', '')
            if symbol not in symbol_returns:
                symbol_returns[symbol] = []
            symbol_returns[symbol].append(result.metrics.total_return * 100)
        
        # 绘制箱线图
        data = []
        labels = []
        for symbol, returns in symbol_returns.items():
            data.append(returns)
            labels.append(symbol)
        
        plt.boxplot(data, labels=labels)
        plt.title('各币种收益率分布', fontsize=16)
        plt.xlabel('币种', fontsize=12)
        plt.ylabel('总收益率 (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/returns_comparison.png", dpi=300)
        plt.close()
    
    def _plot_risk_return_scatter(self, results: List[BacktestResult], output_dir: str):
        """绘制风险收益散点图"""
        plt.figure(figsize=(12, 8))
        
        # 按币种分组绘制
        symbols = set([r.config.symbol for r in results])
        colors = plt.cm.rainbow(np.linspace(0, 1, len(symbols)))
        
        for i, symbol in enumerate(symbols):
            symbol_results = [r for r in results if r.config.symbol == symbol]
            
            returns = [r.metrics.annualized_return * 100 for r in symbol_results]
            risks = [r.metrics.volatility * 100 for r in symbol_results]
            
            plt.scatter(risks, returns, color=colors[i], 
                       label=symbol.replace('USDT', ''), s=100, alpha=0.6)
        
        plt.xlabel('年化波动率 (%)', fontsize=12)
        plt.ylabel('年化收益率 (%)', fontsize=12)
        plt.title('风险收益分布图', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 添加夏普比率等值线
        x_range = np.linspace(0, max([r.metrics.volatility * 100 for r in results]), 100)
        for sharpe in [0.5, 1.0, 1.5]:
            y_range = sharpe * x_range
            plt.plot(x_range, y_range, '--', alpha=0.3, label=f'Sharpe={sharpe}')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/risk_return_scatter.png", dpi=300)
        plt.close()
    
    def _plot_equity_curves(self, results: List[BacktestResult], output_dir: str):
        """绘制资金曲线对比"""
        # 选择表现最好的几个策略
        top_results = sorted(results, key=lambda x: x.metrics.annualized_return, reverse=True)[:6]
        
        plt.figure(figsize=(14, 8))
        
        for result in top_results:
            if result.metrics.equity_curve:
                # 归一化资金曲线
                equity_curve = np.array(result.metrics.equity_curve)
                normalized_curve = equity_curve / equity_curve[0] * 100
                
                label = f"{result.config.symbol.replace('USDT', '')} {result.config.timeframe}"
                plt.plot(normalized_curve, label=label, linewidth=2)
        
        plt.xlabel('时间', fontsize=12)
        plt.ylabel('资金曲线 (初始=100)', fontsize=12)
        plt.title('Top 6 策略资金曲线对比', fontsize=16)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/equity_curves.png", dpi=300)
        plt.close()
    
    def _plot_heatmap(self, results: List[BacktestResult], output_dir: str):
        """绘制收益率热力图"""
        # 创建数据透视表
        data_dict = {}
        
        for result in results:
            symbol = result.config.symbol.replace('USDT', '')
            timeframe = result.config.timeframe
            strategy = 'Momentum' if 'momentum' in result.config.strategy_name else 'MeanRev'
            
            key = f"{symbol}_{strategy}"
            if key not in data_dict:
                data_dict[key] = {}
            
            data_dict[key][timeframe] = result.metrics.total_return * 100
        
        # 转换为DataFrame
        df = pd.DataFrame(data_dict).T
        df = df[['15m', '1h', '4h', '1d']]  # 按时间框架排序
        
        # 绘制热力图
        plt.figure(figsize=(10, 12))
        sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': '总收益率 (%)'})
        
        plt.title('策略收益率热力图', fontsize=16)
        plt.xlabel('时间框架', fontsize=12)
        plt.ylabel('币种_策略', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/returns_heatmap.png", dpi=300)
        plt.close()
    
    def save_results(self, results: List[BacktestResult], report: Dict[str, Any], 
                    output_dir: str = "backtest_results"):
        """保存回测结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存详细结果
        results_data = []
        for result in results:
            results_data.append({
                'strategy': result.config.strategy_name,
                'symbol': result.config.symbol,
                'timeframe': result.config.timeframe,
                'total_return': result.metrics.total_return,
                'annualized_return': result.metrics.annualized_return,
                'sharpe_ratio': result.metrics.sharpe_ratio,
                'max_drawdown': result.metrics.max_drawdown,
                'win_rate': result.metrics.win_rate,
                'total_trades': result.metrics.total_trades,
                'parameters': result.config.parameters
            })
        
        # 保存为JSON
        with open(f"{output_dir}/backtest_results.json", 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=2)
        
        # 保存报告
        with open(f"{output_dir}/backtest_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 保存CSV格式
        df = pd.DataFrame(results_data)
        df.to_csv(f"{output_dir}/backtest_results.csv", index=False)
        
        logger.info(f"💾 结果已保存到: {output_dir}")

def main():
    """主函数"""
    print("🚀 多币种多时间框架批量回测系统")
    print("=" * 80)
    
    # 创建回测器
    backtest_system = MultiCryptoBacktester()
    
    # 设置回测时间范围
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2025, 7, 15)
    
    logger.info(f"📅 回测期间: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"⏱️ 时间跨度: {(end_date - start_date).days} 天")
    
    # 获取所有策略配置
    configs = backtest_system.get_strategy_configs()
    logger.info(f"📊 总策略数: {len(configs)}")
    
    # 运行批量回测
    results = backtest_system.run_parallel_backtests(configs, start_date, end_date, max_workers=4)
    
    if results:
        # 生成综合报告
        report = backtest_system.generate_comprehensive_report(results)
        
        # 打印摘要
        print("\n" + "="*80)
        print("📊 回测结果摘要")
        print("="*80)
        
        print("\n🏆 TOP 10 最佳策略:")
        print(f"{'排名':<6} {'策略名称':<40} {'币种':<10} {'时间框架':<10} {'总收益率':<12} {'年化收益':<12} {'夏普比率':<10}")
        print("-"*110)
        
        for performer in report['top_performers']:
            print(f"{performer['rank']:<6} {performer['strategy']:<40} "
                  f"{performer['symbol'].replace('USDT',''):<10} {performer['timeframe']:<10} "
                  f"{performer['total_return']*100:>10.2f}% {performer['annualized_return']*100:>10.2f}% "
                  f"{performer['sharpe_ratio']:>8.3f}")
        
        print("\n💡 投资建议:")
        for rec in report['recommendations']:
            print(f"   • {rec}")
        
        # 可视化结果
        backtest_system.visualize_results(results)
        
        # 保存结果
        backtest_system.save_results(results, report)
        
        print("\n✅ 批量回测完成！")
        print(f"📊 已处理 {len(results)} 个策略")
        print(f"📁 结果保存在: backtest_results/")
    
    else:
        print("\n❌ 回测失败，未获得有效结果")

if __name__ == "__main__":
    main()