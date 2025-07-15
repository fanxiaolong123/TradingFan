#!/usr/bin/env python3
"""
使用真实历史数据运行回测
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_historical_backtest import RealHistoricalBacktester, PerformanceMetrics
from auto_trader.strategies.base import StrategyConfig

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def run_backtest_for_config(symbol: str, timeframe: str, strategy_type: str):
    """运行单个配置的回测"""
    
    print(f"\n{'='*80}")
    print(f"🎯 回测配置: {symbol} {timeframe} {strategy_type}")
    print(f"{'='*80}")
    
    # 创建回测引擎
    engine = RealHistoricalBacktester()
    
    # 根据策略类型设置参数
    if strategy_type == "momentum":
        parameters = {
            'short_ma_period': 24 if timeframe == '1h' else 12,
            'long_ma_period': 72 if timeframe == '1h' else 48,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'momentum_period': 48 if timeframe == '1h' else 24,
            'momentum_threshold': 0.02,
            'volume_threshold': 1.5,
            'position_size': 0.3,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        }
    else:  # mean_reversion
        parameters = {
            'ma_period': 48 if timeframe == '1h' else 24,
            'deviation_threshold': 0.02,
            'min_volume': 10,
            'position_size': 0.25,
            'stop_loss_pct': 0.025,
            'take_profit_pct': 0.05
        }
    
    # 创建策略配置
    strategy_config = StrategyConfig(
        name=f"{strategy_type}_{symbol}_{timeframe}",
        symbol=symbol,
        timeframe=timeframe,
        parameters=parameters
    )
    
    # 设置回测时间范围（使用下载的数据范围）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=89)  # 约3个月
    
    try:
        # 运行回测
        metrics = engine.run_backtest(strategy_config, start_date, end_date)
        
        # 生成报告
        report = engine.generate_report(metrics, f"{symbol} {timeframe} {strategy_type}")
        print(report)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy_type': strategy_type,
            'metrics': metrics,
            'trades': engine.trades
        }
        
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        return None

def create_visualizations(results):
    """创建可视化图表"""
    
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    # 1. 收益率对比柱状图
    plt.figure(figsize=(12, 6))
    
    strategies = []
    returns = []
    colors = []
    
    for result in results:
        if result and result['metrics'].total_return != 0:
            label = f"{result['symbol'].replace('USDT','')} {result['timeframe']} {result['strategy_type']}"
            strategies.append(label)
            returns.append(result['metrics'].total_return * 100)
            colors.append('green' if result['metrics'].total_return > 0 else 'red')
    
    bars = plt.bar(strategies, returns, color=colors, alpha=0.7)
    plt.xlabel('策略', fontsize=12)
    plt.ylabel('总收益率 (%)', fontsize=12)
    plt.title('策略收益率对比', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
    
    plt.tight_layout()
    plt.savefig(output_dir / "returns_comparison.png", dpi=300)
    plt.close()
    
    # 2. 风险指标对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 夏普比率
    ax = axes[0, 0]
    sharpe_data = [(r['symbol'].replace('USDT',''), r['metrics'].sharpe_ratio) 
                   for r in results if r]
    if sharpe_data:
        symbols, sharpes = zip(*sharpe_data)
        ax.bar(symbols, sharpes, color='skyblue')
        ax.set_title('夏普比率对比')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
    
    # 最大回撤
    ax = axes[0, 1]
    dd_data = [(r['symbol'].replace('USDT',''), abs(r['metrics'].max_drawdown * 100)) 
               for r in results if r]
    if dd_data:
        symbols, dds = zip(*dd_data)
        ax.bar(symbols, dds, color='coral')
        ax.set_title('最大回撤对比')
        ax.set_ylabel('最大回撤 (%)')
        ax.grid(True, alpha=0.3)
    
    # 胜率
    ax = axes[1, 0]
    wr_data = [(r['symbol'].replace('USDT',''), r['metrics'].win_rate * 100) 
               for r in results if r and r['metrics'].win_rate > 0]
    if wr_data:
        symbols, wrs = zip(*wr_data)
        ax.bar(symbols, wrs, color='lightgreen')
        ax.set_title('胜率对比')
        ax.set_ylabel('胜率 (%)')
        ax.grid(True, alpha=0.3)
    
    # 交易次数
    ax = axes[1, 1]
    trade_data = [(r['symbol'].replace('USDT',''), r['metrics'].total_trades) 
                  for r in results if r]
    if trade_data:
        symbols, trades = zip(*trade_data)
        ax.bar(symbols, trades, color='gold')
        ax.set_title('交易次数对比')
        ax.set_ylabel('交易次数')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "risk_metrics.png", dpi=300)
    plt.close()
    
    # 3. 资金曲线
    plt.figure(figsize=(14, 8))
    
    for result in results:
        if result and len(result['metrics'].equity_curve) > 0:
            equity_curve = np.array(result['metrics'].equity_curve)
            normalized_curve = equity_curve / equity_curve[0] * 100
            
            label = f"{result['symbol'].replace('USDT','')} {result['timeframe']} {result['strategy_type']}"
            plt.plot(normalized_curve, label=label, linewidth=2)
    
    plt.xlabel('时间', fontsize=12)
    plt.ylabel('资金曲线 (初始=100)', fontsize=12)
    plt.title('策略资金曲线对比', fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 可视化图表已保存到: {output_dir}")

def generate_final_report(results):
    """生成最终报告"""
    
    print("\n" + "="*80)
    print("📊 综合回测报告")
    print("="*80)
    
    # 找出最佳策略
    valid_results = [r for r in results if r and r['metrics'].total_return != 0]
    
    if not valid_results:
        print("❌ 没有有效的回测结果")
        return
    
    # 按收益率排序
    sorted_results = sorted(valid_results, 
                           key=lambda x: x['metrics'].total_return, 
                           reverse=True)
    
    print("\n🏆 策略排名（按总收益率）:")
    print(f"{'排名':<6} {'策略':<30} {'总收益率':<12} {'年化收益':<12} {'夏普比率':<10} {'最大回撤':<10}")
    print("-"*80)
    
    for i, result in enumerate(sorted_results[:5]):
        strategy_name = f"{result['symbol']} {result['timeframe']} {result['strategy_type']}"
        print(f"{i+1:<6} {strategy_name:<30} "
              f"{result['metrics'].total_return*100:>10.2f}% "
              f"{result['metrics'].annualized_return*100:>10.2f}% "
              f"{result['metrics'].sharpe_ratio:>8.3f} "
              f"{result['metrics'].max_drawdown*100:>8.2f}%")
    
    # 统计分析
    print("\n📈 统计分析:")
    
    # 按币种分析
    btc_results = [r for r in valid_results if 'BTC' in r['symbol']]
    eth_results = [r for r in valid_results if 'ETH' in r['symbol']]
    
    if btc_results:
        avg_btc_return = np.mean([r['metrics'].total_return for r in btc_results])
        print(f"   • BTC策略平均收益率: {avg_btc_return*100:.2f}%")
    
    if eth_results:
        avg_eth_return = np.mean([r['metrics'].total_return for r in eth_results])
        print(f"   • ETH策略平均收益率: {avg_eth_return*100:.2f}%")
    
    # 按策略类型分析
    momentum_results = [r for r in valid_results if r['strategy_type'] == 'momentum']
    mean_rev_results = [r for r in valid_results if r['strategy_type'] == 'mean_reversion']
    
    if momentum_results:
        avg_momentum_return = np.mean([r['metrics'].total_return for r in momentum_results])
        print(f"   • 动量策略平均收益率: {avg_momentum_return*100:.2f}%")
    
    if mean_rev_results:
        avg_mean_rev_return = np.mean([r['metrics'].total_return for r in mean_rev_results])
        print(f"   • 均值回归策略平均收益率: {avg_mean_rev_return*100:.2f}%")
    
    # 风险分析
    all_drawdowns = [abs(r['metrics'].max_drawdown) for r in valid_results]
    avg_drawdown = np.mean(all_drawdowns)
    
    print(f"\n⚠️ 风险指标:")
    print(f"   • 平均最大回撤: {avg_drawdown*100:.2f}%")
    print(f"   • 最大单一回撤: {max(all_drawdowns)*100:.2f}%")
    
    # 投资建议
    print(f"\n💡 投资建议:")
    
    best_strategy = sorted_results[0]
    print(f"   • 最佳策略: {best_strategy['symbol']} {best_strategy['timeframe']} {best_strategy['strategy_type']}")
    print(f"   • 建议仓位: 根据风险承受能力，建议分配20-30%资金")
    
    if avg_drawdown > 0.15:
        print(f"   • 风险提示: 平均回撤较大，建议设置严格的止损")
    
    # 保存报告
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    
    report_data = {
        'summary': {
            'total_strategies': len(valid_results),
            'avg_return': np.mean([r['metrics'].total_return for r in valid_results]),
            'best_return': sorted_results[0]['metrics'].total_return,
            'avg_drawdown': avg_drawdown
        },
        'rankings': [
            {
                'rank': i+1,
                'symbol': r['symbol'],
                'timeframe': r['timeframe'],
                'strategy_type': r['strategy_type'],
                'total_return': r['metrics'].total_return,
                'sharpe_ratio': r['metrics'].sharpe_ratio,
                'max_drawdown': r['metrics'].max_drawdown
            }
            for i, r in enumerate(sorted_results)
        ]
    }
    
    with open(output_dir / "backtest_report.json", 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 详细报告已保存到: {output_dir}/backtest_report.json")

def main():
    """主函数"""
    print("🚀 真实历史数据回测系统")
    print("=" * 80)
    
    # 检查数据文件
    data_dir = Path("binance_historical_data/processed")
    if not data_dir.exists():
        print("❌ 数据目录不存在，请先下载历史数据")
        return
    
    # 获取可用的数据文件
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("❌ 没有找到历史数据文件")
        return
    
    print(f"📊 找到 {len(csv_files)} 个数据文件")
    
    # 定义回测配置
    configs = []
    
    # 检查哪些数据文件可用
    for symbol in ['BTCUSDT', 'ETHUSDT']:
        for timeframe in ['1h', '4h']:
            filename = f"{symbol}_{timeframe}_combined.csv"
            if (data_dir / filename).exists():
                configs.append((symbol, timeframe, 'momentum'))
                configs.append((symbol, timeframe, 'mean_reversion'))
    
    print(f"📈 将运行 {len(configs)} 个策略回测")
    
    # 运行所有回测
    results = []
    for symbol, timeframe, strategy_type in configs:
        result = run_backtest_for_config(symbol, timeframe, strategy_type)
        if result:
            results.append(result)
    
    # 生成可视化
    if results:
        create_visualizations(results)
        
        # 生成最终报告
        generate_final_report(results)
    
    print("\n✅ 回测完成！")

if __name__ == "__main__":
    main()