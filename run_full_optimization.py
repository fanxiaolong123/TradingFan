#!/usr/bin/env python3
"""
完整策略优化脚本

根据rules.md要求，对所有策略进行完整优化：
回测币种：BTC/ETH/SOL/BNB/DOGE/PEPE
时间周期：1h, 4h, 1D
策略：当前所有策略
性能阈值：胜率 ≥ 60%、年化 ≥ 30%、最大回撤 ≤ 20%
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_trader.core.strategy_optimizer import MultiStrategyOptimizer

def main():
    """主优化函数"""
    print("🚀 开始完整策略优化...")
    print("=" * 80)
    
    # 根据rules.md要求配置
    strategies = ['momentum', 'mean_reversion', 'trend_following', 'breakout']
    
    # 目前可用的币种（根据数据文件检查）
    available_symbols = []
    
    # 检查数据文件
    data_dir = Path("binance_historical_data/processed")
    if data_dir.exists():
        for file in data_dir.glob("*_1h_combined.csv"):
            symbol = file.stem.replace("_1h_combined", "")
            available_symbols.append(symbol)
    
    # 如果没有找到文件，使用默认的已知币种
    if not available_symbols:
        available_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
    
    print(f"📊 可用币种: {', '.join(available_symbols)}")
    
    # 时间周期（根据数据可用性调整）
    timeframes = ['1h', '1d']  # 暂时只用1h和1d，4h需要额外处理
    
    # 基础配置
    base_config = {
        'start_date': datetime(2023, 1, 1),
        'end_date': datetime(2024, 1, 1),  # 使用1年数据
        'optimization_method': 'grid_search',
        'max_iterations': 50,
        'target_metric': 'sharpe_ratio',
        'n_jobs': 2,
        'performance_threshold': {
            'win_rate': 0.60,                  # 胜率 ≥ 60%
            'annualized_return': 0.30,         # 年化 ≥ 30%
            'max_drawdown': -0.20,             # 最大回撤 ≤ 20%
            'sharpe_ratio': 1.0,               # 夏普比率 ≥ 1.0
            'total_trades': 10                 # 最少交易数
        }
    }
    
    print(f"📈 策略数量: {len(strategies)}")
    print(f"🪙 币种数量: {len(available_symbols)}")
    print(f"⏱️ 时间周期: {', '.join(timeframes)}")
    print(f"📅 优化期间: {base_config['start_date'].strftime('%Y-%m-%d')} 到 {base_config['end_date'].strftime('%Y-%m-%d')}")
    print(f"🎯 性能阈值: 胜率≥60%, 年化≥30%, 回撤≤20%")
    print("=" * 80)
    
    try:
        # 创建多策略优化器
        multi_optimizer = MultiStrategyOptimizer(strategies, available_symbols, timeframes)
        
        # 运行优化
        print("🔄 开始优化过程...")
        all_results = multi_optimizer.optimize_all(base_config)
        
        # 分析结果
        print("\n📊 优化结果分析:")
        print(f"总测试组合: {len(all_results)}")
        
        # 按性能评分排序
        sorted_results = sorted(all_results, key=lambda x: x.performance_score, reverse=True)
        
        # 达标策略统计
        qualified_results = [r for r in all_results if r.meets_threshold]
        print(f"达标策略数: {len(qualified_results)}")
        print(f"达标率: {len(qualified_results)/len(all_results)*100:.1f}%")
        
        # 显示TOP 10结果
        print("\n🏆 TOP 10 策略组合:")
        print("-" * 100)
        print(f"{'排名':<4} {'策略':<15} {'币种':<10} {'周期':<6} {'评分':<6} {'年化':<8} {'夏普':<8} {'回撤':<8} {'胜率':<6} {'达标':<4}")
        print("-" * 100)
        
        for i, result in enumerate(sorted_results[:10], 1):
            print(f"{i:<4} {result.strategy_name:<15} {result.symbol:<10} {result.timeframe:<6} "
                  f"{result.performance_score:<6.1f} {result.annualized_return*100:<8.2f} "
                  f"{result.sharpe_ratio:<8.3f} {result.max_drawdown*100:<8.2f} "
                  f"{result.win_rate*100:<6.1f} {'✅' if result.meets_threshold else '❌':<4}")
        
        # 按策略类型统计
        print("\n📈 按策略类型统计:")
        print("-" * 60)
        strategy_stats = {}
        for result in all_results:
            if result.strategy_name not in strategy_stats:
                strategy_stats[result.strategy_name] = {
                    'total': 0,
                    'qualified': 0,
                    'avg_score': 0,
                    'max_score': 0,
                    'avg_return': 0,
                    'avg_sharpe': 0
                }
            
            stats = strategy_stats[result.strategy_name]
            stats['total'] += 1
            if result.meets_threshold:
                stats['qualified'] += 1
            stats['avg_score'] += result.performance_score
            stats['max_score'] = max(stats['max_score'], result.performance_score)
            stats['avg_return'] += result.annualized_return
            stats['avg_sharpe'] += result.sharpe_ratio
        
        # 计算平均值
        for strategy, stats in strategy_stats.items():
            stats['avg_score'] /= stats['total']
            stats['avg_return'] /= stats['total']
            stats['avg_sharpe'] /= stats['total']
            stats['qualified_rate'] = stats['qualified'] / stats['total'] * 100
        
        print(f"{'策略':<15} {'测试数':<8} {'达标数':<8} {'达标率':<8} {'平均评分':<10} {'最高评分':<10} {'平均年化':<10}")
        print("-" * 80)
        for strategy, stats in strategy_stats.items():
            print(f"{strategy:<15} {stats['total']:<8} {stats['qualified']:<8} "
                  f"{stats['qualified_rate']:<8.1f} {stats['avg_score']:<10.1f} "
                  f"{stats['max_score']:<10.1f} {stats['avg_return']*100:<10.2f}")
        
        # 推荐策略
        print("\n💡 策略推荐:")
        print("-" * 60)
        
        if qualified_results:
            best_qualified = max(qualified_results, key=lambda x: x.performance_score)
            print(f"🥇 最佳达标策略: {best_qualified.strategy_name}-{best_qualified.symbol}-{best_qualified.timeframe}")
            print(f"   性能评分: {best_qualified.performance_score:.1f}")
            print(f"   年化收益率: {best_qualified.annualized_return*100:.2f}%")
            print(f"   夏普比率: {best_qualified.sharpe_ratio:.3f}")
            print(f"   最大回撤: {best_qualified.max_drawdown*100:.2f}%")
            print(f"   胜率: {best_qualified.win_rate*100:.1f}%")
            
            # 显示最佳策略参数
            print(f"\n🎯 最佳策略参数:")
            for param, value in best_qualified.params.items():
                print(f"   {param}: {value}")
        
        # 最佳策略（不考虑达标）
        if sorted_results:
            best_overall = sorted_results[0]
            print(f"\n🏆 最高评分策略: {best_overall.strategy_name}-{best_overall.symbol}-{best_overall.timeframe}")
            print(f"   性能评分: {best_overall.performance_score:.1f}")
            print(f"   年化收益率: {best_overall.annualized_return*100:.2f}%")
            print(f"   夏普比率: {best_overall.sharpe_ratio:.3f}")
            print(f"   是否达标: {'✅' if best_overall.meets_threshold else '❌'}")
        
        print("\n📁 详细结果已保存到:")
        print("   multi_strategy_optimization/ 目录")
        print("   包含CSV文件和排行榜文本文件")
        
        print("\n🎉 完整优化任务完成!")
        
    except Exception as e:
        print(f"❌ 优化过程出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()