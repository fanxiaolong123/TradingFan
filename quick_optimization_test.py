#!/usr/bin/env python3
"""
快速策略优化测试脚本

使用较小的数据集快速测试策略优化框架
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_trader.core.strategy_optimizer import StrategyOptimizer, MultiStrategyOptimizer, OptimizationConfig

def quick_single_strategy_test():
    """快速单策略测试"""
    print("🧪 快速单策略优化测试...")
    
    # 创建优化配置 - 使用较小的时间范围
    config = OptimizationConfig(
        strategy_name='momentum',
        symbol='BTCUSDT',
        timeframe='1h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 15),  # 只测试2周
        param_ranges={
            'short_ma_period': [12, 20],
            'long_ma_period': [50, 100],
            'position_size': [0.2, 0.3]
        },
        optimization_method='grid_search',
        max_iterations=8,
        target_metric='sharpe_ratio',
        n_jobs=1  # 单线程避免复杂度
    )
    
    try:
        # 创建优化器
        optimizer = StrategyOptimizer(config)
        
        # 运行优化
        result = optimizer.optimize()
        
        # 显示结果
        print(f"\n✅ 优化完成!")
        print(f"最优参数: {result.params}")
        print(f"夏普比率: {result.sharpe_ratio:.3f}")
        print(f"年化收益率: {result.annualized_return*100:.2f}%")
        print(f"最大回撤: {result.max_drawdown*100:.2f}%")
        print(f"胜率: {result.win_rate*100:.1f}%")
        print(f"性能评分: {result.performance_score:.1f}/100")
        print(f"是否达标: {'✅' if result.meets_threshold else '❌'}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_multi_strategy_test():
    """快速多策略测试"""
    print("\n🚀 快速多策略优化测试...")
    
    # 配置 - 使用更少的组合
    strategies = ['momentum']
    symbols = ['BTCUSDT']
    timeframes = ['1h']
    
    # 基础配置
    base_config = {
        'start_date': datetime(2023, 1, 1),
        'end_date': datetime(2023, 1, 8),  # 只测试1周
        'optimization_method': 'grid_search',
        'max_iterations': 4,
        'target_metric': 'sharpe_ratio',
        'n_jobs': 1
    }
    
    try:
        # 创建多策略优化器
        multi_optimizer = MultiStrategyOptimizer(strategies, symbols, timeframes)
        
        # 运行优化
        all_results = multi_optimizer.optimize_all(base_config)
        
        # 显示结果
        print(f"\n✅ 多策略优化完成!")
        print(f"总组合数: {len(all_results)}")
        
        if all_results:
            best_result = max(all_results, key=lambda x: x.performance_score)
            print(f"最佳策略: {best_result.strategy_name}-{best_result.symbol}-{best_result.timeframe}")
            print(f"性能评分: {best_result.performance_score:.1f}")
            print(f"夏普比率: {best_result.sharpe_ratio:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 快速策略优化测试开始...")
    print("=" * 50)
    
    # 1. 快速单策略测试
    if not quick_single_strategy_test():
        print("❌ 单策略测试失败")
        return
    
    # 2. 快速多策略测试
    if not quick_multi_strategy_test():
        print("❌ 多策略测试失败")
        return
    
    print("\n🎉 所有快速测试完成！")
    print("=" * 50)
    
    print("\n📁 结果文件位置:")
    print("   - 单策略结果: optimization_results/")
    print("   - 多策略结果: multi_strategy_optimization/")
    
    print("\n💡 可以运行完整测试:")
    print("   python3 test_strategy_optimization.py")

if __name__ == "__main__":
    main()