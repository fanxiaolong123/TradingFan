#!/usr/bin/env python3
"""
策略优化测试脚本

测试策略优化框架的功能，包括：
1. 单策略参数优化
2. 多策略批量优化
3. 性能评估和排行榜生成
"""

import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_trader.core.strategy_optimizer import StrategyOptimizer, MultiStrategyOptimizer, OptimizationConfig

def test_single_strategy_optimization():
    """测试单策略优化"""
    print("🧪 测试单策略优化...")
    
    # 创建优化配置
    config = OptimizationConfig(
        strategy_name='momentum',
        symbol='BTCUSDT',
        timeframe='1h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 6, 1),
        param_ranges={
            'short_ma_period': [12, 20, 30],
            'long_ma_period': [50, 100],
            'position_size': [0.2, 0.3]
        },
        optimization_method='grid_search',
        max_iterations=50,
        target_metric='sharpe_ratio',
        n_jobs=2
    )
    
    try:
        # 创建优化器
        optimizer = StrategyOptimizer(config)
        
        # 运行优化
        result = optimizer.optimize()
        
        # 显示结果
        print("\n📊 单策略优化结果:")
        print(f"最优参数: {result.params}")
        print(f"夏普比率: {result.sharpe_ratio:.3f}")
        print(f"年化收益率: {result.annualized_return*100:.2f}%")
        print(f"最大回撤: {result.max_drawdown*100:.2f}%")
        print(f"胜率: {result.win_rate*100:.1f}%")
        print(f"性能评分: {result.performance_score:.1f}/100")
        print(f"是否达标: {'✅' if result.meets_threshold else '❌'}")
        
        # 生成报告
        report = optimizer.generate_optimization_report(result)
        print("\n📝 优化报告:")
        print(report)
        
        return True
        
    except Exception as e:
        print(f"❌ 单策略优化失败: {e}")
        return False

def test_multi_strategy_optimization():
    """测试多策略优化"""
    print("\n🚀 测试多策略优化...")
    
    # 策略配置
    strategies = ['momentum', 'mean_reversion']
    symbols = ['BTCUSDT', 'ETHUSDT']
    timeframes = ['1h', '1d']
    
    # 基础配置
    base_config = {
        'start_date': datetime(2023, 1, 1),
        'end_date': datetime(2023, 3, 1),
        'optimization_method': 'grid_search',
        'max_iterations': 20,
        'target_metric': 'sharpe_ratio',
        'n_jobs': 2
    }
    
    try:
        # 创建多策略优化器
        multi_optimizer = MultiStrategyOptimizer(strategies, symbols, timeframes)
        
        # 运行优化
        all_results = multi_optimizer.optimize_all(base_config)
        
        # 显示结果汇总
        print(f"\n📊 多策略优化完成:")
        print(f"总组合数: {len(all_results)}")
        
        # 按性能评分排序
        sorted_results = sorted(all_results, key=lambda x: x.performance_score, reverse=True)
        
        print("\n🏆 TOP 5 策略组合:")
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"{i}. {result.strategy_name}-{result.symbol}-{result.timeframe}: "
                  f"评分 {result.performance_score:.1f}, "
                  f"年化收益率 {result.annualized_return*100:.2f}%, "
                  f"夏普比率 {result.sharpe_ratio:.3f}")
        
        # 达标策略统计
        qualified_strategies = [r for r in all_results if r.meets_threshold]
        print(f"\n✅ 达标策略数: {len(qualified_strategies)}/{len(all_results)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 多策略优化失败: {e}")
        return False

def check_data_availability():
    """检查数据可用性"""
    print("\n🔍 检查数据可用性...")
    
    # 检查所需的数据文件
    required_files = [
        'binance_historical_data/processed/BTCUSDT_1h_combined.csv',
        'binance_historical_data/processed/BTCUSDT_1d_combined.csv',
        'binance_historical_data/processed/ETHUSDT_1h_combined.csv'
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            print(f"✅ {file_path}: {file_size:,} bytes")
    
    if missing_files:
        print(f"\n❌ 缺少数据文件:")
        for file_path in missing_files:
            print(f"   {file_path}")
        return False
    else:
        print("\n✅ 所有必需数据文件都已准备就绪")
        return True

def test_data_quality():
    """测试数据质量"""
    print("\n🎯 测试数据质量...")
    
    try:
        import pandas as pd
        from auto_trader.core.data_loader import DataQualityChecker
        
        # 创建数据质量检查器
        quality_checker = DataQualityChecker()
        
        # 测试BTC数据质量
        print("\n📊 BTC数据质量检查:")
        btc_data = pd.read_csv('binance_historical_data/processed/BTCUSDT_1h_combined.csv')
        btc_data['timestamp'] = pd.to_datetime(btc_data['timestamp'])
        
        btc_quality = quality_checker.check_data_quality(btc_data, 'BTCUSDT', '1h')
        print(f"   数据量: {len(btc_data):,}条")
        print(f"   质量评分: {btc_quality['score']:.1f}/100")
        print(f"   时间范围: {btc_data['timestamp'].min()} 到 {btc_data['timestamp'].max()}")
        
        # 测试ETH数据质量
        print("\n📊 ETH数据质量检查:")
        eth_data = pd.read_csv('binance_historical_data/processed/ETHUSDT_1h_combined.csv')
        eth_data['timestamp'] = pd.to_datetime(eth_data['timestamp'])
        
        eth_quality = quality_checker.check_data_quality(eth_data, 'ETHUSDT', '1h')
        print(f"   数据量: {len(eth_data):,}条")
        print(f"   质量评分: {eth_quality['score']:.1f}/100")
        print(f"   时间范围: {eth_data['timestamp'].min()} 到 {eth_data['timestamp'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据质量检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 策略优化测试开始...")
    print("=" * 60)
    
    # 1. 检查数据可用性
    if not check_data_availability():
        print("❌ 数据文件不完整，请先运行数据下载器")
        return
    
    # 2. 测试数据质量
    if not test_data_quality():
        print("❌ 数据质量检查失败")
        return
    
    # 3. 测试单策略优化
    if not test_single_strategy_optimization():
        print("❌ 单策略优化测试失败")
        return
    
    # 4. 测试多策略优化
    if not test_multi_strategy_optimization():
        print("❌ 多策略优化测试失败")
        return
    
    print("\n🎉 所有测试完成！")
    print("=" * 60)
    
    # 显示结果文件位置
    print("\n📁 结果文件位置:")
    print("   - 单策略优化结果: optimization_results/")
    print("   - 多策略优化结果: multi_strategy_optimization/")
    
    print("\n💡 下一步建议:")
    print("   1. 查看生成的优化报告")
    print("   2. 分析策略排行榜")
    print("   3. 选择表现最佳的策略进行实盘测试")

if __name__ == "__main__":
    main()