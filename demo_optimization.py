#!/usr/bin/env python3
"""
策略优化演示脚本

演示策略优化框架的核心功能
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_trader.core.strategy_optimizer import StrategyOptimizer, OptimizationConfig

def demo_optimization():
    """演示策略优化"""
    print("🎯 策略优化演示")
    print("=" * 50)
    
    # 创建优化配置
    config = OptimizationConfig(
        strategy_name='momentum',
        symbol='BTCUSDT',
        timeframe='1h',
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 1, 7),  # 使用1周数据快速演示
        param_ranges={
            'short_ma_period': [12, 20],
            'long_ma_period': [50, 100],
            'position_size': [0.2, 0.3]
        },
        optimization_method='grid_search',
        max_iterations=8,
        target_metric='sharpe_ratio',
        n_jobs=1,
        performance_threshold={
            'win_rate': 0.60,
            'annualized_return': 0.30,
            'max_drawdown': -0.20,
            'sharpe_ratio': 1.0,
            'total_trades': 5
        }
    )
    
    print(f"📊 策略: {config.strategy_name}")
    print(f"🪙 币种: {config.symbol}")
    print(f"⏱️ 周期: {config.timeframe}")
    print(f"📅 时间: {config.start_date.strftime('%Y-%m-%d')} 到 {config.end_date.strftime('%Y-%m-%d')}")
    print(f"🎯 参数组合数: {len(config.param_ranges['short_ma_period']) * len(config.param_ranges['long_ma_period']) * len(config.param_ranges['position_size'])}")
    
    try:
        # 创建优化器
        optimizer = StrategyOptimizer(config)
        
        # 运行优化
        print("\n🔄 开始优化...")
        result = optimizer.optimize()
        
        # 显示结果
        print("\n📊 优化结果:")
        print("-" * 50)
        print(f"最优参数: {result.params}")
        print(f"性能评分: {result.performance_score:.1f}/100")
        print(f"年化收益率: {result.annualized_return*100:.2f}%")
        print(f"夏普比率: {result.sharpe_ratio:.3f}")
        print(f"最大回撤: {result.max_drawdown*100:.2f}%")
        print(f"胜率: {result.win_rate*100:.1f}%")
        print(f"交易次数: {result.total_trades}")
        print(f"是否达标: {'✅ 通过' if result.meets_threshold else '❌ 未通过'}")
        
        # 生成报告
        print("\n📝 生成详细报告...")
        report = optimizer.generate_optimization_report(result)
        
        # 保存报告
        report_file = Path("optimization_demo_report.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📁 报告已保存到: {report_file}")
        
        # 生成专业化可视化报告
        print("\n🎨 生成专业化可视化报告...")
        try:
            from auto_trader.utils.professional_report_generator import ProfessionalReportGenerator
            
            # 创建报告生成器
            report_generator = ProfessionalReportGenerator(output_dir="reports/demo")
            
            # 从优化结果目录读取数据
            import pandas as pd
            import json
            
            # 读取刚生成的结果
            results_dir = Path("optimization_results")
            latest_files = sorted(results_dir.glob("*_detailed.csv"))
            
            if latest_files:
                latest_file = latest_files[-1]
                df = pd.read_csv(latest_file)
                results_data = df.to_dict('records')
                
                # 生成可视化报告
                html_report = report_generator.generate_comprehensive_report(
                    optimization_results=results_data,
                    title="策略优化演示报告"
                )
                
                print(f"✅ 可视化报告已生成: {html_report}")
                print("🌐 请在浏览器中打开查看完整的交互式报告")
            else:
                print("⚠️ 未找到优化结果文件，跳过可视化报告生成")
                
        except ImportError:
            print("⚠️ 专业化报告生成器未安装，跳过可视化报告")
        except Exception as e:
            print(f"⚠️ 生成可视化报告失败: {e}")
        
        print("\n🎉 演示完成!")
        
        return True
        
    except Exception as e:
        print(f"❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("🚀 策略优化系统演示")
    print("=" * 60)
    
    # 检查数据文件
    btc_file = Path("binance_historical_data/processed/BTCUSDT_1h_combined.csv")
    if not btc_file.exists():
        print("❌ 未找到BTC数据文件，请先运行数据下载器")
        return
    
    print(f"✅ 数据文件: {btc_file}")
    print(f"📊 文件大小: {btc_file.stat().st_size:,} bytes")
    
    # 运行演示
    if demo_optimization():
        print("\n💡 完整功能:")
        print("   1. 运行 run_full_optimization.py 进行完整优化")
        print("   2. 查看 optimization_results/ 目录获取详细结果")
        print("   3. 查看 multi_strategy_optimization/ 目录获取排行榜")
    
    print("\n📋 系统功能总结:")
    print("   ✅ 策略参数优化")
    print("   ✅ 多策略批量优化")
    print("   ✅ 性能评分和阈值检查")
    print("   ✅ 详细报告生成")
    print("   ✅ 排行榜和统计分析")
    print("   ✅ 并行处理支持")

if __name__ == "__main__":
    main()