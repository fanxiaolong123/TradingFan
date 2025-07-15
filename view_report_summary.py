#!/usr/bin/env python3
"""
回测报告查看器

快速查看和总结回测结果
"""

import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def view_optimization_results():
    """查看优化结果摘要"""
    print("📊 TradingFan 回测结果摘要")
    print("=" * 60)
    
    # 检查结果目录
    results_dir = Path("optimization_results")
    if not results_dir.exists():
        print("❌ 优化结果目录不存在")
        return
    
    # 查找所有结果文件
    csv_files = list(results_dir.glob("*_detailed.csv"))
    json_files = list(results_dir.glob("*_summary.json"))
    
    print(f"📁 找到 {len(csv_files)} 个详细结果文件")
    print(f"📁 找到 {len(json_files)} 个摘要文件")
    
    if not csv_files and not json_files:
        print("❌ 没有找到任何结果文件")
        return
    
    # 读取所有结果
    all_results = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_results.extend(df.to_dict('records'))
        except Exception as e:
            print(f"⚠️ 读取 {csv_file} 失败: {e}")
    
    if not all_results:
        print("❌ 没有有效的结果数据")
        return
    
    # 转换为DataFrame进行分析
    df = pd.DataFrame(all_results)
    
    # 数据清洗
    numeric_cols = ['sharpe_ratio', 'annualized_return', 'max_drawdown', 'win_rate', 'total_trades']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(f"\n📈 回测结果分析 (共 {len(df)} 个结果)")
    print("-" * 50)
    
    # 夏普比率统计
    if 'sharpe_ratio' in df.columns:
        sharpe_stats = df['sharpe_ratio'].describe()
        print(f"📊 夏普比率统计:")
        print(f"   平均值: {sharpe_stats['mean']:.3f}")
        print(f"   最大值: {sharpe_stats['max']:.3f}")
        print(f"   最小值: {sharpe_stats['min']:.3f}")
        print(f"   标准差: {sharpe_stats['std']:.3f}")
    
    # 年化收益率统计
    if 'annualized_return' in df.columns:
        return_stats = df['annualized_return'].describe()
        print(f"\n💰 年化收益率统计:")
        print(f"   平均值: {return_stats['mean']*100:.2f}%")
        print(f"   最大值: {return_stats['max']*100:.2f}%")
        print(f"   最小值: {return_stats['min']*100:.2f}%")
        print(f"   标准差: {return_stats['std']*100:.2f}%")
    
    # 最大回撤统计
    if 'max_drawdown' in df.columns:
        drawdown_stats = df['max_drawdown'].describe()
        print(f"\n📉 最大回撤统计:")
        print(f"   平均值: {drawdown_stats['mean']*100:.2f}%")
        print(f"   最大值: {drawdown_stats['max']*100:.2f}%")
        print(f"   最小值: {drawdown_stats['min']*100:.2f}%")
        print(f"   标准差: {drawdown_stats['std']*100:.2f}%")
    
    # 胜率统计
    if 'win_rate' in df.columns:
        win_rate_stats = df['win_rate'].describe()
        print(f"\n🎯 胜率统计:")
        print(f"   平均值: {win_rate_stats['mean']*100:.1f}%")
        print(f"   最大值: {win_rate_stats['max']*100:.1f}%")
        print(f"   最小值: {win_rate_stats['min']*100:.1f}%")
        print(f"   标准差: {win_rate_stats['std']*100:.1f}%")
    
    # 找出最佳结果
    print(f"\n🏆 最佳结果 (按夏普比率排序):")
    print("-" * 50)
    
    if 'sharpe_ratio' in df.columns:
        # 按夏普比率排序
        top_results = df.nlargest(5, 'sharpe_ratio')
        
        for i, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"\n#{i}")
            print(f"   夏普比率: {row.get('sharpe_ratio', 0):.3f}")
            print(f"   年化收益率: {row.get('annualized_return', 0)*100:.2f}%")
            print(f"   最大回撤: {row.get('max_drawdown', 0)*100:.2f}%")
            print(f"   胜率: {row.get('win_rate', 0)*100:.1f}%")
            print(f"   交易次数: {row.get('total_trades', 0)}")
            
            # 显示参数
            params = eval(row.get('params', '{}')) if isinstance(row.get('params'), str) else row.get('params', {})
            if params:
                print(f"   参数: {params}")
    
    # 性能阈值分析
    print(f"\n🎯 性能阈值分析:")
    print("-" * 50)
    
    if 'sharpe_ratio' in df.columns:
        sharpe_above_1 = len(df[df['sharpe_ratio'] > 1])
        print(f"   夏普比率 > 1.0: {sharpe_above_1}/{len(df)} ({sharpe_above_1/len(df)*100:.1f}%)")
    
    if 'annualized_return' in df.columns:
        return_above_30 = len(df[df['annualized_return'] > 0.3])
        print(f"   年化收益率 > 30%: {return_above_30}/{len(df)} ({return_above_30/len(df)*100:.1f}%)")
    
    if 'max_drawdown' in df.columns:
        drawdown_below_20 = len(df[df['max_drawdown'] > -0.2])
        print(f"   最大回撤 < 20%: {drawdown_below_20}/{len(df)} ({drawdown_below_20/len(df)*100:.1f}%)")
    
    if 'win_rate' in df.columns:
        win_rate_above_60 = len(df[df['win_rate'] > 0.6])
        print(f"   胜率 > 60%: {win_rate_above_60}/{len(df)} ({win_rate_above_60/len(df)*100:.1f}%)")
    
    # 检查是否有专业化报告
    print(f"\n📊 可视化报告:")
    print("-" * 50)
    
    reports_dir = Path("reports/html")
    if reports_dir.exists():
        html_files = list(reports_dir.glob("*.html"))
        if html_files:
            latest_report = max(html_files, key=lambda x: x.stat().st_mtime)
            print(f"   最新报告: {latest_report}")
            print(f"   生成时间: {datetime.fromtimestamp(latest_report.stat().st_mtime)}")
            print(f"   大小: {latest_report.stat().st_size:,} bytes")
            print(f"   📖 在浏览器中打开: file://{latest_report.absolute()}")
        else:
            print("   暂无可视化报告")
    else:
        print("   暂无可视化报告")
    
    print(f"\n💡 建议:")
    print("-" * 50)
    print("1. 运行 'python3 generate_professional_report.py' 生成专业化可视化报告")
    print("2. 运行 'python3 demo_optimization.py' 进行新的策略优化")
    print("3. 检查 optimization_results/ 目录查看详细数据")

def main():
    """主函数"""
    try:
        view_optimization_results()
    except Exception as e:
        print(f"❌ 查看结果失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()