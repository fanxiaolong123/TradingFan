#!/usr/bin/env python3
"""
专业化回测报告生成脚本

这个脚本将现有的回测结果转换为专业化的可视化报告
包含完整的图表、分析和HTML输出
"""

import sys
import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_trader.utils.professional_report_generator import ProfessionalReportGenerator

def collect_all_optimization_results(results_dir: str = "optimization_results") -> list:
    """
    收集所有优化结果
    
    Args:
        results_dir: 结果目录
        
    Returns:
        list: 所有优化结果
    """
    results = []
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ 结果目录不存在: {results_dir}")
        return results
    
    print(f"📁 搜索结果目录: {results_path}")
    
    # 搜索所有CSV文件
    csv_files = list(results_path.glob("*_detailed.csv"))
    json_files = list(results_path.glob("*_summary.json"))
    
    print(f"📊 找到 {len(csv_files)} 个详细结果文件")
    print(f"📋 找到 {len(json_files)} 个摘要文件")
    
    # 读取CSV文件
    for csv_file in csv_files:
        try:
            print(f"📖 读取文件: {csv_file}")
            df = pd.read_csv(csv_file)
            
            # 解析参数字符串
            if 'params' in df.columns:
                for idx, row in df.iterrows():
                    try:
                        # 如果params是字符串，尝试解析
                        if isinstance(row['params'], str):
                            params = eval(row['params'])
                        else:
                            params = row['params']
                        
                        # 创建结果记录
                        result = {
                            'strategy_name': csv_file.stem.split('_')[0],
                            'symbol': csv_file.stem.split('_')[1] if len(csv_file.stem.split('_')) > 1 else 'UNKNOWN',
                            'timeframe': csv_file.stem.split('_')[2] if len(csv_file.stem.split('_')) > 2 else '1h',
                            'params': params,
                            'total_return': row.get('total_return', 0),
                            'annualized_return': row.get('annualized_return', 0),
                            'volatility': row.get('volatility', 0),
                            'sharpe_ratio': row.get('sharpe_ratio', 0),
                            'sortino_ratio': row.get('sortino_ratio', 0),
                            'max_drawdown': row.get('max_drawdown', 0),
                            'win_rate': row.get('win_rate', 0),
                            'profit_factor': row.get('profit_factor', 0),
                            'var_95': row.get('var_95', 0),
                            'cvar_95': row.get('cvar_95', 0),
                            'calmar_ratio': row.get('calmar_ratio', 0),
                            'total_trades': row.get('total_trades', 0),
                            'avg_win': row.get('avg_win', 0),
                            'avg_loss': row.get('avg_loss', 0)
                        }
                        
                        # 合并参数到结果中
                        if isinstance(params, dict):
                            result.update(params)
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"⚠️ 解析行数据失败: {e}")
                        continue
                        
        except Exception as e:
            print(f"❌ 读取文件失败 {csv_file}: {e}")
            continue
    
    # 读取JSON文件作为补充
    for json_file in json_files:
        try:
            print(f"📖 读取JSON文件: {json_file}")
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取关键信息
            if 'performance_metrics' in data:
                metrics = data['performance_metrics']
                params = data.get('best_params', {})
                
                result = {
                    'strategy_name': data.get('strategy_name', 'unknown'),
                    'symbol': data.get('symbol', 'UNKNOWN'),
                    'timeframe': data.get('timeframe', '1h'),
                    'params': params,
                    'total_return': metrics.get('total_return', 0),
                    'annualized_return': metrics.get('annualized_return', 0),
                    'volatility': metrics.get('volatility', 0),
                    'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                    'sortino_ratio': metrics.get('sortino_ratio', 0),
                    'max_drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'profit_factor': metrics.get('profit_factor', 0),
                    'var_95': metrics.get('var_95', 0),
                    'cvar_95': metrics.get('cvar_95', 0),
                    'calmar_ratio': metrics.get('calmar_ratio', 0),
                    'total_trades': metrics.get('total_trades', 0),
                    'avg_win': metrics.get('avg_win', 0),
                    'avg_loss': metrics.get('avg_loss', 0)
                }
                
                # 合并参数
                if isinstance(params, dict):
                    result.update(params)
                
                results.append(result)
                
        except Exception as e:
            print(f"❌ 读取JSON文件失败 {json_file}: {e}")
            continue
    
    print(f"✅ 总共收集到 {len(results)} 个结果")
    return results

def generate_report_from_existing_results():
    """从现有结果生成专业化报告"""
    print("🚀 开始生成专业化回测报告...")
    
    # 1. 收集所有结果
    all_results = collect_all_optimization_results()
    
    if not all_results:
        print("❌ 没有找到任何优化结果，请先运行策略优化")
        return
    
    # 2. 创建报告生成器
    report_generator = ProfessionalReportGenerator(output_dir="reports")
    
    # 3. 生成综合报告
    try:
        report_path = report_generator.generate_comprehensive_report(
            optimization_results=all_results,
            title="TradingFan 量化策略回测分析报告"
        )
        
        print(f"✅ 专业化报告已生成: {report_path}")
        print(f"📊 包含 {len(all_results)} 个策略结果")
        
        # 4. 生成数据摘要
        df = pd.DataFrame(all_results)
        summary_stats = {
            "总策略数": len(df),
            "平均夏普比率": df['sharpe_ratio'].mean(),
            "最高夏普比率": df['sharpe_ratio'].max(),
            "平均年化收益率": df['annualized_return'].mean() * 100,
            "最高年化收益率": df['annualized_return'].max() * 100,
            "夏普比率>1的策略": len(df[df['sharpe_ratio'] > 1]),
            "年化收益率>30%的策略": len(df[df['annualized_return'] > 0.3]),
            "最大回撤<-20%的策略": len(df[df['max_drawdown'] > -0.2])
        }
        
        print("\n📈 数据摘要:")
        print("=" * 50)
        for key, value in summary_stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        # 5. 显示最佳策略
        if len(df) > 0:
            best_strategy = df.loc[df['sharpe_ratio'].idxmax()]
            print(f"\n🏆 最佳策略:")
            print(f"策略: {best_strategy['strategy_name']}")
            print(f"交易对: {best_strategy['symbol']}")
            print(f"时间框架: {best_strategy['timeframe']}")
            print(f"夏普比率: {best_strategy['sharpe_ratio']:.3f}")
            print(f"年化收益率: {best_strategy['annualized_return']*100:.2f}%")
            print(f"最大回撤: {best_strategy['max_drawdown']*100:.2f}%")
            print(f"胜率: {best_strategy['win_rate']*100:.1f}%")
        
        return report_path
        
    except Exception as e:
        print(f"❌ 生成报告失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_individual_reports():
    """为每个策略生成独立报告"""
    print("📊 生成各策略独立报告...")
    
    # 收集结果
    all_results = collect_all_optimization_results()
    
    if not all_results:
        print("❌ 没有找到任何优化结果")
        return
    
    # 按策略分组
    df = pd.DataFrame(all_results)
    
    report_generator = ProfessionalReportGenerator(output_dir="reports/individual")
    
    for strategy_name in df['strategy_name'].unique():
        strategy_results = df[df['strategy_name'] == strategy_name].to_dict('records')
        
        try:
            report_path = report_generator.generate_comprehensive_report(
                optimization_results=strategy_results,
                title=f"{strategy_name} 策略回测分析报告"
            )
            
            print(f"✅ {strategy_name} 策略报告已生成: {report_path}")
            
        except Exception as e:
            print(f"❌ {strategy_name} 策略报告生成失败: {e}")

def main():
    """主函数"""
    print("🎯 TradingFan 专业化回测报告生成器")
    print("=" * 60)
    
    # 检查结果目录是否存在
    if not Path("optimization_results").exists():
        print("❌ 优化结果目录不存在，请先运行策略优化")
        print("💡 运行建议: python demo_optimization.py")
        return
    
    # 生成综合报告
    print("1️⃣ 生成综合报告...")
    comprehensive_report = generate_report_from_existing_results()
    
    if comprehensive_report:
        print(f"\n🎉 综合报告生成成功!")
        print(f"📁 报告位置: {comprehensive_report}")
        
        # 询问是否生成个别报告
        print("\n2️⃣ 是否生成各策略独立报告? (可选)")
        try:
            choice = input("输入 y/yes 继续，其他任意键跳过: ").lower()
            if choice in ['y', 'yes']:
                generate_individual_reports()
        except KeyboardInterrupt:
            print("\n✨ 用户中断，报告生成完成")
        
        print("\n📊 报告功能特点:")
        print("✅ 交互式图表 (Plotly)")
        print("✅ 多维度性能分析")
        print("✅ 参数敏感性分析")
        print("✅ 风险收益可视化")
        print("✅ 专业级HTML报告")
        print("✅ 响应式设计")
        
        print(f"\n🌐 在浏览器中打开: {comprehensive_report}")
        
    else:
        print("❌ 报告生成失败，请检查数据和配置")

if __name__ == "__main__":
    main()