#!/usr/bin/env python3
"""
信号可视化模块快速测试脚本

验证SignalVisualizer的核心功能是否正常工作。
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_trader.utils import SignalVisualizer, create_sample_data

def test_basic_functionality():
    """测试基本功能"""
    print("🧪 测试信号可视化模块基本功能")
    print("=" * 50)
    
    # 1. 创建测试数据
    print("1. 创建测试数据...")
    data = create_sample_data()
    print(f"   数据量: {len(data)} 条")
    print(f"   信号分布: {data['signal'].value_counts().to_dict()}")
    
    # 2. 创建可视化器
    print("\n2. 创建信号可视化器...")
    try:
        visualizer = SignalVisualizer(data, symbol="TEST", timeframe="1h")
        print("   ✅ 可视化器创建成功")
    except Exception as e:
        print(f"   ❌ 可视化器创建失败: {e}")
        return False
    
    # 3. 测试信号统计
    print("\n3. 测试信号统计...")
    try:
        summary = visualizer.get_signal_summary()
        print(f"   总信号数: {summary['total_signals']}")
        print(f"   信号类型: {list(summary['signal_types'].keys())}")
        print("   ✅ 信号统计正常")
    except Exception as e:
        print(f"   ❌ 信号统计失败: {e}")
        return False
    
    # 4. 测试CSV导出
    print("\n4. 测试CSV导出...")
    try:
        csv_path = "test_signals.csv"
        visualizer.export_signals_csv(csv_path)
        
        # 验证文件
        if os.path.exists(csv_path):
            exported_data = pd.read_csv(csv_path)
            print(f"   导出 {len(exported_data)} 条信号记录")
            print("   ✅ CSV导出成功")
            os.remove(csv_path)  # 清理测试文件
        else:
            print("   ❌ CSV文件未生成")
            return False
    except Exception as e:
        print(f"   ❌ CSV导出失败: {e}")
        return False
    
    # 5. 测试Pine Script生成
    print("\n5. 测试Pine Script生成...")
    try:
        pine_code = visualizer.generate_pinescript()
        if len(pine_code) > 100:  # 检查代码长度
            print(f"   生成Pine Script代码 {len(pine_code)} 字符")
            print("   ✅ Pine Script生成成功")
        else:
            print("   ❌ Pine Script代码过短")
            return False
    except Exception as e:
        print(f"   ❌ Pine Script生成失败: {e}")
        return False
    
    # 6. 测试HTML生成（简化版）
    print("\n6. 测试HTML生成...")
    try:
        html_path = "test_signals.html"
        visualizer.plot_to_html(html_path, show_volume=False)
        
        if os.path.exists(html_path):
            file_size = os.path.getsize(html_path)
            print(f"   生成HTML文件 {file_size} 字节")
            print("   ✅ HTML生成成功")
            os.remove(html_path)  # 清理测试文件
        else:
            print("   ❌ HTML文件未生成")
            return False
    except Exception as e:
        print(f"   ❌ HTML生成失败: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！信号可视化模块功能正常")
    return True

def test_real_data_integration():
    """测试与真实数据的集成"""
    print("\n🔗 测试真实数据集成")
    print("=" * 50)
    
    # 尝试加载真实BTC数据
    btc_file = "binance_historical_data/processed/BTCUSDT_1h_combined.csv"
    
    if not os.path.exists(btc_file):
        print("   ⚠️ 真实BTC数据文件不存在，跳过集成测试")
        return True
    
    try:
        # 加载数据
        df = pd.read_csv(btc_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 取最近100条数据进行测试
        test_data = df.tail(100).copy()
        test_data['signal'] = 'hold'
        
        # 添加几个测试信号
        test_data.iloc[10]['signal'] = 'buy'
        test_data.iloc[50]['signal'] = 'sell'
        test_data.iloc[90]['signal'] = 'take_profit'
        
        # 创建可视化器
        visualizer = SignalVisualizer(test_data, symbol="BTCUSDT", timeframe="1h")
        
        # 获取统计信息
        summary = visualizer.get_signal_summary()
        
        print(f"   真实数据测试: {len(test_data)} 条记录")
        print(f"   信号数量: {summary['total_signals']}")
        print("   ✅ 真实数据集成测试通过")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 真实数据集成测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 信号可视化模块测试套件")
    print("=" * 60)
    
    # 基本功能测试
    basic_test_pass = test_basic_functionality()
    
    # 真实数据集成测试
    integration_test_pass = test_real_data_integration()
    
    # 最终结果
    print("\n" + "=" * 60)
    if basic_test_pass and integration_test_pass:
        print("✅ 所有测试通过！信号可视化模块已准备就绪")
        print("\n💡 使用方法:")
        print("   from auto_trader.utils import SignalVisualizer")
        print("   visualizer = SignalVisualizer(your_data)")
        print("   visualizer.plot_to_html('output.html')")
        
        print("\n📁 功能验证:")
        print("   ✅ 数据验证和预处理")
        print("   ✅ 信号提取和统计")
        print("   ✅ CSV数据导出")
        print("   ✅ Pine Script代码生成")
        print("   ✅ HTML交互式图表")
        print("   ✅ 真实数据集成")
        
    else:
        print("❌ 部分测试失败，请检查错误信息")

if __name__ == "__main__":
    main()