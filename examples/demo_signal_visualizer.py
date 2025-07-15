#!/usr/bin/env python3
"""
交易信号可视化模块演示脚本

该脚本展示了SignalVisualizer的所有功能：
1. 从真实回测数据创建可视化
2. 生成多种格式的图表
3. 导出信号数据
4. 生成Pine Script代码

使用方法：
    python demo_signal_visualizer.py

作者：量化交易系统
版本：1.0.0
创建时间：2025-07-16
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_trader.utils import SignalVisualizer, create_sample_data
from auto_trader.utils.logger import get_logger

# 配置日志
logger = get_logger(__name__)


def load_real_backtest_data() -> pd.DataFrame:
    """
    加载真实的回测数据并添加模拟交易信号
    
    Returns:
        pd.DataFrame: 包含OHLCV和信号的数据
    """
    # 尝试加载真实的BTC数据
    data_path = Path("binance_historical_data/processed/BTCUSDT_1h_combined.csv")
    
    if not data_path.exists():
        logger.warning("真实数据文件不存在，使用示例数据")
        return create_sample_data()
    
    try:
        # 加载真实数据
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 取最近500条数据用于演示
        df = df.tail(500).copy().reset_index(drop=True)
        
        # 添加模拟交易信号（基于简单的技术指标）
        df['signal'] = 'hold'  # 默认持有
        
        # 计算移动平均线
        df['ma_short'] = df['close'].rolling(window=20).mean()
        df['ma_long'] = df['close'].rolling(window=50).mean()
        
        # 计算RSI
        def calculate_rsi(prices, period=14):
            """计算RSI指标"""
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi'] = calculate_rsi(df['close'])
        
        # 生成交易信号
        for i in range(50, len(df)):  # 从第50行开始，确保有足够的历史数据
            # 买入信号：短期均线上穿长期均线 且 RSI < 70
            if (df.loc[i, 'ma_short'] > df.loc[i, 'ma_long'] and
                df.loc[i-1, 'ma_short'] <= df.loc[i-1, 'ma_long'] and
                df.loc[i, 'rsi'] < 70):
                df.loc[i, 'signal'] = 'buy'
            
            # 卖出信号：短期均线下穿长期均线 且 RSI > 30
            elif (df.loc[i, 'ma_short'] < df.loc[i, 'ma_long'] and
                  df.loc[i-1, 'ma_short'] >= df.loc[i-1, 'ma_long'] and
                  df.loc[i, 'rsi'] > 30):
                df.loc[i, 'signal'] = 'sell'
            
            # 止盈信号：RSI > 80
            elif df.loc[i, 'rsi'] > 80:
                df.loc[i, 'signal'] = 'take_profit'
            
            # 止损信号：RSI < 20
            elif df.loc[i, 'rsi'] < 20:
                df.loc[i, 'signal'] = 'stop_loss'
        
        # 随机添加一些exit信号
        exit_indices = np.random.choice(df.index[50:], size=3, replace=False)
        df.loc[exit_indices, 'signal'] = 'exit'
        
        logger.info(f"加载真实数据成功，数据量: {len(df)} 条")
        logger.info(f"信号统计: {df['signal'].value_counts().to_dict()}")
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'signal']]
    
    except Exception as e:
        logger.error(f"加载真实数据失败: {e}")
        logger.info("使用示例数据")
        return create_sample_data()


def demo_basic_functionality():
    """演示基本功能"""
    print("\n" + "="*80)
    print("🎯 交易信号可视化模块功能演示")
    print("="*80)
    
    # 1. 加载数据
    print("\n📊 1. 加载数据...")
    data = load_real_backtest_data()
    print(f"   数据量: {len(data)} 条")
    print(f"   时间范围: {data['timestamp'].min()} 到 {data['timestamp'].max()}")
    
    # 2. 创建可视化器
    print("\n🔧 2. 创建信号可视化器...")
    visualizer = SignalVisualizer(data, symbol="BTCUSDT", timeframe="1h")
    
    # 3. 获取信号摘要
    print("\n📈 3. 信号统计摘要:")
    summary = visualizer.get_signal_summary()
    print(f"   总信号数: {summary['total_signals']}")
    print("   信号类型分布:")
    for signal_type, count in summary['signal_types'].items():
        print(f"     - {signal_type}: {count} 个")
    print(f"   价格范围: {summary['price_range']['min']:.2f} - {summary['price_range']['max']:.2f}")
    
    return visualizer


def demo_visualization_outputs(visualizer: SignalVisualizer):
    """演示可视化输出功能"""
    print("\n📊 4. 生成可视化图表...")
    
    # 创建输出目录
    output_dir = Path("signal_visualization_demo")
    output_dir.mkdir(exist_ok=True)
    
    # 4.1 生成HTML交互式图表
    print("   📱 生成HTML交互式图表...")
    html_path = output_dir / "btc_signals_interactive.html"
    visualizer.plot_to_html(str(html_path), show_volume=True)
    
    # 4.2 生成PNG静态图表
    print("   🖼️ 生成PNG静态图表...")
    png_path = output_dir / "btc_signals_static.png"
    visualizer.plot_to_png(str(png_path), figsize=(20, 12), show_volume=True)
    
    # 4.3 生成plotly SVG图表
    print("   🎨 生成Plotly图表...")
    try:
        fig = visualizer.plot_plotly(show_volume=True, save_path=str(output_dir / "btc_signals.svg"), auto_open=False)
        print("   ✅ Plotly图表生成成功")
    except Exception as e:
        print(f"   ⚠️ Plotly图表生成失败: {e}")
    
    print(f"   📁 图表已保存到: {output_dir}/")


def demo_data_export(visualizer: SignalVisualizer):
    """演示数据导出功能"""
    print("\n💾 5. 导出功能演示...")
    
    output_dir = Path("signal_visualization_demo")
    
    # 5.1 导出信号CSV
    print("   📄 导出信号数据到CSV...")
    csv_path = output_dir / "btc_signals_export.csv"
    visualizer.export_signals_csv(str(csv_path))
    
    # 读取并显示导出的数据
    exported_data = pd.read_csv(csv_path)
    print(f"   ✅ 已导出 {len(exported_data)} 个信号到 {csv_path}")
    print("   前5行数据预览:")
    print(exported_data.head().to_string(index=False))


def demo_pinescript_generation(visualizer: SignalVisualizer):
    """演示Pine Script代码生成"""
    print("\n🌲 6. Pine Script代码生成...")
    
    output_dir = Path("signal_visualization_demo")
    
    # 生成Pine Script代码
    pine_path = output_dir / "btc_signals_tradingview.pine"
    pine_code = visualizer.generate_pinescript(str(pine_path))
    
    print(f"   ✅ Pine Script代码已生成: {pine_path}")
    print("   代码片段预览:")
    print("-" * 60)
    print(pine_code[:500] + "...")
    print("-" * 60)
    
    # 显示使用说明
    print("\n📝 TradingView使用说明:")
    print("   1. 复制生成的Pine Script代码")
    print("   2. 打开TradingView.com")
    print("   3. 打开Pine Editor")
    print("   4. 粘贴代码并点击'添加到图表'")
    print("   5. 在图表上查看信号点复现")


def demo_integration_example():
    """演示与回测系统集成的示例"""
    print("\n🔗 7. 回测系统集成示例...")
    
    # 模拟从回测引擎获取数据
    print("   模拟回测系统调用...")
    
    # 示例：假设这是从回测引擎获取的数据
    backtest_results = {
        'data': load_real_backtest_data(),
        'strategy_name': 'BTC_Momentum_Strategy',
        'performance': {
            'total_return': 0.2405,
            'sharpe_ratio': 2.97,
            'max_drawdown': -0.0919
        }
    }
    
    # 创建可视化器
    visualizer = SignalVisualizer(
        backtest_results['data'], 
        symbol="BTCUSDT", 
        timeframe="1h"
    )
    
    # 一键生成完整报告
    output_dir = Path("signal_visualization_demo")
    report_name = f"{backtest_results['strategy_name']}_signal_report"
    
    print(f"   📋 生成策略报告: {report_name}")
    
    # 生成所有输出
    visualizer.plot_to_html(str(output_dir / f"{report_name}.html"))
    visualizer.plot_to_png(str(output_dir / f"{report_name}.png"))
    visualizer.export_signals_csv(str(output_dir / f"{report_name}_signals.csv"))
    visualizer.generate_pinescript(str(output_dir / f"{report_name}.pine"))
    
    print("   ✅ 完整策略信号报告已生成")


def demo_advanced_features():
    """演示高级功能"""
    print("\n🚀 8. 高级功能演示...")
    
    # 创建更复杂的数据
    print("   📊 创建复杂信号数据...")
    
    # 使用真实数据并添加更多信号类型
    data = load_real_backtest_data()
    
    # 创建多层次可视化器
    visualizer = SignalVisualizer(data, symbol="BTCUSDT", timeframe="1h")
    
    # 演示批量处理
    output_dir = Path("signal_visualization_demo/advanced")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成不同时间段的图表
    print("   🎯 生成多时间段分析...")
    
    # 按月份分组生成图表
    data['month'] = pd.to_datetime(data['timestamp']).dt.to_period('M')
    
    for month, month_data in data.groupby('month'):
        if len(month_data) < 10:  # 跳过数据太少的月份
            continue
            
        month_visualizer = SignalVisualizer(
            month_data.drop('month', axis=1), 
            symbol=f"BTCUSDT_{month}", 
            timeframe="1h"
        )
        
        month_output = output_dir / f"btc_signals_{month}.html"
        month_visualizer.plot_to_html(str(month_output), show_volume=False)
        
        print(f"     📅 {month} 图表已生成")
    
    print("   ✅ 高级功能演示完成")


def main():
    """主演示函数"""
    try:
        # 基本功能演示
        visualizer = demo_basic_functionality()
        
        # 可视化输出演示
        demo_visualization_outputs(visualizer)
        
        # 数据导出演示
        demo_data_export(visualizer)
        
        # Pine Script生成演示
        demo_pinescript_generation(visualizer)
        
        # 系统集成演示
        demo_integration_example()
        
        # 高级功能演示
        demo_advanced_features()
        
        # 最终总结
        print("\n" + "="*80)
        print("🎉 交易信号可视化模块演示完成！")
        print("="*80)
        print("\n📁 生成的文件:")
        print("   signal_visualization_demo/")
        print("   ├── btc_signals_interactive.html    # 交互式图表")
        print("   ├── btc_signals_static.png          # 静态图表")
        print("   ├── btc_signals_export.csv          # 信号数据导出")
        print("   ├── btc_signals_tradingview.pine    # Pine Script代码")
        print("   ├── BTC_Momentum_Strategy_*.*        # 策略报告文件")
        print("   └── advanced/                       # 高级功能演示")
        
        print("\n🔧 使用方法:")
        print("   1. 查看HTML文件获得交互式图表体验")
        print("   2. 使用PNG文件作为报告插图")
        print("   3. 导入CSV文件进行进一步分析")
        print("   4. 复制Pine Script代码到TradingView")
        
        print("\n💡 集成到您的系统:")
        print("   from auto_trader.utils import SignalVisualizer")
        print("   visualizer = SignalVisualizer(your_data)")
        print("   visualizer.plot_to_html('output.html')")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        raise


if __name__ == "__main__":
    main()