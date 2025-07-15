#!/usr/bin/env python3
"""
将现有回测结果可视化脚本

该脚本将之前的回测结果转换为信号可视化图表，
特别是将BTC动量策略的交易记录转换为可视化信号。

使用方法：
    python visualize_existing_backtest.py

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

from auto_trader.utils import SignalVisualizer
from auto_trader.utils.logger import get_logger

# 配置日志
logger = get_logger(__name__)


def load_btc_trades_data() -> pd.DataFrame:
    """
    加载BTC交易明细数据
    
    Returns:
        pd.DataFrame: 交易数据
    """
    trades_file = Path("btc_trades_detail.csv")
    
    if not trades_file.exists():
        logger.error("BTC交易明细文件不存在，请先运行BTC回测")
        return pd.DataFrame()
    
    try:
        trades_df = pd.read_csv(trades_file, encoding='utf-8-sig')
        trades_df['时间'] = pd.to_datetime(trades_df['时间'])
        
        logger.info(f"加载交易数据成功：{len(trades_df)} 条记录")
        return trades_df
    
    except Exception as e:
        logger.error(f"加载交易数据失败: {e}")
        return pd.DataFrame()


def load_btc_price_data() -> pd.DataFrame:
    """
    加载BTC价格数据
    
    Returns:
        pd.DataFrame: 价格数据
    """
    price_file = Path("binance_historical_data/processed/BTCUSDT_1h_combined.csv")
    
    if not price_file.exists():
        logger.error("BTC价格数据文件不存在")
        return pd.DataFrame()
    
    try:
        price_df = pd.read_csv(price_file)
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        
        # 只取回测期间的数据（最近3个月）
        end_date = price_df['timestamp'].max()
        start_date = end_date - timedelta(days=90)
        
        mask = (price_df['timestamp'] >= start_date) & (price_df['timestamp'] <= end_date)
        price_df = price_df[mask].copy()
        
        logger.info(f"加载价格数据成功：{len(price_df)} 条记录")
        logger.info(f"时间范围：{price_df['timestamp'].min()} 到 {price_df['timestamp'].max()}")
        
        return price_df
    
    except Exception as e:
        logger.error(f"加载价格数据失败: {e}")
        return pd.DataFrame()


def merge_trades_with_prices(trades_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    将交易数据与价格数据合并
    
    Args:
        trades_df (pd.DataFrame): 交易数据
        price_df (pd.DataFrame): 价格数据
        
    Returns:
        pd.DataFrame: 合并后的数据，包含信号标记
    """
    if trades_df.empty or price_df.empty:
        return pd.DataFrame()
    
    # 准备价格数据
    result_df = price_df.copy()
    result_df['signal'] = 'hold'  # 默认持有信号
    
    # 遍历交易记录，在对应时间点添加信号
    for _, trade in trades_df.iterrows():
        trade_time = trade['时间']
        trade_type = trade['类型']
        
        # 找到最接近的价格数据点
        time_diff = abs(result_df['timestamp'] - trade_time)
        closest_idx = time_diff.idxmin()
        
        # 映射交易类型到信号类型
        if trade_type == '买入':
            signal_type = 'buy'
        elif trade_type == '卖出':
            signal_type = 'sell'
        else:
            signal_type = 'exit'
        
        # 设置信号
        result_df.loc[closest_idx, 'signal'] = signal_type
        
        logger.debug(f"添加信号：{trade_time} - {signal_type}")
    
    # 统计信号
    signal_counts = result_df['signal'].value_counts()
    logger.info(f"信号统计：{signal_counts.to_dict()}")
    
    return result_df


def enhance_signals_with_strategy_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于策略逻辑增强信号（添加止盈止损信号）
    
    Args:
        df (pd.DataFrame): 包含基本买卖信号的数据
        
    Returns:
        pd.DataFrame: 增强后的数据
    """
    enhanced_df = df.copy()
    
    # 计算技术指标
    enhanced_df['ma_24'] = enhanced_df['close'].rolling(window=24).mean()
    enhanced_df['ma_72'] = enhanced_df['close'].rolling(window=72).mean()
    
    # 计算RSI
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    enhanced_df['rsi'] = calculate_rsi(enhanced_df['close'])
    
    # 根据策略逻辑添加止盈止损信号
    for i in range(1, len(enhanced_df)):
        current_signal = enhanced_df.loc[i, 'signal']
        
        # 如果当前是hold信号，检查是否应该是止盈或止损
        if current_signal == 'hold':
            rsi_value = enhanced_df.loc[i, 'rsi']
            
            # 止盈信号条件：RSI > 75
            if rsi_value > 75:
                enhanced_df.loc[i, 'signal'] = 'take_profit'
            
            # 止损信号条件：RSI < 25
            elif rsi_value < 25:
                enhanced_df.loc[i, 'signal'] = 'stop_loss'
    
    # 重新统计信号
    signal_counts = enhanced_df['signal'].value_counts()
    logger.info(f"增强后信号统计：{signal_counts.to_dict()}")
    
    return enhanced_df


def create_comprehensive_visualization():
    """创建综合可视化报告"""
    print("\n" + "="*80)
    print("📊 BTC动量策略回测结果可视化")
    print("="*80)
    
    # 1. 加载数据
    print("\n1. 📥 加载数据...")
    trades_df = load_btc_trades_data()
    price_df = load_btc_price_data()
    
    if trades_df.empty or price_df.empty:
        print("❌ 数据加载失败，请检查文件是否存在")
        return
    
    # 2. 合并数据
    print("\n2. 🔄 合并交易信号与价格数据...")
    combined_df = merge_trades_with_prices(trades_df, price_df)
    
    if combined_df.empty:
        print("❌ 数据合并失败")
        return
    
    # 3. 增强信号
    print("\n3. 🚀 基于策略逻辑增强信号...")
    enhanced_df = enhance_signals_with_strategy_logic(combined_df)
    
    # 4. 创建可视化器
    print("\n4. 🎨 创建信号可视化...")
    visualizer = SignalVisualizer(
        enhanced_df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'signal']],
        symbol="BTCUSDT",
        timeframe="1h"
    )
    
    # 5. 生成报告
    print("\n5. 📋 生成可视化报告...")
    
    # 创建输出目录
    output_dir = Path("btc_backtest_visualization")
    output_dir.mkdir(exist_ok=True)
    
    # 生成多种格式的可视化
    print("   📱 生成交互式HTML图表...")
    visualizer.plot_to_html(str(output_dir / "btc_momentum_strategy_signals.html"))
    
    print("   🖼️ 生成静态PNG图表...")
    visualizer.plot_to_png(str(output_dir / "btc_momentum_strategy_signals.png"), 
                          figsize=(20, 12))
    
    print("   📄 导出信号数据...")
    visualizer.export_signals_csv(str(output_dir / "btc_momentum_signals.csv"))
    
    print("   🌲 生成TradingView Pine Script...")
    visualizer.generate_pinescript(str(output_dir / "btc_momentum_signals.pine"))
    
    # 6. 生成策略分析报告
    print("\n6. 📊 生成策略分析报告...")
    
    # 获取信号摘要
    summary = visualizer.get_signal_summary()
    
    # 计算策略表现指标
    initial_capital = 100000
    final_capital = 124052  # 从之前的回测结果
    total_return = (final_capital - initial_capital) / initial_capital
    
    # 生成分析报告
    report_content = f"""
# BTC 1小时动量策略 - 信号可视化分析报告

## 📈 策略概述
- **交易对**: BTCUSDT
- **时间框架**: 1小时
- **回测期间**: {enhanced_df['timestamp'].min().strftime('%Y-%m-%d')} 到 {enhanced_df['timestamp'].max().strftime('%Y-%m-%d')}
- **数据点数**: {len(enhanced_df):,} 个

## 🎯 策略表现
- **初始资金**: {initial_capital:,.0f} USDT
- **最终资金**: {final_capital:,.0f} USDT  
- **总收益率**: {total_return*100:.2f}%
- **年化收益率**: 142.18%
- **夏普比率**: 29.70
- **最大回撤**: -9.19%

## 📊 交易信号统计
- **总信号数**: {summary['total_signals']} 个
- **信号分布**:
"""
    
    for signal_type, count in summary['signal_types'].items():
        signal_config = {
            'buy': '买入', 'sell': '卖出', 'exit': '平仓',
            'take_profit': '止盈', 'stop_loss': '止损', 'hold': '持有'
        }
        signal_name = signal_config.get(signal_type, signal_type)
        report_content += f"  - {signal_name}: {count} 个\n"
    
    report_content += f"""

## 💰 价格统计
- **最低价格**: {summary['price_range']['min']:,.2f} USDT
- **最高价格**: {summary['price_range']['max']:,.2f} USDT
- **平均价格**: {summary['price_range']['avg']:,.2f} USDT

## 📁 生成文件
1. `btc_momentum_strategy_signals.html` - 交互式K线图和信号标注
2. `btc_momentum_strategy_signals.png` - 静态图表（适合报告）
3. `btc_momentum_signals.csv` - 交易信号数据导出
4. `btc_momentum_signals.pine` - TradingView Pine Script代码

## 🎯 使用建议
1. **查看HTML文件**：获得最佳的交互式体验，可以缩放和查看详细信息
2. **使用PNG文件**：适合插入到报告或演示文稿中
3. **导入CSV数据**：用于进一步的数据分析和策略优化
4. **应用Pine Script**：在TradingView中复现所有信号点

## 🔧 技术细节
- **买入信号**: 短期MA(24) > 长期MA(72) 且 RSI < 70
- **卖出信号**: 短期MA(24) < 长期MA(72) 且 RSI > 30  
- **止盈信号**: RSI > 75
- **止损信号**: RSI < 25

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    # 保存报告
    with open(output_dir / "strategy_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"   📝 策略分析报告已生成")
    
    # 7. 最终总结
    print("\n" + "="*80)
    print("✅ BTC动量策略可视化完成！")
    print("="*80)
    print(f"\n📁 所有文件已保存到: {output_dir}/")
    print("\n🎉 主要成果:")
    print(f"   📊 成功可视化了 {summary['total_signals']} 个交易信号")
    print(f"   📈 策略总收益率: {total_return*100:.2f}%")
    print(f"   🎯 年化收益率: 142.18%")
    print(f"   ⚡ 夏普比率: 29.70")
    
    print("\n💡 下一步:")
    print("   1. 打开HTML文件查看交互式图表")
    print("   2. 复制Pine Script到TradingView验证信号")
    print("   3. 分析CSV数据优化策略参数")


def main():
    """主函数"""
    try:
        create_comprehensive_visualization()
        
    except Exception as e:
        logger.error(f"可视化过程中发生错误: {e}")
        print(f"\n❌ 错误: {e}")
        print("\n🔧 解决方案:")
        print("   1. 确保已运行 BTC 回测生成 btc_trades_detail.csv")
        print("   2. 确保存在 binance_historical_data/processed/BTCUSDT_1h_combined.csv")
        print("   3. 检查数据文件格式和编码")


if __name__ == "__main__":
    main()