#!/usr/bin/env python3
"""
BTCUSDT 1h 动量策略详细分析
在K线图上显示买卖点
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from real_historical_backtest import RealHistoricalBacktester, PerformanceMetrics
from auto_trader.strategies.base import StrategyConfig

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class BTCMomentumAnalyzer:
    """BTC动量策略详细分析器"""
    
    def __init__(self):
        self.trades = []
        self.price_data = None
        self.signals = []
        
    def run_detailed_backtest(self):
        """运行详细回测并记录所有交易"""
        print("🚀 开始BTCUSDT 1h动量策略详细分析")
        print("=" * 80)
        
        # 创建回测引擎
        engine = RealHistoricalBacktester()
        
        # 策略参数
        parameters = {
            'short_ma_period': 24,
            'long_ma_period': 72,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'momentum_period': 48,
            'momentum_threshold': 0.02,
            'volume_threshold': 1.5,
            'position_size': 0.3,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.06
        }
        
        # 创建策略配置
        strategy_config = StrategyConfig(
            name="momentum_BTCUSDT_1h",
            symbol="BTCUSDT",
            timeframe="1h",
            parameters=parameters
        )
        
        # 设置回测时间范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=89)
        
        # 运行回测
        metrics = engine.run_backtest(strategy_config, start_date, end_date)
        
        # 保存交易记录和价格数据
        self.trades = engine.trades
        
        # 加载价格数据
        data_path = Path("binance_historical_data/processed/BTCUSDT_1h_combined.csv")
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 筛选回测期间的数据
        mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
        self.price_data = df[mask].copy()
        
        return metrics, engine
    
    def create_detailed_kline_chart(self, output_path="btc_momentum_details.png"):
        """创建带买卖点的K线图"""
        print("\n📊 生成K线图和买卖点标注...")
        
        # 准备数据
        df = self.price_data.copy()
        df = df.set_index('timestamp')
        
        # 创建图表
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # 主K线图
        ax1 = axes[0]
        
        # 绘制K线
        for idx, row in df.iterrows():
            color = 'g' if row['close'] >= row['open'] else 'r'
            # 绘制高低线
            ax1.plot([idx, idx], [row['low'], row['high']], color=color, linewidth=0.5)
            # 绘制开收柱
            height = abs(row['close'] - row['open'])
            bottom = min(row['close'], row['open'])
            rect = Rectangle((mdates.date2num(idx) - 0.02, bottom), 0.04, height, 
                           facecolor=color, edgecolor=color)
            ax1.add_patch(rect)
        
        # 添加均线
        df['MA24'] = df['close'].rolling(window=24).mean()
        df['MA72'] = df['close'].rolling(window=72).mean()
        ax1.plot(df.index, df['MA24'], 'b-', label='MA24', linewidth=1.5, alpha=0.7)
        ax1.plot(df.index, df['MA72'], 'orange', label='MA72', linewidth=1.5, alpha=0.7)
        
        # 标注买卖点
        buy_trades = [t for t in self.trades if t.side == 'BUY']
        sell_trades = [t for t in self.trades if t.side == 'SELL']
        
        # 买入点
        for trade in buy_trades:
            ax1.scatter(trade.timestamp, trade.price, color='green', marker='^', 
                       s=100, zorder=5, alpha=0.8)
            # 添加价格标签
            ax1.annotate(f'{trade.price:.0f}', 
                        xy=(trade.timestamp, trade.price),
                        xytext=(0, -15), 
                        textcoords='offset points',
                        fontsize=8, 
                        ha='center',
                        color='green')
        
        # 卖出点
        for trade in sell_trades:
            ax1.scatter(trade.timestamp, trade.price, color='red', marker='v', 
                       s=100, zorder=5, alpha=0.8)
            # 添加价格标签
            ax1.annotate(f'{trade.price:.0f}', 
                        xy=(trade.timestamp, trade.price),
                        xytext=(0, 15), 
                        textcoords='offset points',
                        fontsize=8, 
                        ha='center',
                        color='red')
        
        ax1.set_title('BTCUSDT 1小时 K线图 - 动量策略买卖点', fontsize=16)
        ax1.set_ylabel('价格 (USDT)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 成交量图
        ax2 = axes[1]
        colors = ['g' if row['close'] >= row['open'] else 'r' for _, row in df.iterrows()]
        ax2.bar(df.index, df['volume'], color=colors, alpha=0.5)
        ax2.set_ylabel('成交量', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # RSI指标
        ax3 = axes[2]
        
        # 计算RSI
        close_delta = df['close'].diff()
        gains = close_delta.where(close_delta > 0, 0)
        losses = -close_delta.where(close_delta < 0, 0)
        avg_gains = gains.rolling(window=14).mean()
        avg_losses = losses.rolling(window=14).mean()
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        ax3.plot(df.index, rsi, 'purple', linewidth=1.5)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5)
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5)
        ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        ax3.set_ylabel('RSI', fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        # 格式化x轴
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:00'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=24))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ K线图已保存: {output_path}")
    
    def generate_trade_statistics(self):
        """生成交易统计报告"""
        print("\n📊 交易统计分析")
        print("=" * 80)
        
        # 基础统计
        total_trades = len(self.trades)
        buy_trades = [t for t in self.trades if t.side == 'BUY']
        sell_trades = [t for t in self.trades if t.side == 'SELL']
        
        print(f"总交易次数: {total_trades}")
        print(f"买入次数: {len(buy_trades)}")
        print(f"卖出次数: {len(sell_trades)}")
        
        # 计算盈亏
        profitable_trades = []
        losing_trades = []
        
        for trade in self.trades:
            if trade.side == 'SELL' and trade.pnl != 0:
                if trade.pnl > 0:
                    profitable_trades.append(trade)
                else:
                    losing_trades.append(trade)
        
        win_rate = len(profitable_trades) / len(sell_trades) * 100 if sell_trades else 0
        
        print(f"\n盈利交易: {len(profitable_trades)}")
        print(f"亏损交易: {len(losing_trades)}")
        print(f"胜率: {win_rate:.1f}%")
        
        if profitable_trades:
            avg_profit = np.mean([t.pnl for t in profitable_trades])
            max_profit = max([t.pnl for t in profitable_trades])
            print(f"平均盈利: {avg_profit:.2f} USDT")
            print(f"最大单笔盈利: {max_profit:.2f} USDT")
        
        if losing_trades:
            avg_loss = np.mean([t.pnl for t in losing_trades])
            max_loss = min([t.pnl for t in losing_trades])
            print(f"平均亏损: {avg_loss:.2f} USDT")
            print(f"最大单笔亏损: {max_loss:.2f} USDT")
        
        # 持仓时间分析
        holding_times = []
        for i in range(len(sell_trades)):
            # 找到对应的买入交易
            sell_time = sell_trades[i].timestamp
            # 找到最近的买入交易
            recent_buys = [b for b in buy_trades if b.timestamp < sell_time]
            if recent_buys:
                buy_time = recent_buys[-1].timestamp
                holding_time = (sell_time - buy_time).total_seconds() / 3600  # 小时
                holding_times.append(holding_time)
        
        if holding_times:
            print(f"\n平均持仓时间: {np.mean(holding_times):.1f} 小时")
            print(f"最长持仓时间: {max(holding_times):.1f} 小时")
            print(f"最短持仓时间: {min(holding_times):.1f} 小时")
    
    def generate_detailed_trade_list(self, output_path="btc_trades_detail.csv"):
        """生成详细交易列表"""
        trade_data = []
        
        for trade in self.trades:
            trade_data.append({
                '时间': trade.timestamp.strftime('%Y-%m-%d %H:%M'),
                '类型': '买入' if trade.side == 'BUY' else '卖出',
                '价格': f"{trade.price:.2f}",
                '数量': f"{trade.quantity:.6f}",
                '金额': f"{trade.price * trade.quantity:.2f}",
                '手续费': f"{trade.commission:.2f}",
                '盈亏': f"{trade.pnl:.2f}" if trade.pnl != 0 else "-",
                '累计盈亏': f"{trade.cumulative_pnl:.2f}" if trade.cumulative_pnl != 0 else "-"
            })
        
        df = pd.DataFrame(trade_data)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 交易明细已保存: {output_path}")
        
        # 打印最近10笔交易
        print("\n最近10笔交易:")
        print(df.tail(10).to_string(index=False))
    
    def create_performance_chart(self, metrics, output_path="btc_performance.png"):
        """创建策略表现图表"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 资金曲线
        ax1 = axes[0, 0]
        equity_curve = metrics.equity_curve
        dates = pd.date_range(start=self.price_data['timestamp'].min(), 
                            periods=len(equity_curve), freq='H')
        ax1.plot(dates, equity_curve, 'b-', linewidth=2)
        ax1.fill_between(dates, 100000, equity_curve, alpha=0.3)
        ax1.set_title('资金曲线', fontsize=14)
        ax1.set_ylabel('账户余额 (USDT)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 收益率曲线
        ax2 = axes[0, 1]
        returns = [(v - 100000) / 100000 * 100 for v in equity_curve]
        ax2.plot(dates, returns, 'g-', linewidth=2)
        ax2.fill_between(dates, 0, returns, where=[r > 0 for r in returns], 
                        alpha=0.3, color='green', label='盈利')
        ax2.fill_between(dates, 0, returns, where=[r < 0 for r in returns], 
                        alpha=0.3, color='red', label='亏损')
        ax2.set_title('累计收益率', fontsize=14)
        ax2.set_ylabel('收益率 (%)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 回撤分析
        ax3 = axes[1, 0]
        # 计算回撤
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max * 100
        ax3.fill_between(dates, 0, drawdowns, alpha=0.5, color='red')
        ax3.plot(dates, drawdowns, 'r-', linewidth=1)
        ax3.set_title('回撤分析', fontsize=14)
        ax3.set_ylabel('回撤 (%)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        
        # 月度收益
        ax4 = axes[1, 1]
        # 计算月度收益
        monthly_returns = []
        current_month = self.price_data['timestamp'].min().month
        month_start_value = equity_curve[0]
        month_values = []
        
        for i, date in enumerate(dates):
            if date.month != current_month:
                # 计算上个月收益
                if month_values:
                    month_end_value = month_values[-1]
                    monthly_return = (month_end_value - month_start_value) / month_start_value * 100
                    monthly_returns.append(monthly_return)
                    month_start_value = month_end_value
                current_month = date.month
                month_values = [equity_curve[i]]
            else:
                month_values.append(equity_curve[i])
        
        # 添加最后一个月
        if month_values:
            month_end_value = month_values[-1]
            monthly_return = (month_end_value - month_start_value) / month_start_value * 100
            monthly_returns.append(monthly_return)
        
        months = ['4月', '5月', '6月', '7月'][:len(monthly_returns)]
        colors = ['g' if r > 0 else 'r' for r in monthly_returns]
        bars = ax4.bar(months, monthly_returns, color=colors, alpha=0.7)
        ax4.set_title('月度收益率', fontsize=14)
        ax4.set_ylabel('收益率 (%)', fontsize=12)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, value in zip(bars, monthly_returns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', 
                    va='bottom' if height > 0 else 'top')
        
        # 调整布局
        for ax in axes.flat:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 策略表现图表已保存: {output_path}")

def main():
    """主函数"""
    analyzer = BTCMomentumAnalyzer()
    
    # 运行回测
    metrics, engine = analyzer.run_detailed_backtest()
    
    # 生成详细报告
    print("\n" + "="*80)
    print("📊 BTCUSDT 1小时动量策略详细报告")
    print("="*80)
    
    # 打印策略表现
    report = engine.generate_report(metrics, "BTCUSDT 1h Momentum Strategy")
    print(report)
    
    # 生成交易统计
    analyzer.generate_trade_statistics()
    
    # 生成K线图
    analyzer.create_detailed_kline_chart()
    
    # 生成策略表现图表
    analyzer.create_performance_chart(metrics)
    
    # 生成交易明细
    analyzer.generate_detailed_trade_list()
    
    print("\n✅ 所有分析完成！")
    print("\n生成的文件:")
    print("1. btc_momentum_details.png - K线图和买卖点")
    print("2. btc_performance.png - 策略表现分析")
    print("3. btc_trades_detail.csv - 详细交易记录")

if __name__ == "__main__":
    main()