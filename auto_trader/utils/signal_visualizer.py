#!/usr/bin/env python3
"""
交易信号可视化模块

该模块用于将本地回测系统中的交易信号可视化，生成类似TradingView的买卖点标注图。
支持K线图绘制、信号标注、信号导出和Pine Script代码生成。

作者：量化交易系统
版本：1.0.0
创建时间：2025-07-16
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple
import warnings
from pathlib import Path
import logging

# 设置中文字体和忽略警告
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 配置日志
logger = logging.getLogger(__name__)


class SignalVisualizer:
    """
    交易信号可视化器
    
    主要功能：
    1. K线图绘制
    2. 交易信号标注（买入、卖出、止盈、止损）
    3. 图表保存（PNG、HTML、SVG）
    4. 信号数据导出（CSV）
    5. TradingView Pine Script代码生成
    
    属性：
        data (pd.DataFrame): 包含OHLCV和信号的数据
        signals (pd.DataFrame): 提取的信号数据
        symbol (str): 交易对符号
        timeframe (str): 时间周期
    """
    
    def __init__(self, data: pd.DataFrame, symbol: str = "CRYPTO", timeframe: str = "1h"):
        """
        初始化信号可视化器
        
        Args:
            data (pd.DataFrame): 包含OHLCV和信号的数据框
            symbol (str): 交易对符号，默认"CRYPTO"
            timeframe (str): 时间周期，默认"1h"
        """
        self.data = self._validate_and_prepare_data(data)  # 验证并准备数据
        self.signals = self._extract_signals()  # 提取信号数据
        self.symbol = symbol  # 交易对符号
        self.timeframe = timeframe  # 时间周期
        
        # 信号配置字典，定义不同信号的显示样式
        self.signal_config = {
            'buy': {'color': 'green', 'marker': '^', 'size': 100, 'label': '买入'},
            'sell': {'color': 'red', 'marker': 'v', 'size': 100, 'label': '卖出'},
            'exit': {'color': 'orange', 'marker': 'x', 'size': 80, 'label': '平仓'},
            'take_profit': {'color': 'lime', 'marker': 's', 'size': 60, 'label': '止盈'},
            'stop_loss': {'color': 'crimson', 'marker': 'd', 'size': 60, 'label': '止损'},
            'hold': {'color': 'gray', 'marker': 'o', 'size': 30, 'label': '持有'}
        }
        
        logger.info(f"SignalVisualizer初始化完成: {symbol} {timeframe}")
        logger.info(f"数据量: {len(self.data)} 条, 信号数: {len(self.signals)} 个")
    
    def _validate_and_prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        验证并准备数据
        
        Args:
            data (pd.DataFrame): 原始数据
            
        Returns:
            pd.DataFrame: 验证后的数据
            
        Raises:
            ValueError: 当数据格式不正确时抛出异常
        """
        # 检查必需的列
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"数据缺少必需的列: {missing_columns}")
        
        # 复制数据避免修改原始数据
        df = data.copy()
        
        # 处理时间戳列
        if df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        elif df['timestamp'].dtype in ['int64', 'float64']:
            # 判断是秒级还是毫秒级时间戳
            sample_timestamp = df['timestamp'].iloc[0]
            if sample_timestamp > 1e10:  # 毫秒级
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            else:  # 秒级
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # 如果没有signal列，创建默认的hold信号
        if 'signal' not in df.columns:
            df['signal'] = 'hold'
            logger.warning("数据中没有signal列，已创建默认hold信号")
        
        # 按时间戳排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 验证OHLC数据的合理性
        invalid_ohlc = (df['high'] < df['low']) | (df['high'] < df['open']) | \
                      (df['high'] < df['close']) | (df['low'] > df['open']) | \
                      (df['low'] > df['close'])
        
        if invalid_ohlc.any():
            logger.warning(f"发现 {invalid_ohlc.sum()} 条不合理的OHLC数据")
        
        logger.info(f"数据验证完成，时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        
        return df
    
    def _extract_signals(self) -> pd.DataFrame:
        """
        从数据中提取交易信号
        
        Returns:
            pd.DataFrame: 包含信号信息的数据框
        """
        # 过滤出有效信号（非hold的信号）
        signals = self.data[self.data['signal'] != 'hold'].copy()
        
        # 如果没有有效信号，返回空DataFrame
        if signals.empty:
            logger.warning("数据中没有找到有效的交易信号")
            return pd.DataFrame(columns=['timestamp', 'signal', 'price'])
        
        # 添加价格列（使用收盘价作为信号价格）
        signals['price'] = signals['close']
        
        # 添加信号编号
        signals['signal_id'] = range(1, len(signals) + 1)
        
        # 重置索引
        signals = signals.reset_index(drop=True)
        
        # 统计信号类型
        signal_counts = signals['signal'].value_counts()
        logger.info(f"信号统计: {signal_counts.to_dict()}")
        
        return signals[['timestamp', 'signal', 'price', 'signal_id']]
    
    def plot_matplotlib(self, figsize: Tuple[int, int] = (16, 10), 
                       show_volume: bool = True, save_path: Optional[str] = None) -> None:
        """
        使用matplotlib绘制K线图和交易信号
        
        Args:
            figsize (Tuple[int, int]): 图表大小，默认(16, 10)
            show_volume (bool): 是否显示成交量，默认True
            save_path (Optional[str]): 保存路径，如果为None则不保存
        """
        # 根据是否显示成交量确定子图数量
        if show_volume:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                          gridspec_kw={'height_ratios': [3, 1]})
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=figsize)
            ax2 = None
        
        # 绘制K线图
        self._plot_candlesticks_matplotlib(ax1)
        
        # 绘制交易信号
        self._plot_signals_matplotlib(ax1)
        
        # 绘制成交量（如果需要）
        if show_volume and ax2 is not None:
            self._plot_volume_matplotlib(ax2)
        
        # 设置标题和标签
        title = f"{self.symbol} {self.timeframe} - 交易信号分析"
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"matplotlib图表已保存到: {save_path}")
        
        # 显示图表
        plt.show()
    
    def _plot_candlesticks_matplotlib(self, ax) -> None:
        """
        使用matplotlib绘制K线图
        
        Args:
            ax: matplotlib坐标轴对象
        """
        # 遍历每个数据点绘制K线
        for idx, row in self.data.iterrows():
            timestamp = row['timestamp']
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']
            
            # 确定K线颜色（绿涨红跌）
            color = 'green' if close_price >= open_price else 'red'
            
            # 绘制高低线
            ax.plot([timestamp, timestamp], [low_price, high_price], 
                   color=color, linewidth=0.8, alpha=0.8)
            
            # 绘制实体部分
            height = abs(close_price - open_price)
            bottom = min(open_price, close_price)
            
            # 计算K线宽度（基于时间间隔）
            if len(self.data) > 1:
                time_diff = (self.data['timestamp'].iloc[1] - self.data['timestamp'].iloc[0])
                width = time_diff * 0.6  # K线宽度为时间间隔的60%
            else:
                width = timedelta(hours=1)  # 默认1小时宽度
            
            # 绘制矩形实体
            rect = Rectangle((timestamp - width/2, bottom), width, height,
                           facecolor=color, edgecolor=color, alpha=0.8)
            ax.add_patch(rect)
        
        # 设置坐标轴
        ax.set_ylabel('价格 (USDT)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴时间显示
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=max(1, len(self.data) // 20)))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_signals_matplotlib(self, ax) -> None:
        """
        在matplotlib图表上绘制交易信号
        
        Args:
            ax: matplotlib坐标轴对象
        """
        # 遍历所有信号类型
        for signal_type, config in self.signal_config.items():
            # 筛选当前信号类型的数据
            signal_data = self.signals[self.signals['signal'] == signal_type]
            
            if signal_data.empty:
                continue
            
            # 绘制信号点
            ax.scatter(signal_data['timestamp'], signal_data['price'],
                      color=config['color'], marker=config['marker'],
                      s=config['size'], label=config['label'],
                      zorder=5, alpha=0.9, edgecolors='white', linewidth=1)
            
            # 添加信号编号标注
            for _, row in signal_data.iterrows():
                ax.annotate(f"{row['signal_id']}", 
                           xy=(row['timestamp'], row['price']),
                           xytext=(0, 15 if signal_type == 'buy' else -20),
                           textcoords='offset points',
                           fontsize=8, ha='center',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=config['color'], alpha=0.7),
                           color='white', fontweight='bold')
        
        # 添加图例
        ax.legend(loc='upper left', framealpha=0.9)
    
    def _plot_volume_matplotlib(self, ax) -> None:
        """
        使用matplotlib绘制成交量图
        
        Args:
            ax: matplotlib坐标轴对象
        """
        # 绘制成交量柱状图
        colors = ['green' if row['close'] >= row['open'] else 'red' 
                 for _, row in self.data.iterrows()]
        
        ax.bar(self.data['timestamp'], self.data['volume'], 
               color=colors, alpha=0.6, width=0.8)
        
        # 设置坐标轴
        ax.set_ylabel('成交量', fontsize=12)
        ax.set_xlabel('时间', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 格式化x轴
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def plot_plotly(self, show_volume: bool = True, save_path: Optional[str] = None,
                   auto_open: bool = True) -> go.Figure:
        """
        使用plotly绘制交互式K线图和交易信号
        
        Args:
            show_volume (bool): 是否显示成交量，默认True
            save_path (Optional[str]): 保存路径，如果为None则不保存
            auto_open (bool): 是否自动打开浏览器显示，默认True
            
        Returns:
            go.Figure: plotly图表对象
        """
        # 根据是否显示成交量创建子图
        if show_volume:
            fig = make_subplots(rows=2, cols=1, 
                              shared_xaxes=True,
                              vertical_spacing=0.1,
                              subplot_titles=(f'{self.symbol} {self.timeframe} K线图', '成交量'),
                              row_width=[0.7, 0.3])
        else:
            fig = go.Figure()
        
        # 绘制K线图
        candlestick = go.Candlestick(
            x=self.data['timestamp'],
            open=self.data['open'],
            high=self.data['high'],
            low=self.data['low'],
            close=self.data['close'],
            name='K线',
            increasing_line_color='green',
            decreasing_line_color='red'
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # 绘制交易信号
        self._plot_signals_plotly(fig, row=1 if show_volume else None)
        
        # 绘制成交量
        if show_volume:
            self._plot_volume_plotly(fig)
        
        # 设置布局
        title = f"{self.symbol} {self.timeframe} - 交易信号分析 (共{len(self.signals)}个信号)"
        fig.update_layout(
            title=title,
            xaxis_title="时间",
            yaxis_title="价格 (USDT)",
            template="plotly_white",
            showlegend=True,
            height=800 if show_volume else 600,
            hovermode='x unified'
        )
        
        # 隐藏rangeslider
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        # 保存图表
        if save_path:
            if save_path.endswith('.html'):
                fig.write_html(save_path)
            elif save_path.endswith('.png'):
                fig.write_image(save_path, width=1600, height=800)
            elif save_path.endswith('.svg'):
                fig.write_image(save_path, format='svg')
            else:
                fig.write_html(save_path + '.html')
            
            logger.info(f"plotly图表已保存到: {save_path}")
        
        # 显示图表
        if auto_open:
            pyo.plot(fig, auto_open=True)
        
        return fig
    
    def _plot_signals_plotly(self, fig: go.Figure, row: Optional[int] = None) -> None:
        """
        在plotly图表上绘制交易信号
        
        Args:
            fig (go.Figure): plotly图表对象
            row (Optional[int]): 子图行号，如果为None则为单图
        """
        # 定义plotly的标记符号映射
        plotly_symbols = {
            'buy': 'triangle-up',
            'sell': 'triangle-down', 
            'exit': 'x',
            'take_profit': 'square',
            'stop_loss': 'diamond',
            'hold': 'circle'
        }
        
        # 遍历所有信号类型
        for signal_type, config in self.signal_config.items():
            signal_data = self.signals[self.signals['signal'] == signal_type]
            
            if signal_data.empty:
                continue
            
            # 创建散点图
            scatter = go.Scatter(
                x=signal_data['timestamp'],
                y=signal_data['price'],
                mode='markers+text',
                name=config['label'],
                marker=dict(
                    symbol=plotly_symbols.get(signal_type, 'circle'),
                    size=config['size'] // 5,  # plotly使用较小的尺寸
                    color=config['color'],
                    line=dict(width=2, color='white')
                ),
                text=[f"#{row['signal_id']}" for _, row in signal_data.iterrows()],
                textposition="top center" if signal_type == 'buy' else "bottom center",
                textfont=dict(size=10, color='white'),
                hovertemplate=f"<b>{config['label']}</b><br>" +
                            "时间: %{x}<br>" +
                            "价格: %{y:.2f}<br>" +
                            "信号ID: %{text}<br>" +
                            "<extra></extra>"
            )
            
            # 添加到图表
            if row is not None:
                fig.add_trace(scatter, row=row, col=1)
            else:
                fig.add_trace(scatter)
    
    def _plot_volume_plotly(self, fig: go.Figure) -> None:
        """
        使用plotly绘制成交量图
        
        Args:
            fig (go.Figure): plotly图表对象
        """
        # 创建成交量柱状图
        colors = ['green' if row['close'] >= row['open'] else 'red' 
                 for _, row in self.data.iterrows()]
        
        volume_bar = go.Bar(
            x=self.data['timestamp'],
            y=self.data['volume'],
            name='成交量',
            marker_color=colors,
            opacity=0.6,
            hovertemplate="时间: %{x}<br>成交量: %{y:,.0f}<extra></extra>"
        )
        
        fig.add_trace(volume_bar, row=2, col=1)
        
        # 更新y轴标签
        fig.update_yaxes(title_text="成交量", row=2, col=1)
    
    def export_signals_csv(self, filepath: str) -> None:
        """
        导出交易信号到CSV文件
        
        Args:
            filepath (str): 导出文件路径
        """
        if self.signals.empty:
            logger.warning("没有交易信号可导出")
            return
        
        # 准备导出数据
        export_data = self.signals.copy()
        
        # 格式化时间戳
        export_data['timestamp_str'] = export_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # 添加额外信息
        export_data['symbol'] = self.symbol
        export_data['timeframe'] = self.timeframe
        
        # 重新排列列顺序
        columns = ['signal_id', 'timestamp', 'timestamp_str', 'signal', 'price', 'symbol', 'timeframe']
        export_data = export_data[columns]
        
        # 保存到CSV
        export_data.to_csv(filepath, index=False, encoding='utf-8-sig')
        
        logger.info(f"交易信号已导出到CSV: {filepath}")
        logger.info(f"导出 {len(export_data)} 个信号")
    
    def generate_pinescript(self, output_path: Optional[str] = None) -> str:
        """
        生成TradingView Pine Script代码
        
        Args:
            output_path (Optional[str]): 输出文件路径，如果为None则只返回代码字符串
            
        Returns:
            str: Pine Script代码
        """
        if self.signals.empty:
            logger.warning("没有交易信号，无法生成Pine Script")
            return ""
        
        # Pine Script模板开头
        pinescript = f'''// @version=5
indicator("{self.symbol} {self.timeframe} 交易信号复现", overlay=true)

// 信号数据来源：本地回测系统
// 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// 总信号数：{len(self.signals)}

'''
        
        # 为每种信号类型生成代码
        for signal_type in self.signals['signal'].unique():
            signal_data = self.signals[self.signals['signal'] == signal_type]
            config = self.signal_config.get(signal_type, self.signal_config['hold'])
            
            # 生成时间戳数组
            timestamps = [int(ts.timestamp() * 1000) for ts in signal_data['timestamp']]
            prices = signal_data['price'].tolist()
            
            pinescript += f'''
// {config['label']}信号 ({len(signal_data)}个)
{signal_type}_timestamps = array.from({timestamps})
{signal_type}_prices = array.from({prices})

// 检查当前K线是否有{config['label']}信号
{signal_type}_signal = false
for i = 0 to array.size({signal_type}_timestamps) - 1
    if time == array.get({signal_type}_timestamps, i)
        {signal_type}_signal := true
        break

// 绘制{config['label']}信号
plotshape({signal_type}_signal, title="{config['label']}", 
          style=shape.{'triangleup' if signal_type == 'buy' else 'triangledown' if signal_type == 'sell' else 'circle'}, 
          location=location.{'belowbar' if signal_type == 'buy' else 'abovebar'}, 
          color=color.{config['color']}, size=size.small)
'''
        
        # 添加信息表格
        pinescript += f'''
// 信息统计表格
var table infoTable = table.new(position.top_right, 2, {len(self.signals['signal'].value_counts()) + 2})

if barstate.islast
    table.cell(infoTable, 0, 0, "信号类型", bgcolor=color.gray, text_color=color.white)
    table.cell(infoTable, 1, 0, "数量", bgcolor=color.gray, text_color=color.white)
    
'''
        
        # 添加统计信息
        row = 1
        for signal_type, count in self.signals['signal'].value_counts().items():
            config = self.signal_config.get(signal_type, self.signal_config['hold'])
            pinescript += f'''    table.cell(infoTable, 0, {row}, "{config['label']}", text_color=color.white)
    table.cell(infoTable, 1, {row}, "{count}", text_color=color.{config['color']})
'''
            row += 1
        
        # 添加总计
        pinescript += f'''    table.cell(infoTable, 0, {row}, "总计", bgcolor=color.blue, text_color=color.white)
    table.cell(infoTable, 1, {row}, "{len(self.signals)}", bgcolor=color.blue, text_color=color.white)
'''
        
        # 保存到文件
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(pinescript)
            logger.info(f"Pine Script代码已保存到: {output_path}")
        
        return pinescript
    
    def plot_to_html(self, filepath: str, show_volume: bool = True) -> None:
        """
        生成HTML格式的交互式图表
        
        Args:
            filepath (str): 输出文件路径
            show_volume (bool): 是否显示成交量
        """
        fig = self.plot_plotly(show_volume=show_volume, save_path=filepath, auto_open=False)
        logger.info(f"HTML图表已生成: {filepath}")
    
    def plot_to_png(self, filepath: str, figsize: Tuple[int, int] = (16, 10), 
                   show_volume: bool = True) -> None:
        """
        生成PNG格式的静态图表
        
        Args:
            filepath (str): 输出文件路径
            figsize (Tuple[int, int]): 图表大小
            show_volume (bool): 是否显示成交量
        """
        self.plot_matplotlib(figsize=figsize, show_volume=show_volume, save_path=filepath)
    
    def get_signal_summary(self) -> Dict:
        """
        获取信号统计摘要
        
        Returns:
            Dict: 包含信号统计信息的字典
        """
        if self.signals.empty:
            return {"total_signals": 0, "signal_types": {}}
        
        # 基本统计
        summary = {
            "total_signals": len(self.signals),
            "signal_types": self.signals['signal'].value_counts().to_dict(),
            "time_range": {
                "start": self.signals['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                "end": self.signals['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            "price_range": {
                "min": float(self.signals['price'].min()),
                "max": float(self.signals['price'].max()),
                "avg": float(self.signals['price'].mean())
            }
        }
        
        return summary


def create_sample_data() -> pd.DataFrame:
    """
    创建示例数据用于测试
    
    Returns:
        pd.DataFrame: 包含OHLCV和信号的示例数据
    """
    # 生成示例价格数据
    np.random.seed(42)
    n_points = 100
    
    # 生成时间序列
    start_time = datetime(2025, 1, 1)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]
    
    # 生成价格数据（随机游走）
    price_base = 50000  # 基础价格
    price_changes = np.random.normal(0, 0.02, n_points)  # 2%的标准差
    prices = [price_base]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    # 生成OHLCV数据
    data = []
    for i, (timestamp, close_price) in enumerate(zip(timestamps, prices)):
        # 生成开高低价
        open_price = prices[i-1] if i > 0 else close_price
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.01)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.lognormal(15, 1)
        
        # 生成随机信号
        signal_prob = np.random.random()
        if signal_prob < 0.05:
            signal = 'buy'
        elif signal_prob < 0.10:
            signal = 'sell'
        elif signal_prob < 0.12:
            signal = 'exit'
        else:
            signal = 'hold'
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'signal': signal
        })
    
    return pd.DataFrame(data)


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    sample_data = create_sample_data()
    
    # 创建可视化器
    visualizer = SignalVisualizer(sample_data, symbol="BTCUSDT", timeframe="1h")
    
    # 打印信号摘要
    summary = visualizer.get_signal_summary()
    print("信号摘要:", summary)
    
    # 生成各种格式的图表
    visualizer.plot_to_html("signal_demo.html")
    visualizer.plot_to_png("signal_demo.png")
    
    # 导出信号数据
    visualizer.export_signals_csv("signals_demo.csv")
    
    # 生成Pine Script
    pine_code = visualizer.generate_pinescript("signals_demo.pine")
    print("\nPine Script代码已生成")