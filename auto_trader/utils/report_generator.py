"""
交易报告生成器

该模块提供了生成交易报告的功能，支持：
- PDF格式报告
- HTML格式报告
- 详细的交易分析和统计
- 图表和可视化
- 风险指标分析
- 策略表现评估
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 尝试导入PDF生成相关库
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


@dataclass
class TradingMetrics:
    """交易指标数据"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_return: float = 0.0
    avg_return_per_trade: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_return': self.total_return,
            'avg_return_per_trade': self.avg_return_per_trade,
            'max_drawdown': self.max_drawdown,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'volatility': self.volatility,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy
        }


class ReportGenerator:
    """交易报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        (self.output_dir / "charts").mkdir(exist_ok=True)
        (self.output_dir / "pdf").mkdir(exist_ok=True)
        (self.output_dir / "html").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"报告生成器初始化完成，输出目录: {self.output_dir}")
    
    def calculate_trading_metrics(self, 
                                 trades_data: List[Dict],
                                 account_data: List[Dict]) -> TradingMetrics:
        """
        计算交易指标
        
        Args:
            trades_data: 交易数据
            account_data: 账户数据
            
        Returns:
            TradingMetrics: 交易指标对象
        """
        metrics = TradingMetrics()
        
        if not trades_data:
            return metrics
        
        # 基本统计
        metrics.total_trades = len(trades_data)
        
        # 计算盈亏
        profits = []
        losses = []
        
        for trade in trades_data:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                profits.append(pnl)
                metrics.winning_trades += 1
            elif pnl < 0:
                losses.append(abs(pnl))
                metrics.losing_trades += 1
        
        # 胜率
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # 收益率
        if account_data:
            initial_value = account_data[0].get('total_value', 0)
            final_value = account_data[-1].get('total_value', 0)
            if initial_value > 0:
                metrics.total_return = (final_value - initial_value) / initial_value
        
        # 平均收益
        if metrics.total_trades > 0:
            total_pnl = sum(trade.get('pnl', 0) for trade in trades_data)
            metrics.avg_return_per_trade = total_pnl / metrics.total_trades
        
        # 最大回撤
        if account_data:
            values = [data.get('total_value', 0) for data in account_data]
            peak_value = values[0]
            max_drawdown = 0
            
            for value in values:
                if value > peak_value:
                    peak_value = value
                else:
                    drawdown = (peak_value - value) / peak_value
                    max_drawdown = max(max_drawdown, drawdown)
            
            metrics.max_drawdown = max_drawdown
        
        # 夏普比率和索提诺比率
        if len(trades_data) > 1:
            returns = [trade.get('pnl', 0) for trade in trades_data]
            metrics.volatility = np.std(returns) if returns else 0
            
            if metrics.volatility > 0:
                avg_return = np.mean(returns)
                metrics.sharpe_ratio = avg_return / metrics.volatility
                
                # 索提诺比率（只考虑下行风险）
                negative_returns = [r for r in returns if r < 0]
                if negative_returns:
                    downside_deviation = np.std(negative_returns)
                    metrics.sortino_ratio = avg_return / downside_deviation
        
        # 卡尔马比率
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.total_return / metrics.max_drawdown
        
        # VaR计算
        if len(trades_data) > 10:
            returns = [trade.get('pnl', 0) for trade in trades_data]
            metrics.var_95 = np.percentile(returns, 5)
            metrics.var_99 = np.percentile(returns, 1)
        
        # 连续胜负次数
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in trades_data:
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            elif pnl < 0:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        metrics.max_consecutive_wins = max_consecutive_wins
        metrics.max_consecutive_losses = max_consecutive_losses
        
        # 盈利因子
        total_profit = sum(profits) if profits else 0
        total_loss = sum(losses) if losses else 0
        if total_loss > 0:
            metrics.profit_factor = total_profit / total_loss
        
        # 数学期望
        if metrics.total_trades > 0:
            avg_profit = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses) if losses else 0
            metrics.expectancy = (metrics.win_rate * avg_profit) - ((1 - metrics.win_rate) * avg_loss)
        
        return metrics
    
    def generate_charts(self, 
                       trades_data: List[Dict],
                       account_data: List[Dict],
                       chart_prefix: str = "chart") -> Dict[str, str]:
        """
        生成图表
        
        Args:
            trades_data: 交易数据
            account_data: 账户数据
            chart_prefix: 图表文件前缀
            
        Returns:
            Dict[str, str]: 图表文件路径字典
        """
        charts = {}
        
        # 设置图表样式
        plt.style.use('seaborn-v0_8')
        
        # 1. 账户价值曲线
        if account_data:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dates = [datetime.fromisoformat(data['date']) if isinstance(data['date'], str) 
                    else data['date'] for data in account_data]
            values = [data['total_value'] for data in account_data]
            
            ax.plot(dates, values, linewidth=2, color='blue', label='账户价值')
            ax.set_title('账户价值曲线', fontsize=16, fontweight='bold')
            ax.set_xlabel('日期', fontsize=12)
            ax.set_ylabel('账户价值 (USDT)', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 格式化x轴
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            chart_path = self.output_dir / "charts" / f"{chart_prefix}_account_value.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['account_value'] = str(chart_path)
        
        # 2. 每日收益分布
        if trades_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            pnl_values = [trade.get('pnl', 0) for trade in trades_data]
            
            ax.hist(pnl_values, bins=30, alpha=0.7, color='green', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='盈亏平衡线')
            ax.set_title('交易盈亏分布', fontsize=16, fontweight='bold')
            ax.set_xlabel('盈亏 (USDT)', fontsize=12)
            ax.set_ylabel('频次', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            chart_path = self.output_dir / "charts" / f"{chart_prefix}_pnl_distribution.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['pnl_distribution'] = str(chart_path)
        
        # 3. 月度收益热力图
        if trades_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 按月份聚合收益
            monthly_returns = {}
            for trade in trades_data:
                if 'timestamp' in trade:
                    date = datetime.fromisoformat(trade['timestamp']) if isinstance(trade['timestamp'], str) else trade['timestamp']
                    month_key = date.strftime('%Y-%m')
                    if month_key not in monthly_returns:
                        monthly_returns[month_key] = []
                    monthly_returns[month_key].append(trade.get('pnl', 0))
            
            # 计算月度收益率
            months = sorted(monthly_returns.keys())
            monthly_pnl = [sum(monthly_returns[month]) for month in months]
            
            if len(months) > 1:
                # 创建热力图数据
                data_for_heatmap = []
                for i, month in enumerate(months):
                    year, month_num = month.split('-')
                    data_for_heatmap.append([int(year), int(month_num), monthly_pnl[i]])
                
                df = pd.DataFrame(data_for_heatmap, columns=['Year', 'Month', 'PnL'])
                pivot_df = df.pivot(index='Year', columns='Month', values='PnL')
                
                sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                           center=0, ax=ax, cbar_kws={'label': '月度盈亏 (USDT)'})
                ax.set_title('月度收益热力图', fontsize=16, fontweight='bold')
                ax.set_xlabel('月份', fontsize=12)
                ax.set_ylabel('年份', fontsize=12)
            
            plt.tight_layout()
            chart_path = self.output_dir / "charts" / f"{chart_prefix}_monthly_heatmap.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['monthly_heatmap'] = str(chart_path)
        
        # 4. 回撤分析
        if account_data:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            dates = [datetime.fromisoformat(data['date']) if isinstance(data['date'], str) 
                    else data['date'] for data in account_data]
            values = [data['total_value'] for data in account_data]
            
            # 计算回撤
            peak_values = []
            drawdowns = []
            current_peak = values[0]
            
            for value in values:
                if value > current_peak:
                    current_peak = value
                peak_values.append(current_peak)
                drawdown = (current_peak - value) / current_peak
                drawdowns.append(drawdown)
            
            # 账户价值和峰值
            ax1.plot(dates, values, linewidth=2, color='blue', label='账户价值')
            ax1.plot(dates, peak_values, linewidth=2, color='red', alpha=0.7, label='历史峰值')
            ax1.set_title('账户价值与历史峰值', fontsize=14, fontweight='bold')
            ax1.set_ylabel('价值 (USDT)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # 回撤曲线
            ax2.fill_between(dates, 0, [-d for d in drawdowns], alpha=0.3, color='red', label='回撤')
            ax2.plot(dates, [-d for d in drawdowns], linewidth=2, color='red')
            ax2.set_title('回撤分析', fontsize=14, fontweight='bold')
            ax2.set_xlabel('日期', fontsize=12)
            ax2.set_ylabel('回撤比例', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 格式化x轴
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            chart_path = self.output_dir / "charts" / f"{chart_prefix}_drawdown_analysis.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            charts['drawdown_analysis'] = str(chart_path)
        
        self.logger.info(f"生成了 {len(charts)} 个图表")
        return charts
    
    def generate_html_report(self, 
                           metrics: TradingMetrics,
                           trades_data: List[Dict],
                           account_data: List[Dict],
                           charts: Dict[str, str],
                           title: str = "交易报告") -> str:
        """
        生成HTML报告
        
        Args:
            metrics: 交易指标
            trades_data: 交易数据
            account_data: 账户数据
            charts: 图表文件路径字典
            title: 报告标题
            
        Returns:
            str: HTML报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = self.output_dir / "html" / f"{title}_{timestamp}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    line-height: 1.6;
                    color: #333;
                }}
                .header {{
                    text-align: center;
                    border-bottom: 2px solid #4CAF50;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background: #f9f9f9;
                    padding: 20px;
                    border-radius: 8px;
                    border-left: 4px solid #4CAF50;
                }}
                .metric-card h3 {{
                    margin-top: 0;
                    color: #4CAF50;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2196F3;
                }}
                .charts-section {{
                    margin-top: 30px;
                }}
                .chart-container {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                .trades-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                .trades-table th, .trades-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .trades-table th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                .trades-table tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .positive {{
                    color: #4CAF50;
                }}
                .negative {{
                    color: #f44336;
                }}
                .summary {{
                    background: #e8f5e8;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="summary">
                <h2>📊 报告摘要</h2>
                <p>本报告分析了 {len(trades_data)} 笔交易，时间跨度为 {len(account_data)} 个交易日。</p>
                <p>总收益率: <span class="{'positive' if metrics.total_return >= 0 else 'negative'}">{metrics.total_return:.2%}</span></p>
                <p>胜率: <span class="metric-value">{metrics.win_rate:.1%}</span></p>
                <p>最大回撤: <span class="negative">{metrics.max_drawdown:.2%}</span></p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <h3>交易统计</h3>
                    <div class="metric-value">{metrics.total_trades}</div>
                    <p>总交易次数</p>
                    <p>盈利: {metrics.winning_trades} | 亏损: {metrics.losing_trades}</p>
                </div>
                
                <div class="metric-card">
                    <h3>收益指标</h3>
                    <div class="metric-value">{metrics.total_return:.2%}</div>
                    <p>总收益率</p>
                    <p>平均每笔收益: {metrics.avg_return_per_trade:.2f} USDT</p>
                </div>
                
                <div class="metric-card">
                    <h3>风险指标</h3>
                    <div class="metric-value">{metrics.max_drawdown:.2%}</div>
                    <p>最大回撤</p>
                    <p>波动率: {metrics.volatility:.2%}</p>
                </div>
                
                <div class="metric-card">
                    <h3>夏普比率</h3>
                    <div class="metric-value">{metrics.sharpe_ratio:.3f}</div>
                    <p>风险调整后收益</p>
                    <p>索提诺比率: {metrics.sortino_ratio:.3f}</p>
                </div>
                
                <div class="metric-card">
                    <h3>连续表现</h3>
                    <div class="metric-value">{metrics.max_consecutive_wins}</div>
                    <p>最大连续盈利</p>
                    <p>最大连续亏损: {metrics.max_consecutive_losses}</p>
                </div>
                
                <div class="metric-card">
                    <h3>盈利因子</h3>
                    <div class="metric-value">{metrics.profit_factor:.2f}</div>
                    <p>盈利/亏损比率</p>
                    <p>数学期望: {metrics.expectancy:.2f}</p>
                </div>
            </div>
        """
        
        # 添加图表
        html_content += '<div class="charts-section"><h2>📈 图表分析</h2>'
        
        chart_titles = {
            'account_value': '账户价值曲线',
            'pnl_distribution': '盈亏分布',
            'monthly_heatmap': '月度收益热力图',
            'drawdown_analysis': '回撤分析'
        }
        
        for chart_key, chart_path in charts.items():
            if os.path.exists(chart_path):
                # 使用相对路径
                relative_path = os.path.relpath(chart_path, html_path.parent)
                html_content += f"""
                <div class="chart-container">
                    <h3>{chart_titles.get(chart_key, chart_key)}</h3>
                    <img src="{relative_path}" alt="{chart_titles.get(chart_key, chart_key)}">
                </div>
                """
        
        html_content += '</div>'
        
        # 添加交易明细表
        if trades_data:
            html_content += """
            <div class="trades-section">
                <h2>💼 交易明细</h2>
                <table class="trades-table">
                    <thead>
                        <tr>
                            <th>时间</th>
                            <th>交易对</th>
                            <th>方向</th>
                            <th>数量</th>
                            <th>价格</th>
                            <th>盈亏</th>
                            <th>策略</th>
                        </tr>
                    </thead>
                    <tbody>
            """
            
            # 显示最近50笔交易
            recent_trades = trades_data[-50:] if len(trades_data) > 50 else trades_data
            
            for trade in recent_trades:
                pnl = trade.get('pnl', 0)
                pnl_class = 'positive' if pnl >= 0 else 'negative'
                
                html_content += f"""
                <tr>
                    <td>{trade.get('timestamp', '')}</td>
                    <td>{trade.get('symbol', '')}</td>
                    <td>{trade.get('side', '')}</td>
                    <td>{trade.get('quantity', 0):.6f}</td>
                    <td>{trade.get('price', 0):.2f}</td>
                    <td class="{pnl_class}">{pnl:.2f}</td>
                    <td>{trade.get('strategy', '')}</td>
                </tr>
                """
            
            html_content += """
                    </tbody>
                </table>
            </div>
            """
        
        html_content += """
            <div class="footer" style="text-align: center; margin-top: 40px; color: #666;">
                <p>由量化交易系统自动生成 | 生成时间: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
            </div>
        </body>
        </html>
        """
        
        # 写入HTML文件
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML报告已生成: {html_path}")
        return str(html_path)
    
    def generate_pdf_report(self, 
                          metrics: TradingMetrics,
                          trades_data: List[Dict],
                          account_data: List[Dict],
                          charts: Dict[str, str],
                          title: str = "交易报告") -> str:
        """
        生成PDF报告
        
        Args:
            metrics: 交易指标
            trades_data: 交易数据
            account_data: 账户数据
            charts: 图表文件路径字典
            title: 报告标题
            
        Returns:
            str: PDF报告文件路径
        """
        if not PDF_AVAILABLE:
            self.logger.warning("PDF生成库未安装，跳过PDF报告生成")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = self.output_dir / "pdf" / f"{title}_{timestamp}.pdf"
        
        # 创建PDF文档
        doc = SimpleDocTemplate(str(pdf_path), pagesize=A4)
        story = []
        
        # 获取样式
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2196F3')
        )
        
        # 标题
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # 摘要
        summary_data = [
            ['指标', '数值'],
            ['总交易次数', str(metrics.total_trades)],
            ['胜率', f"{metrics.win_rate:.1%}"],
            ['总收益率', f"{metrics.total_return:.2%}"],
            ['最大回撤', f"{metrics.max_drawdown:.2%}"],
            ['夏普比率', f"{metrics.sharpe_ratio:.3f}"],
            ['索提诺比率', f"{metrics.sortino_ratio:.3f}"],
            ['盈利因子', f"{metrics.profit_factor:.2f}"]
        ]
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("交易指标摘要", styles['Heading2']))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # 添加图表
        story.append(Paragraph("图表分析", styles['Heading2']))
        
        for chart_key, chart_path in charts.items():
            if os.path.exists(chart_path):
                try:
                    # 调整图片大小
                    img = Image(chart_path)
                    img.drawHeight = 4 * inch
                    img.drawWidth = 6 * inch
                    story.append(img)
                    story.append(Spacer(1, 12))
                except Exception as e:
                    self.logger.warning(f"添加图表 {chart_key} 失败: {e}")
        
        # 生成PDF
        try:
            doc.build(story)
            self.logger.info(f"PDF报告已生成: {pdf_path}")
            return str(pdf_path)
        except Exception as e:
            self.logger.error(f"PDF生成失败: {e}")
            return ""
    
    def generate_json_report(self, 
                           metrics: TradingMetrics,
                           trades_data: List[Dict],
                           account_data: List[Dict],
                           title: str = "交易报告") -> str:
        """
        生成JSON格式报告
        
        Args:
            metrics: 交易指标
            trades_data: 交易数据
            account_data: 账户数据
            title: 报告标题
            
        Returns:
            str: JSON报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = self.output_dir / "data" / f"{title}_{timestamp}.json"
        
        report_data = {
            'title': title,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.to_dict(),
            'trades_summary': {
                'total_trades': len(trades_data),
                'date_range': {
                    'start': trades_data[0].get('timestamp') if trades_data else None,
                    'end': trades_data[-1].get('timestamp') if trades_data else None
                }
            },
            'account_summary': {
                'total_days': len(account_data),
                'initial_value': account_data[0].get('total_value') if account_data else 0,
                'final_value': account_data[-1].get('total_value') if account_data else 0
            }
        }
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"JSON报告已生成: {json_path}")
        return str(json_path)
    
    def generate_complete_report(self, 
                               trades_data: List[Dict],
                               account_data: List[Dict],
                               title: str = "交易报告",
                               include_pdf: bool = True,
                               include_html: bool = True,
                               include_json: bool = True) -> Dict[str, str]:
        """
        生成完整报告（包括所有格式）
        
        Args:
            trades_data: 交易数据
            account_data: 账户数据
            title: 报告标题
            include_pdf: 是否包含PDF
            include_html: 是否包含HTML
            include_json: 是否包含JSON
            
        Returns:
            Dict[str, str]: 生成的报告文件路径字典
        """
        self.logger.info(f"开始生成完整报告: {title}")
        
        # 计算交易指标
        metrics = self.calculate_trading_metrics(trades_data, account_data)
        
        # 生成图表
        charts = self.generate_charts(trades_data, account_data, title)
        
        # 生成各种格式报告
        report_paths = {}
        
        if include_html:
            html_path = self.generate_html_report(metrics, trades_data, account_data, charts, title)
            if html_path:
                report_paths['html'] = html_path
        
        if include_pdf:
            pdf_path = self.generate_pdf_report(metrics, trades_data, account_data, charts, title)
            if pdf_path:
                report_paths['pdf'] = pdf_path
        
        if include_json:
            json_path = self.generate_json_report(metrics, trades_data, account_data, title)
            if json_path:
                report_paths['json'] = json_path
        
        # 图表路径
        report_paths['charts'] = charts
        
        self.logger.info(f"报告生成完成: {len(report_paths)} 个文件")
        return report_paths
    
    def create_sample_data(self) -> Tuple[List[Dict], List[Dict]]:
        """
        创建示例数据用于测试
        
        Returns:
            Tuple[List[Dict], List[Dict]]: 交易数据和账户数据
        """
        # 生成示例交易数据
        trades_data = []
        base_date = datetime.now() - timedelta(days=30)
        
        np.random.seed(42)
        
        for i in range(50):
            date = base_date + timedelta(days=i // 2)
            pnl = np.random.normal(5, 20)  # 平均盈利5，标准差20
            
            trade = {
                'timestamp': date.isoformat(),
                'symbol': np.random.choice(['BTCUSDT', 'ETHUSDT', 'BNBUSDT']),
                'side': np.random.choice(['BUY', 'SELL']),
                'quantity': np.random.uniform(0.01, 1.0),
                'price': np.random.uniform(20000, 50000),
                'pnl': pnl,
                'strategy': np.random.choice(['mean_reversion', 'momentum', 'arbitrage'])
            }
            trades_data.append(trade)
        
        # 生成示例账户数据
        account_data = []
        initial_value = 10000
        current_value = initial_value
        
        for i in range(30):
            date = base_date + timedelta(days=i)
            # 添加一些随机波动
            daily_change = np.random.normal(0, 100)
            current_value += daily_change
            
            account_data.append({
                'date': date.isoformat(),
                'total_value': current_value,
                'pnl': current_value - initial_value,
                'return': (current_value - initial_value) / initial_value
            })
        
        return trades_data, account_data