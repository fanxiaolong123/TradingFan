#!/usr/bin/env python3
"""
专业化回测报告生成器

这个模块提供了完整的回测结果报告生成功能：
- 多维度性能分析
- 交互式可视化图表
- 多币种、多时间框架对比
- 专业级HTML报告
- 详细的量化指标分析
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import base64
from io import BytesIO

# 可视化库
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
pio.templates.default = "plotly_white"


@dataclass
class BacktestSummary:
    """回测结果摘要"""
    strategy_name: str
    symbol: str
    timeframe: str
    period: str
    
    # 基础性能指标
    total_return: float
    annualized_return: float
    benchmark_return: float
    excess_return: float
    
    # 风险指标
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # 交易统计
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    
    # 风险测量
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    
    # 额外指标
    kelly_criterion: float
    information_ratio: float
    treynor_ratio: float
    
    # 时间信息
    start_date: str
    end_date: str
    total_days: int
    
    # 参数信息
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 评估结果
    performance_score: float = 0.0
    risk_score: float = 0.0
    overall_score: float = 0.0
    recommendation: str = ""


class ProfessionalReportGenerator:
    """专业化回测报告生成器"""
    
    def __init__(self, output_dir: str = "reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 创建子目录
        self.charts_dir = self.output_dir / "charts"
        self.data_dir = self.output_dir / "data"
        self.html_dir = self.output_dir / "html"
        
        for dir_path in [self.charts_dir, self.data_dir, self.html_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # 样式配置
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'secondary': '#6c757d'
        }
        
        print("✅ 专业化报告生成器初始化完成")
    
    def generate_comprehensive_report(self, 
                                    optimization_results: List[Dict], 
                                    backtest_data: Optional[Dict] = None,
                                    title: str = "量化策略回测分析报告") -> str:
        """
        生成综合性的回测报告
        
        Args:
            optimization_results: 优化结果列表
            backtest_data: 回测详细数据
            title: 报告标题
            
        Returns:
            str: 报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"comprehensive_report_{timestamp}"
        
        print(f"📊 生成综合回测报告: {report_name}")
        
        # 1. 数据预处理
        processed_data = self._process_optimization_results(optimization_results)
        
        # 2. 生成各类图表
        charts = self._generate_all_charts(processed_data, report_name)
        
        # 3. 计算分析指标
        analytics = self._calculate_analytics(processed_data)
        
        # 4. 生成HTML报告
        html_report = self._generate_html_report(
            processed_data, charts, analytics, title, report_name
        )
        
        # 5. 保存报告
        report_path = self.html_dir / f"{report_name}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"✅ 报告已生成: {report_path}")
        return str(report_path)
    
    def _process_optimization_results(self, results: List[Dict]) -> pd.DataFrame:
        """处理优化结果数据"""
        if not results:
            return pd.DataFrame()
        
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 数据清洗和转换
        if 'params' in df.columns:
            # 解析参数字典
            params_df = pd.json_normalize(df['params'])
            df = pd.concat([df.drop('params', axis=1), params_df], axis=1)
        
        # 数据类型转换和清洗
        numeric_columns = ['total_return', 'annualized_return', 'volatility', 'sharpe_ratio', 
                          'sortino_ratio', 'max_drawdown', 'win_rate', 'profit_factor',
                          'var_95', 'cvar_95', 'calmar_ratio', 'total_trades', 'avg_win', 'avg_loss']
        
        for col in numeric_columns:
            if col in df.columns:
                # 转换为数值类型，无法转换的设为NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # 填充NaN值
                if col in ['total_trades']:
                    df[col] = df[col].fillna(0).astype(int)
                else:
                    df[col] = df[col].fillna(0.0)
        
        # 计算百分比形式的指标
        for col in ['total_return', 'annualized_return', 'max_drawdown', 'win_rate']:
            if col in df.columns:
                df[f'{col}_pct'] = df[col] * 100
        
        # 计算排名（只对数值列进行排名）
        if 'sharpe_ratio' in df.columns and df['sharpe_ratio'].notna().sum() > 0:
            df['performance_rank'] = df['sharpe_ratio'].rank(ascending=False)
        else:
            df['performance_rank'] = 1
            
        if 'annualized_return' in df.columns and df['annualized_return'].notna().sum() > 0:
            df['return_rank'] = df['annualized_return'].rank(ascending=False)
        else:
            df['return_rank'] = 1
            
        if 'max_drawdown' in df.columns and df['max_drawdown'].notna().sum() > 0:
            df['risk_rank'] = df['max_drawdown'].rank(ascending=True)
        else:
            df['risk_rank'] = 1
        
        # 添加评级
        df['rating'] = df.apply(self._calculate_rating, axis=1)
        
        return df
    
    def _calculate_rating(self, row: pd.Series) -> str:
        """计算策略评级"""
        score = 0
        
        # 夏普比率评分
        sharpe_ratio = row.get('sharpe_ratio', 0)
        if pd.notna(sharpe_ratio):
            if sharpe_ratio >= 2.0:
                score += 30
            elif sharpe_ratio >= 1.5:
                score += 25
            elif sharpe_ratio >= 1.0:
                score += 20
            elif sharpe_ratio >= 0.5:
                score += 10
        
        # 年化收益率评分
        annualized_return = row.get('annualized_return', 0)
        if pd.notna(annualized_return):
            if annualized_return >= 0.5:
                score += 25
            elif annualized_return >= 0.3:
                score += 20
            elif annualized_return >= 0.2:
                score += 15
            elif annualized_return >= 0.1:
                score += 10
        
        # 最大回撤评分
        max_drawdown = row.get('max_drawdown', 0)
        if pd.notna(max_drawdown):
            if max_drawdown >= -0.1:
                score += 25
            elif max_drawdown >= -0.15:
                score += 20
            elif max_drawdown >= -0.2:
                score += 15
            elif max_drawdown >= -0.3:
                score += 10
        
        # 胜率评分
        win_rate = row.get('win_rate', 0)
        if pd.notna(win_rate):
            if win_rate >= 0.7:
                score += 20
            elif win_rate >= 0.6:
                score += 15
            elif win_rate >= 0.5:
                score += 10
            elif win_rate >= 0.4:
                score += 5
        
        # 评级映射
        if score >= 80:
            return "A+"
        elif score >= 70:
            return "A"
        elif score >= 60:
            return "A-"
        elif score >= 50:
            return "B+"
        elif score >= 40:
            return "B"
        elif score >= 30:
            return "B-"
        elif score >= 20:
            return "C"
        else:
            return "D"
    
    def _generate_all_charts(self, data: pd.DataFrame, report_name: str) -> Dict[str, str]:
        """生成所有图表"""
        charts = {}
        
        if data.empty:
            return charts
        
        print("📈 生成可视化图表...")
        
        # 1. 性能排行榜
        charts['performance_ranking'] = self._create_performance_ranking_chart(data, report_name)
        
        # 2. 风险收益散点图
        charts['risk_return_scatter'] = self._create_risk_return_scatter(data, report_name)
        
        # 3. 参数敏感性分析
        charts['parameter_sensitivity'] = self._create_parameter_sensitivity_chart(data, report_name)
        
        # 4. 多维度雷达图
        charts['radar_chart'] = self._create_radar_chart(data, report_name)
        
        # 5. 时间序列分析
        charts['time_series'] = self._create_time_series_chart(data, report_name)
        
        # 6. 统计分布图
        charts['distribution'] = self._create_distribution_charts(data, report_name)
        
        # 7. 相关性热力图
        charts['correlation'] = self._create_correlation_heatmap(data, report_name)
        
        # 8. 交易统计图
        charts['trade_stats'] = self._create_trade_statistics_chart(data, report_name)
        
        return charts
    
    def _create_performance_ranking_chart(self, data: pd.DataFrame, report_name: str) -> str:
        """创建性能排行榜图表"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('夏普比率排行', '年化收益率排行', '最大回撤排行', '胜率排行'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        # 选择前10名
        top_10 = data.nlargest(10, 'sharpe_ratio')
        
        # 夏普比率
        fig.add_trace(
            go.Bar(
                x=top_10.index,
                y=top_10['sharpe_ratio'],
                name='夏普比率',
                marker_color=self.colors['primary']
            ),
            row=1, col=1
        )
        
        # 年化收益率
        fig.add_trace(
            go.Bar(
                x=top_10.index,
                y=top_10['annualized_return_pct'],
                name='年化收益率(%)',
                marker_color=self.colors['success']
            ),
            row=1, col=2
        )
        
        # 最大回撤
        fig.add_trace(
            go.Bar(
                x=top_10.index,
                y=top_10['max_drawdown_pct'],
                name='最大回撤(%)',
                marker_color=self.colors['danger']
            ),
            row=2, col=1
        )
        
        # 胜率
        fig.add_trace(
            go.Bar(
                x=top_10.index,
                y=top_10['win_rate_pct'],
                name='胜率(%)',
                marker_color=self.colors['warning']
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="策略性能排行榜",
            height=800,
            showlegend=False
        )
        
        # 保存图表
        chart_path = self.charts_dir / f"{report_name}_performance_ranking.html"
        fig.write_html(chart_path)
        
        return str(chart_path)
    
    def _create_risk_return_scatter(self, data: pd.DataFrame, report_name: str) -> str:
        """创建风险收益散点图"""
        fig = go.Figure()
        
        # 按评级分组
        for rating in data['rating'].unique():
            rating_data = data[data['rating'] == rating]
            
            # 确保标记大小为正数
            marker_sizes = np.abs(rating_data['sharpe_ratio']) * 3 + 5  # 加5确保最小尺寸
            marker_sizes = np.clip(marker_sizes, 5, 50)  # 限制尺寸范围
            
            fig.add_trace(go.Scatter(
                x=rating_data['max_drawdown_pct'],
                y=rating_data['annualized_return_pct'],
                mode='markers',
                name=f'评级 {rating}',
                marker=dict(
                    size=marker_sizes,
                    opacity=0.7,
                    line=dict(width=2)
                ),
                text=rating_data.index,
                hovertemplate='<b>策略 %{text}</b><br>' +
                              '年化收益率: %{y:.2f}%<br>' +
                              '最大回撤: %{x:.2f}%<br>' +
                              '夏普比率: %{customdata:.2f}<br>' +
                              '评级: ' + rating + '<extra></extra>',
                customdata=rating_data['sharpe_ratio']
            ))
        
        fig.update_layout(
            title='风险收益散点图 (气泡大小表示夏普比率)',
            xaxis_title='最大回撤 (%)',
            yaxis_title='年化收益率 (%)',
            height=600,
            hovermode='closest'
        )
        
        # 添加基准线
        fig.add_hline(y=20, line_dash="dash", line_color="gray", 
                     annotation_text="收益率基准线 (20%)")
        fig.add_vline(x=-20, line_dash="dash", line_color="gray", 
                     annotation_text="回撤基准线 (-20%)")
        
        # 保存图表
        chart_path = self.charts_dir / f"{report_name}_risk_return_scatter.html"
        fig.write_html(chart_path)
        
        return str(chart_path)
    
    def _create_parameter_sensitivity_chart(self, data: pd.DataFrame, report_name: str) -> str:
        """创建参数敏感性分析图表"""
        # 寻找数值型参数列
        numeric_params = []
        excluded_cols = [
            'total_return', 'annualized_return', 'volatility', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'total_trades', 'performance_rank',
            'total_return_pct', 'annualized_return_pct', 'max_drawdown_pct', 
            'win_rate_pct', 'return_rank', 'risk_rank', 'rating', 'strategy_name',
            'symbol', 'timeframe', 'sortino_ratio', 'profit_factor', 'var_95',
            'cvar_95', 'calmar_ratio', 'avg_win', 'avg_loss'
        ]
        
        for col in data.columns:
            if col not in excluded_cols and pd.api.types.is_numeric_dtype(data[col]):
                numeric_params.append(col)
        
        if not numeric_params:
            return ""
        
        # 选择前4个参数
        params_to_plot = numeric_params[:4]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{param} 敏感性分析' for param in params_to_plot]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, param in enumerate(params_to_plot):
            if i >= 4:
                break
            
            row, col = positions[i]
            
            # 按参数值分组计算平均夏普比率
            param_groups = data.groupby(param)['sharpe_ratio'].mean().sort_index()
            
            fig.add_trace(
                go.Scatter(
                    x=param_groups.index,
                    y=param_groups.values,
                    mode='lines+markers',
                    name=param,
                    line=dict(width=3),
                    marker=dict(size=8)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title_text="参数敏感性分析",
            height=800,
            showlegend=False
        )
        
        # 保存图表
        chart_path = self.charts_dir / f"{report_name}_parameter_sensitivity.html"
        fig.write_html(chart_path)
        
        return str(chart_path)
    
    def _create_radar_chart(self, data: pd.DataFrame, report_name: str) -> str:
        """创建多维度雷达图"""
        # 选择前5个策略
        top_5 = data.nlargest(5, 'sharpe_ratio')
        
        # 标准化指标 (0-10分制)
        metrics = ['sharpe_ratio', 'annualized_return', 'win_rate', 'total_trades']
        
        fig = go.Figure()
        
        for idx, (_, row) in enumerate(top_5.iterrows()):
            # 计算标准化分数
            scores = []
            for metric in metrics:
                value = row.get(metric, 0)
                if pd.isna(value):
                    score = 0
                elif metric == 'sharpe_ratio':
                    score = min(max(value, 0) / 3 * 10, 10)
                elif metric == 'annualized_return':
                    score = min(max(value, 0) * 10, 10)
                elif metric == 'win_rate':
                    score = max(value, 0) * 10
                elif metric == 'total_trades':
                    score = min(max(value, 0) / 20 * 10, 10)
                else:
                    score = min(abs(value) * 10, 10)
                
                scores.append(max(0, score))
            
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],  # 闭合
                theta=metrics + [metrics[0]],  # 闭合
                fill='toself',
                name=f'策略 {idx + 1}',
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="TOP 5 策略多维度对比雷达图",
            height=600
        )
        
        # 保存图表
        chart_path = self.charts_dir / f"{report_name}_radar_chart.html"
        fig.write_html(chart_path)
        
        return str(chart_path)
    
    def _create_time_series_chart(self, data: pd.DataFrame, report_name: str) -> str:
        """创建时间序列分析图表 (模拟)"""
        # 由于没有真实的时间序列数据，我们创建一个示例
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        fig = go.Figure()
        
        # 选择前3个策略模拟权益曲线
        top_3 = data.nlargest(3, 'sharpe_ratio')
        
        for idx, (_, row) in enumerate(top_3.iterrows()):
            # 模拟权益曲线
            np.random.seed(42 + idx)
            daily_returns = np.random.normal(
                row['annualized_return'] / 365, 
                row['volatility'] / np.sqrt(365), 
                len(dates)
            )
            
            equity_curve = 100000 * (1 + daily_returns).cumprod()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=equity_curve,
                mode='lines',
                name=f'策略 {idx + 1} (夏普比率: {row["sharpe_ratio"]:.2f})',
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='策略权益曲线对比 (模拟)',
            xaxis_title='日期',
            yaxis_title='账户价值 (USDT)',
            height=500,
            hovermode='x unified'
        )
        
        # 保存图表
        chart_path = self.charts_dir / f"{report_name}_time_series.html"
        fig.write_html(chart_path)
        
        return str(chart_path)
    
    def _create_distribution_charts(self, data: pd.DataFrame, report_name: str) -> str:
        """创建统计分布图表"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('夏普比率分布', '年化收益率分布', '最大回撤分布', '胜率分布')
        )
        
        # 夏普比率分布
        fig.add_trace(
            go.Histogram(x=data['sharpe_ratio'], nbinsx=20, name='夏普比率'),
            row=1, col=1
        )
        
        # 年化收益率分布
        fig.add_trace(
            go.Histogram(x=data['annualized_return_pct'], nbinsx=20, name='年化收益率'),
            row=1, col=2
        )
        
        # 最大回撤分布
        fig.add_trace(
            go.Histogram(x=data['max_drawdown_pct'], nbinsx=20, name='最大回撤'),
            row=2, col=1
        )
        
        # 胜率分布
        fig.add_trace(
            go.Histogram(x=data['win_rate_pct'], nbinsx=20, name='胜率'),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="关键指标统计分布",
            height=800,
            showlegend=False
        )
        
        # 保存图表
        chart_path = self.charts_dir / f"{report_name}_distribution.html"
        fig.write_html(chart_path)
        
        return str(chart_path)
    
    def _create_correlation_heatmap(self, data: pd.DataFrame, report_name: str) -> str:
        """创建相关性热力图"""
        # 选择数值型列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_matrix = data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdYlBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate="%{text:.2f}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='参数相关性热力图',
            height=600,
            width=800
        )
        
        # 保存图表
        chart_path = self.charts_dir / f"{report_name}_correlation.html"
        fig.write_html(chart_path)
        
        return str(chart_path)
    
    def _create_trade_statistics_chart(self, data: pd.DataFrame, report_name: str) -> str:
        """创建交易统计图表"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('交易次数vs收益率', '胜率vs夏普比率', '交易次数分布', '盈亏比分布')
        )
        
        # 交易次数vs收益率
        fig.add_trace(
            go.Scatter(
                x=data['total_trades'],
                y=data['annualized_return_pct'],
                mode='markers',
                name='交易次数vs收益率',
                marker=dict(size=8, opacity=0.6)
            ),
            row=1, col=1
        )
        
        # 胜率vs夏普比率
        fig.add_trace(
            go.Scatter(
                x=data['win_rate_pct'],
                y=data['sharpe_ratio'],
                mode='markers',
                name='胜率vs夏普比率',
                marker=dict(size=8, opacity=0.6)
            ),
            row=1, col=2
        )
        
        # 交易次数分布
        fig.add_trace(
            go.Histogram(x=data['total_trades'], nbinsx=15, name='交易次数'),
            row=2, col=1
        )
        
        # 盈亏比分布
        if 'profit_factor' in data.columns:
            fig.add_trace(
                go.Histogram(x=data['profit_factor'], nbinsx=15, name='盈亏比'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="交易统计分析",
            height=800,
            showlegend=False
        )
        
        # 保存图表
        chart_path = self.charts_dir / f"{report_name}_trade_stats.html"
        fig.write_html(chart_path)
        
        return str(chart_path)
    
    def _calculate_analytics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """计算分析指标"""
        if data.empty:
            return {}
        
        analytics = {
            'total_strategies': len(data),
            'avg_sharpe_ratio': data['sharpe_ratio'].mean(),
            'max_sharpe_ratio': data['sharpe_ratio'].max(),
            'min_sharpe_ratio': data['sharpe_ratio'].min(),
            'avg_return': data['annualized_return'].mean(),
            'max_return': data['annualized_return'].max(),
            'avg_drawdown': data['max_drawdown'].mean(),
            'best_drawdown': data['max_drawdown'].max(),
            'avg_win_rate': data['win_rate'].mean(),
            'strategies_above_1_sharpe': len(data[data['sharpe_ratio'] > 1]),
            'strategies_above_30_return': len(data[data['annualized_return'] > 0.3]),
            'strategies_below_20_drawdown': len(data[data['max_drawdown'] > -0.2]),
            'top_strategy_index': data['sharpe_ratio'].idxmax(),
            'rating_distribution': data['rating'].value_counts().to_dict()
        }
        
        return analytics
    
    def _generate_html_report(self, 
                            data: pd.DataFrame, 
                            charts: Dict[str, str], 
                            analytics: Dict[str, Any],
                            title: str,
                            report_name: str) -> str:
        """生成HTML报告"""
        
        # 读取图表HTML内容
        chart_contents = {}
        for chart_name, chart_path in charts.items():
            if os.path.exists(chart_path):
                with open(chart_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 提取Plotly图表的div部分
                    start = content.find('<div id=')
                    end = content.find('</script>', start) + 9
                    if start != -1 and end != -1:
                        chart_contents[chart_name] = content[start:end]
        
        # 生成汇总表格
        if not data.empty:
            top_10 = data.nlargest(10, 'sharpe_ratio')
            summary_table = self._generate_summary_table(top_10)
        else:
            summary_table = "<p>暂无数据</p>"
        
        # HTML模板
        html_template = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            transition: transform 0.2s;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            font-size: 0.9rem;
        }}
        .chart-container {{
            background: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
        .table-responsive {{
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .rating-A\\2B {{
            background-color: #28a745;
            color: white;
        }}
        .rating-A {{
            background-color: #20c997;
            color: white;
        }}
        .rating-B {{
            background-color: #ffc107;
            color: black;
        }}
        .rating-C {{
            background-color: #fd7e14;
            color: white;
        }}
        .rating-D {{
            background-color: #dc3545;
            color: white;
        }}
        .nav-tabs .nav-link.active {{
            background-color: #667eea;
            border-color: #667eea;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1 class="text-center">{title}</h1>
            <p class="text-center mb-0">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </div>

    <div class="container">
        <!-- 关键指标概览 -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value">{analytics.get('total_strategies', 0)}</div>
                    <div class="metric-label">总策略数</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value">{analytics.get('max_sharpe_ratio', 0):.2f}</div>
                    <div class="metric-label">最高夏普比率</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value">{analytics.get('max_return', 0)*100:.1f}%</div>
                    <div class="metric-label">最高年化收益</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card text-center">
                    <div class="metric-value">{analytics.get('strategies_above_1_sharpe', 0)}</div>
                    <div class="metric-label">夏普比率>1的策略</div>
                </div>
            </div>
        </div>

        <!-- 策略评级分布 -->
        <div class="row mb-4">
            <div class="col-md-12">
                <div class="metric-card">
                    <h5>策略评级分布</h5>
                    <div class="row">
"""
        
        # 添加评级分布
        if 'rating_distribution' in analytics:
            for rating, count in analytics['rating_distribution'].items():
                html_template += f"""
                        <div class="col-md-2">
                            <div class="text-center p-2 rounded rating-{rating}">
                                <div class="fw-bold">{rating}</div>
                                <div class="small">{count} 个策略</div>
                            </div>
                        </div>
"""
        
        html_template += """
                    </div>
                </div>
            </div>
        </div>

        <!-- 图表标签页 -->
        <div class="row">
            <div class="col-md-12">
                <ul class="nav nav-tabs" id="chartTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="ranking-tab" data-bs-toggle="tab" data-bs-target="#ranking" type="button" role="tab">性能排行</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="risk-return-tab" data-bs-toggle="tab" data-bs-target="#risk-return" type="button" role="tab">风险收益</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="sensitivity-tab" data-bs-toggle="tab" data-bs-target="#sensitivity" type="button" role="tab">参数敏感性</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="radar-tab" data-bs-toggle="tab" data-bs-target="#radar" type="button" role="tab">雷达图</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="distribution-tab" data-bs-toggle="tab" data-bs-target="#distribution" type="button" role="tab">分布图</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="correlation-tab" data-bs-toggle="tab" data-bs-target="#correlation" type="button" role="tab">相关性</button>
                    </li>
                </ul>
                
                <div class="tab-content" id="chartTabsContent">
"""
        
        # 添加图表标签页内容
        chart_tabs = [
            ('ranking', 'performance_ranking', '性能排行榜'),
            ('risk-return', 'risk_return_scatter', '风险收益散点图'),
            ('sensitivity', 'parameter_sensitivity', '参数敏感性分析'),
            ('radar', 'radar_chart', '多维度雷达图'),
            ('distribution', 'distribution', '统计分布图'),
            ('correlation', 'correlation', '相关性热力图')
        ]
        
        for i, (tab_id, chart_key, chart_title) in enumerate(chart_tabs):
            active_class = 'active' if i == 0 else ''
            chart_content = chart_contents.get(chart_key, f'<p>图表加载失败: {chart_title}</p>')
            
            html_template += f"""
                    <div class="tab-pane fade {'show ' + active_class if i == 0 else ''}" id="{tab_id}" role="tabpanel">
                        <div class="chart-container">
                            <h5>{chart_title}</h5>
                            {chart_content}
                        </div>
                    </div>
"""
        
        html_template += f"""
                </div>
            </div>
        </div>

        <!-- 策略汇总表 -->
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="table-responsive">
                    <h5>TOP 10 策略详细信息</h5>
                    {summary_table}
                </div>
            </div>
        </div>

        <!-- 报告说明 -->
        <div class="row mt-4">
            <div class="col-md-12">
                <div class="metric-card">
                    <h5>报告说明</h5>
                    <ul>
                        <li><strong>夏普比率</strong>: 衡量每承担一单位风险所获得的超额收益</li>
                        <li><strong>年化收益率</strong>: 策略在一年内的预期收益率</li>
                        <li><strong>最大回撤</strong>: 策略在历史上最大的资产损失幅度</li>
                        <li><strong>胜率</strong>: 盈利交易次数占总交易次数的比例</li>
                        <li><strong>评级系统</strong>: A+最优秀, A优秀, B良好, C一般, D较差</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <footer class="text-center mt-5 mb-3">
        <p class="text-muted">© 2024 TradingFan 量化交易系统 - 专业回测报告</p>
    </footer>
</body>
</html>
"""
        
        return html_template
    
    def _generate_summary_table(self, data: pd.DataFrame) -> str:
        """生成汇总表格"""
        if data.empty:
            return "<p>暂无数据</p>"
        
        html = """
        <table class="table table-striped table-hover">
            <thead class="table-dark">
                <tr>
                    <th>排名</th>
                    <th>策略参数</th>
                    <th>评级</th>
                    <th>夏普比率</th>
                    <th>年化收益率</th>
                    <th>最大回撤</th>
                    <th>胜率</th>
                    <th>交易次数</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for i, (idx, row) in enumerate(data.iterrows(), 1):
            # 构建参数字符串
            params_str = ""
            excluded_cols = ['sharpe_ratio', 'annualized_return', 'max_drawdown', 
                           'win_rate', 'total_trades', 'rating', 'performance_rank',
                           'return_rank', 'risk_rank', 'total_return_pct', 
                           'annualized_return_pct', 'max_drawdown_pct', 'win_rate_pct',
                           'strategy_name', 'symbol', 'timeframe', 'total_return',
                           'volatility', 'sortino_ratio', 'profit_factor', 'var_95',
                           'cvar_95', 'calmar_ratio', 'avg_win', 'avg_loss']
            
            for col in data.columns:
                if col not in excluded_cols:
                    try:
                        value = row[col]
                        if pd.notna(value) and value != '':
                            params_str += f"{col}={value}, "
                    except:
                        continue
            
            params_str = params_str.rstrip(', ')
            
            # 安全获取数值
            sharpe_ratio = row.get('sharpe_ratio', 0)
            annualized_return = row.get('annualized_return', 0)
            max_drawdown = row.get('max_drawdown', 0)
            win_rate = row.get('win_rate', 0)
            total_trades = row.get('total_trades', 0)
            rating = row.get('rating', 'D')
            
            html += f"""
                <tr>
                    <td><span class="badge bg-primary">{i}</span></td>
                    <td class="small">{params_str}</td>
                    <td><span class="badge rating-{rating}">{rating}</span></td>
                    <td><strong>{sharpe_ratio:.3f}</strong></td>
                    <td class="text-success"><strong>{annualized_return*100:.2f}%</strong></td>
                    <td class="text-danger"><strong>{max_drawdown*100:.2f}%</strong></td>
                    <td>{win_rate*100:.1f}%</td>
                    <td>{int(total_trades)}</td>
                </tr>
            """
        
        html += """
            </tbody>
        </table>
        """
        
        return html
    
    def generate_simple_report(self, result_file: str, title: str = "策略优化报告") -> str:
        """
        从单个结果文件生成简单报告
        
        Args:
            result_file: 结果文件路径
            title: 报告标题
            
        Returns:
            str: 报告文件路径
        """
        # 读取结果文件
        if result_file.endswith('.csv'):
            data = pd.read_csv(result_file)
        elif result_file.endswith('.json'):
            with open(result_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                data = pd.DataFrame([json_data])
        else:
            raise ValueError("不支持的文件格式")
        
        # 转换为标准格式
        results = data.to_dict('records')
        
        return self.generate_comprehensive_report(results, title=title)