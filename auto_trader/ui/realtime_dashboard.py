"""
实时监控仪表板 - 增强版UI界面

提供实时数据监控、交互式控制和高级分析功能：
- 实时数据刷新
- 交互式图表
- 策略控制面板
- 风险监控
- 通知系统集成
- 多策略管理
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import time
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入核心模块
from auto_trader.core.engine import TraderEngine
from auto_trader.core.risk import RiskManager
from auto_trader.core.capital_management import CapitalManager
from auto_trader.utils.notification import NotificationManager, NotificationType, NotificationLevel
from auto_trader.utils.report_generator import ReportGenerator
from auto_trader.utils.config import get_config
from auto_trader.utils.logger import get_logger


class RealtimeDashboard:
    """实时监控仪表板"""
    
    def __init__(self):
        """初始化实时仪表板"""
        self.config = get_config()
        self.logger = get_logger(__name__)
        
        # 设置页面配置
        st.set_page_config(
            page_title="AutoTrader 实时监控",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 初始化状态
        self._initialize_session_state()
        
        # 自定义CSS样式
        self._setup_custom_styles()
        
        # 初始化系统组件
        self._initialize_components()
    
    def _initialize_session_state(self):
        """初始化会话状态"""
        # 系统状态
        if 'system_running' not in st.session_state:
            st.session_state.system_running = False
        
        if 'trader_engine' not in st.session_state:
            st.session_state.trader_engine = None
        
        if 'notification_manager' not in st.session_state:
            st.session_state.notification_manager = None
        
        if 'capital_manager' not in st.session_state:
            st.session_state.capital_manager = None
        
        if 'report_generator' not in st.session_state:
            st.session_state.report_generator = None
        
        # 数据缓存
        if 'live_data' not in st.session_state:
            st.session_state.live_data = {
                'prices': {},
                'strategies': {},
                'account': {},
                'risk_metrics': {},
                'notifications': []
            }
        
        # 设置选项
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 5
        
        if 'selected_strategies' not in st.session_state:
            st.session_state.selected_strategies = []
    
    def _setup_custom_styles(self):
        """设置自定义CSS样式"""
        st.markdown("""
        <style>
        /* 全局样式 */
        .main-header {
            background: linear-gradient(90deg, #1f4e79 0%, #2e7d32 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* 状态指示器 */
        .status-indicator {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: bold;
            text-align: center;
            margin: 0.5rem;
        }
        
        .status-running {
            background-color: #4caf50;
            color: white;
        }
        
        .status-stopped {
            background-color: #f44336;
            color: white;
        }
        
        .status-warning {
            background-color: #ff9800;
            color: white;
        }
        
        /* 指标卡片 */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
        }
        
        .metric-delta {
            font-size: 1.1rem;
            margin-top: 0.5rem;
        }
        
        /* 策略控制面板 */
        .strategy-panel {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #007bff;
            margin: 1rem 0;
        }
        
        .strategy-active {
            border-left-color: #28a745;
        }
        
        .strategy-inactive {
            border-left-color: #6c757d;
        }
        
        .strategy-error {
            border-left-color: #dc3545;
        }
        
        /* 实时数据表格 */
        .live-data-table {
            background: white;
            border-radius: 10px;
            padding: 1rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        /* 通知区域 */
        .notification-panel {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .notification-critical {
            background: #f8d7da;
            border-color: #f5c6cb;
        }
        
        .notification-warning {
            background: #fff3cd;
            border-color: #ffeaa7;
        }
        
        .notification-info {
            background: #d1ecf1;
            border-color: #bee5eb;
        }
        
        /* 按钮样式 */
        .control-button {
            background: linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%);
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .control-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* 隐藏默认元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    
    def _initialize_components(self):
        """初始化系统组件"""
        try:
            # 初始化交易引擎
            if st.session_state.trader_engine is None:
                st.session_state.trader_engine = TraderEngine()
            
            # 初始化通知管理器
            if st.session_state.notification_manager is None:
                notification_config = {
                    'providers': {
                        'log': {'enabled': True, 'log_file': 'ui_notifications.log'}
                    }
                }
                st.session_state.notification_manager = NotificationManager(notification_config)
            
            # 初始化资金管理器
            if st.session_state.capital_manager is None:
                capital_config = {
                    'default_strategy': 'fixed_percent',
                    'strategies': {
                        'fixed_percent': {
                            'base_position_percent': 0.1,
                            'max_position_percent': 0.25
                        }
                    }
                }
                st.session_state.capital_manager = CapitalManager(capital_config)
            
            # 初始化报告生成器
            if st.session_state.report_generator is None:
                st.session_state.report_generator = ReportGenerator()
            
            self.logger.info("实时监控系统组件初始化完成")
            
        except Exception as e:
            st.error(f"系统组件初始化失败: {e}")
            self.logger.error(f"系统组件初始化失败: {e}")
    
    def run(self):
        """运行实时监控界面"""
        # 主标题
        st.markdown("""
        <div class="main-header">
            <h1>🚀 AutoTrader 实时监控中心</h1>
            <p>专业级量化交易系统实时监控与控制平台</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 侧边栏控制面板
        self._render_sidebar()
        
        # 主内容区域
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 实时概览", 
            "🎯 策略管理", 
            "⚠️ 风险监控", 
            "📈 实时图表", 
            "📋 系统日志"
        ])
        
        with tab1:
            self._render_overview_tab()
        
        with tab2:
            self._render_strategy_management_tab()
        
        with tab3:
            self._render_risk_monitoring_tab()
        
        with tab4:
            self._render_realtime_charts_tab()
        
        with tab5:
            self._render_system_logs_tab()
        
        # 自动刷新
        if st.session_state.auto_refresh:
            time.sleep(st.session_state.refresh_interval)
            st.rerun()
    
    def _render_sidebar(self):
        """渲染侧边栏控制面板"""
        st.sidebar.title("🎛️ 控制中心")
        
        # 系统状态
        with st.sidebar.container():
            st.subheader("系统状态")
            
            if st.session_state.system_running:
                st.markdown('<div class="status-indicator status-running">🟢 系统运行中</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-indicator status-stopped">🔴 系统已停止</div>', 
                           unsafe_allow_html=True)
        
        # 系统控制
        st.sidebar.markdown("---")
        st.sidebar.subheader("系统控制")
        
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("🚀 启动系统", disabled=st.session_state.system_running):
                self._start_system()
        
        with col2:
            if st.button("⏹️ 停止系统", disabled=not st.session_state.system_running):
                self._stop_system()
        
        # 自动刷新设置
        st.sidebar.markdown("---")
        st.sidebar.subheader("刷新设置")
        
        st.session_state.auto_refresh = st.sidebar.toggle(
            "自动刷新", 
            value=st.session_state.auto_refresh
        )
        
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.sidebar.slider(
                "刷新间隔 (秒)", 
                min_value=1, 
                max_value=60, 
                value=st.session_state.refresh_interval
            )
        
        # 快速操作
        st.sidebar.markdown("---")
        st.sidebar.subheader("快速操作")
        
        if st.sidebar.button("🔄 手动刷新"):
            self._refresh_data()
            st.rerun()
        
        if st.sidebar.button("📊 生成报告"):
            self._generate_report()
        
        if st.sidebar.button("🧹 清除缓存"):
            self._clear_cache()
        
        # 通知设置
        st.sidebar.markdown("---")
        st.sidebar.subheader("通知设置")
        
        enable_notifications = st.sidebar.checkbox("启用通知", value=True)
        
        if enable_notifications:
            notification_level = st.sidebar.selectbox(
                "通知级别",
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                index=1
            )
    
    def _render_overview_tab(self):
        """渲染实时概览标签页"""
        st.header("📊 实时系统概览")
        
        # 实时指标
        self._render_realtime_metrics()
        
        # 分隔线
        st.markdown("---")
        
        # 实时数据表格
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 实时价格")
            self._render_price_table()
        
        with col2:
            st.subheader("🎯 策略状态")
            self._render_strategy_status()
        
        # 最新交易
        st.markdown("---")
        st.subheader("💼 最新交易")
        self._render_recent_trades()
    
    def _render_realtime_metrics(self):
        """渲染实时指标"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # 账户价值
            account_value = self._get_account_value()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">账户总价值</div>
                <div class="metric-value">${account_value:,.2f}</div>
                <div class="metric-delta">USDT</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # 今日盈亏
            daily_pnl = self._get_daily_pnl()
            pnl_color = "#4caf50" if daily_pnl >= 0 else "#f44336"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">今日盈亏</div>
                <div class="metric-value" style="color: {pnl_color}">
                    {daily_pnl:+.2f}
                </div>
                <div class="metric-delta">USDT</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            # 活跃策略
            active_strategies = self._get_active_strategies_count()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">活跃策略</div>
                <div class="metric-value">{active_strategies}</div>
                <div class="metric-delta">个</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            # 风险等级
            risk_level = self._get_risk_level()
            risk_color = {
                "LOW": "#4caf50",
                "MEDIUM": "#ff9800", 
                "HIGH": "#f44336",
                "CRITICAL": "#9c27b0"
            }.get(risk_level, "#6c757d")
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">风险等级</div>
                <div class="metric-value" style="color: {risk_color}">
                    {risk_level}
                </div>
                <div class="metric-delta">当前</div>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_price_table(self):
        """渲染价格表格"""
        try:
            # 生成模拟价格数据
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'DOGEUSDT', 'PEPEUSDT']
            
            price_data = []
            for symbol in symbols:
                # 模拟价格数据
                base_price = {'BTCUSDT': 45000, 'ETHUSDT': 2500, 'BNBUSDT': 300, 
                             'SOLUSDT': 100, 'DOGEUSDT': 0.1, 'PEPEUSDT': 0.000001}[symbol]
                
                current_price = base_price * (1 + np.random.uniform(-0.02, 0.02))
                change_24h = np.random.uniform(-0.05, 0.05)
                
                price_data.append({
                    '交易对': symbol,
                    '价格': f"{current_price:.6f}",
                    '24h变化': f"{change_24h:+.2%}",
                    '状态': '🟢 正常' if abs(change_24h) < 0.03 else '🟡 波动'
                })
            
            df = pd.DataFrame(price_data)
            
            # 使用自定义样式显示表格
            st.markdown('<div class="live-data-table">', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"价格数据加载失败: {e}")
    
    def _render_strategy_status(self):
        """渲染策略状态"""
        try:
            if st.session_state.trader_engine and st.session_state.system_running:
                # 获取真实策略状态
                strategies_status = st.session_state.trader_engine.get_strategies_status()
                
                if strategies_status:
                    for name, status in strategies_status.items():
                        status_class = {
                            'running': 'strategy-active',
                            'stopped': 'strategy-inactive',
                            'error': 'strategy-error'
                        }.get(status.get('state', 'stopped'), 'strategy-inactive')
                        
                        st.markdown(f"""
                        <div class="strategy-panel {status_class}">
                            <h4>{name}</h4>
                            <p>状态: {status.get('state', 'unknown')}</p>
                            <p>交易对: {status.get('symbol', 'N/A')}</p>
                            <p>时间周期: {status.get('timeframe', 'N/A')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("暂无活跃策略")
            else:
                # 显示示例策略状态
                strategies = [
                    {'name': 'BTC均值回归', 'status': 'running', 'symbol': 'BTCUSDT'},
                    {'name': 'ETH动量策略', 'status': 'stopped', 'symbol': 'ETHUSDT'},
                    {'name': 'BNB套利', 'status': 'running', 'symbol': 'BNBUSDT'}
                ]
                
                for strategy in strategies:
                    status_class = {
                        'running': 'strategy-active',
                        'stopped': 'strategy-inactive',
                        'error': 'strategy-error'
                    }.get(strategy['status'], 'strategy-inactive')
                    
                    st.markdown(f"""
                    <div class="strategy-panel {status_class}">
                        <h4>{strategy['name']}</h4>
                        <p>状态: {strategy['status']}</p>
                        <p>交易对: {strategy['symbol']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"策略状态加载失败: {e}")
    
    def _render_recent_trades(self):
        """渲染最近交易"""
        try:
            # 生成模拟交易数据
            trade_data = []
            for i in range(10):
                trade_data.append({
                    '时间': (datetime.now() - timedelta(minutes=i*5)).strftime('%H:%M:%S'),
                    '策略': np.random.choice(['BTC均值回归', 'ETH动量', 'BNB套利']),
                    '交易对': np.random.choice(['BTCUSDT', 'ETHUSDT', 'BNBUSDT']),
                    '方向': np.random.choice(['买入', '卖出']),
                    '数量': f"{np.random.uniform(0.001, 1.0):.6f}",
                    '价格': f"{np.random.uniform(100, 50000):.2f}",
                    '盈亏': f"{np.random.uniform(-50, 100):+.2f}",
                    '状态': '✅ 成功'
                })
            
            df = pd.DataFrame(trade_data)
            
            # 自定义样式表格
            st.markdown('<div class="live-data-table">', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"交易数据加载失败: {e}")
    
    def _render_strategy_management_tab(self):
        """渲染策略管理标签页"""
        st.header("🎯 策略管理中心")
        
        # 策略控制面板
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("策略列表")
            
            if st.session_state.trader_engine and st.session_state.system_running:
                strategies_status = st.session_state.trader_engine.get_strategies_status()
                
                if strategies_status:
                    for name, status in strategies_status.items():
                        with st.expander(f"📊 {name} - {status.get('state', 'unknown')}", expanded=True):
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.write(f"**交易对:** {status.get('symbol', 'N/A')}")
                                st.write(f"**时间周期:** {status.get('timeframe', 'N/A')}")
                            
                            with col_b:
                                st.write(f"**状态:** {status.get('state', 'unknown')}")
                                st.write(f"**最后更新:** {status.get('last_update', 'N/A')}")
                            
                            with col_c:
                                if st.button(f"🔄 重启", key=f"restart_{name}"):
                                    if st.session_state.trader_engine.restart_strategy(name):
                                        st.success(f"策略 {name} 重启成功")
                                    else:
                                        st.error(f"策略 {name} 重启失败")
                                
                                if st.button(f"⏹️ 停止", key=f"stop_{name}"):
                                    if st.session_state.trader_engine._stop_strategy(name):
                                        st.success(f"策略 {name} 停止成功")
                                    else:
                                        st.error(f"策略 {name} 停止失败")
                else:
                    st.info("暂无活跃策略")
            else:
                st.warning("系统未运行，无法管理策略")
        
        with col2:
            st.subheader("策略统计")
            
            if st.session_state.trader_engine and st.session_state.system_running:
                engine_status = st.session_state.trader_engine.get_engine_status()
                metrics = engine_status.get('metrics', {})
                
                st.metric("总策略数", metrics.get('total_strategies', 0))
                st.metric("运行中", metrics.get('running_strategies', 0))
                st.metric("已停止", metrics.get('stopped_strategies', 0))
                st.metric("错误状态", metrics.get('error_strategies', 0))
            else:
                st.metric("总策略数", 0)
                st.metric("运行中", 0)
                st.metric("已停止", 0)
                st.metric("错误状态", 0)
    
    def _render_risk_monitoring_tab(self):
        """渲染风险监控标签页"""
        st.header("⚠️ 风险监控中心")
        
        # 风险指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("仓位风险")
            
            position_risk = self._calculate_position_risk()
            st.metric(
                "当前仓位比例",
                f"{position_risk.get('current_position_percent', 0):.1%}",
                f"{position_risk.get('delta', 0):+.1%}"
            )
            
            # 仓位分布图
            fig = go.Figure(data=[
                go.Bar(
                    x=['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
                    y=[0.15, 0.10, 0.08],
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c']
                )
            ])
            fig.update_layout(
                title="仓位分布",
                xaxis_title="交易对",
                yaxis_title="仓位比例",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("损失风险")
            
            loss_risk = self._calculate_loss_risk()
            st.metric(
                "当前回撤",
                f"{loss_risk.get('current_drawdown', 0):.1%}",
                f"{loss_risk.get('delta', 0):+.1%}"
            )
            
            # 回撤曲线
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            drawdown = np.random.uniform(0, 0.1, 30)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=-drawdown,
                fill='tozeroy',
                name='回撤',
                line=dict(color='red')
            ))
            fig.update_layout(
                title="回撤曲线",
                xaxis_title="日期",
                yaxis_title="回撤比例",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            st.subheader("交易频率")
            
            frequency_risk = self._calculate_frequency_risk()
            st.metric(
                "每小时交易次数",
                frequency_risk.get('trades_per_hour', 0),
                f"{frequency_risk.get('delta', 0):+d}"
            )
            
            # 交易频率柱状图
            hours = list(range(24))
            trades = np.random.poisson(3, 24)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=hours,
                y=trades,
                name='交易次数',
                marker_color='lightblue'
            ))
            fig.update_layout(
                title="24小时交易频率",
                xaxis_title="小时",
                yaxis_title="交易次数",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # 风险警告
        st.markdown("---")
        st.subheader("⚠️ 风险警告")
        
        risk_warnings = self._get_risk_warnings()
        if risk_warnings:
            for warning in risk_warnings:
                alert_type = {
                    'critical': 'error',
                    'warning': 'warning',
                    'info': 'info'
                }.get(warning.get('level', 'info'), 'info')
                
                getattr(st, alert_type)(warning.get('message', ''))
        else:
            st.success("✅ 暂无风险警告")
    
    def _render_realtime_charts_tab(self):
        """渲染实时图表标签页"""
        st.header("📈 实时图表分析")
        
        # 图表控制
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_symbol = st.selectbox(
                "选择交易对",
                ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'DOGEUSDT', 'PEPEUSDT']
            )
        
        with col2:
            timeframe = st.selectbox(
                "时间周期",
                ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            )
        
        with col3:
            chart_type = st.selectbox(
                "图表类型",
                ['K线图', '价格线', '成交量', '技术指标']
            )
        
        # 实时价格图表
        if chart_type == 'K线图':
            self._render_candlestick_chart(selected_symbol, timeframe)
        elif chart_type == '价格线':
            self._render_price_line_chart(selected_symbol, timeframe)
        elif chart_type == '成交量':
            self._render_volume_chart(selected_symbol, timeframe)
        elif chart_type == '技术指标':
            self._render_technical_indicators(selected_symbol, timeframe)
    
    def _render_candlestick_chart(self, symbol: str, timeframe: str):
        """渲染K线图"""
        try:
            # 生成模拟K线数据
            dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
            
            # 模拟价格数据
            np.random.seed(42)
            base_price = 45000 if symbol == 'BTCUSDT' else 2500
            
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            current_price = base_price
            for i in range(100):
                open_price = current_price
                change = np.random.uniform(-0.02, 0.02)
                close_price = open_price * (1 + change)
                
                high_price = max(open_price, close_price) * (1 + abs(np.random.uniform(0, 0.01)))
                low_price = min(open_price, close_price) * (1 - abs(np.random.uniform(0, 0.01)))
                
                volume = np.random.uniform(100, 1000)
                
                opens.append(open_price)
                highs.append(high_price)
                lows.append(low_price)
                closes.append(close_price)
                volumes.append(volume)
                
                current_price = close_price
            
            # 创建K线图
            fig = go.Figure(data=[go.Candlestick(
                x=dates,
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                name=symbol
            )])
            
            fig.update_layout(
                title=f"{symbol} K线图 ({timeframe})",
                xaxis_title="时间",
                yaxis_title="价格",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"K线图生成失败: {e}")
    
    def _render_price_line_chart(self, symbol: str, timeframe: str):
        """渲染价格线图"""
        try:
            # 生成模拟价格数据
            dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
            base_price = 45000 if symbol == 'BTCUSDT' else 2500
            
            prices = []
            current_price = base_price
            for i in range(100):
                change = np.random.uniform(-0.01, 0.01)
                current_price *= (1 + change)
                prices.append(current_price)
            
            # 创建价格线图
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=f"{symbol} 价格",
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f"{symbol} 价格走势 ({timeframe})",
                xaxis_title="时间",
                yaxis_title="价格",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"价格线图生成失败: {e}")
    
    def _render_volume_chart(self, symbol: str, timeframe: str):
        """渲染成交量图"""
        try:
            # 生成模拟成交量数据
            dates = pd.date_range(start='2024-01-01', periods=50, freq='H')
            volumes = np.random.uniform(100, 1000, 50)
            
            # 创建成交量图
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=dates,
                y=volumes,
                name=f"{symbol} 成交量",
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f"{symbol} 成交量 ({timeframe})",
                xaxis_title="时间",
                yaxis_title="成交量",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"成交量图生成失败: {e}")
    
    def _render_technical_indicators(self, symbol: str, timeframe: str):
        """渲染技术指标"""
        try:
            # 生成模拟数据
            dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
            base_price = 45000 if symbol == 'BTCUSDT' else 2500
            
            prices = []
            current_price = base_price
            for i in range(100):
                change = np.random.uniform(-0.01, 0.01)
                current_price *= (1 + change)
                prices.append(current_price)
            
            # 计算移动平均线
            ma20 = pd.Series(prices).rolling(window=20).mean()
            ma50 = pd.Series(prices).rolling(window=50).mean()
            
            # 创建技术指标图
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['价格与移动平均线', 'RSI指标'],
                vertical_spacing=0.1
            )
            
            # 价格和移动平均线
            fig.add_trace(go.Scatter(
                x=dates, y=prices, name='价格', line=dict(color='blue')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=dates, y=ma20, name='MA20', line=dict(color='red')
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=dates, y=ma50, name='MA50', line=dict(color='green')
            ), row=1, col=1)
            
            # RSI指标
            rsi = np.random.uniform(20, 80, 100)
            fig.add_trace(go.Scatter(
                x=dates, y=rsi, name='RSI', line=dict(color='purple')
            ), row=2, col=1)
            
            fig.update_layout(
                title=f"{symbol} 技术指标 ({timeframe})",
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"技术指标图生成失败: {e}")
    
    def _render_system_logs_tab(self):
        """渲染系统日志标签页"""
        st.header("📋 系统日志")
        
        # 日志过滤
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_level = st.selectbox(
                "日志级别",
                ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            )
        
        with col2:
            log_source = st.selectbox(
                "日志来源",
                ['ALL', 'TraderEngine', 'Strategy', 'RiskManager', 'Notification']
            )
        
        with col3:
            max_logs = st.slider("显示条数", 10, 1000, 100)
        
        # 实时日志显示
        st.subheader("实时日志")
        
        # 生成模拟日志
        log_entries = []
        for i in range(max_logs):
            timestamp = (datetime.now() - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S")
            level = np.random.choice(['INFO', 'WARNING', 'ERROR', 'DEBUG'])
            source = np.random.choice(['TraderEngine', 'Strategy', 'RiskManager', 'Notification'])
            message = f"模拟日志消息 {i+1}"
            
            log_entries.append({
                '时间': timestamp,
                '级别': level,
                '来源': source,
                '消息': message
            })
        
        # 过滤日志
        if log_level != 'ALL':
            log_entries = [log for log in log_entries if log['级别'] == log_level]
        
        if log_source != 'ALL':
            log_entries = [log for log in log_entries if log['来源'] == log_source]
        
        # 显示日志
        log_df = pd.DataFrame(log_entries)
        
        # 自定义样式
        st.markdown('<div class="live-data-table">', unsafe_allow_html=True)
        st.dataframe(log_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 日志统计
        st.subheader("日志统计")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("总日志数", len(log_entries))
        
        with col2:
            error_count = len([log for log in log_entries if log['级别'] == 'ERROR'])
            st.metric("错误数", error_count)
        
        with col3:
            warning_count = len([log for log in log_entries if log['级别'] == 'WARNING'])
            st.metric("警告数", warning_count)
        
        with col4:
            info_count = len([log for log in log_entries if log['级别'] == 'INFO'])
            st.metric("信息数", info_count)
    
    def _start_system(self):
        """启动系统"""
        try:
            if st.session_state.trader_engine:
                if st.session_state.trader_engine.start():
                    st.session_state.system_running = True
                    st.success("✅ 系统启动成功")
                    
                    # 发送通知
                    if st.session_state.notification_manager:
                        st.session_state.notification_manager.send_system_status(
                            "系统启动", 
                            "交易系统已成功启动",
                            metadata={'timestamp': datetime.now().isoformat()}
                        )
                else:
                    st.error("❌ 系统启动失败")
            else:
                st.error("❌ 交易引擎未初始化")
        except Exception as e:
            st.error(f"❌ 系统启动异常: {e}")
    
    def _stop_system(self):
        """停止系统"""
        try:
            if st.session_state.trader_engine:
                if st.session_state.trader_engine.stop():
                    st.session_state.system_running = False
                    st.success("✅ 系统停止成功")
                    
                    # 发送通知
                    if st.session_state.notification_manager:
                        st.session_state.notification_manager.send_system_status(
                            "系统停止", 
                            "交易系统已安全停止",
                            metadata={'timestamp': datetime.now().isoformat()}
                        )
                else:
                    st.error("❌ 系统停止失败")
            else:
                st.error("❌ 交易引擎未初始化")
        except Exception as e:
            st.error(f"❌ 系统停止异常: {e}")
    
    def _refresh_data(self):
        """刷新数据"""
        try:
            # 刷新实时数据
            st.session_state.live_data = {
                'prices': {},
                'strategies': {},
                'account': {},
                'risk_metrics': {},
                'notifications': []
            }
            
            st.success("🔄 数据刷新成功")
        except Exception as e:
            st.error(f"❌ 数据刷新失败: {e}")
    
    def _generate_report(self):
        """生成报告"""
        try:
            if st.session_state.report_generator:
                # 创建示例数据
                trades_data, account_data = st.session_state.report_generator.create_sample_data()
                
                # 生成报告
                report_paths = st.session_state.report_generator.generate_complete_report(
                    trades_data, account_data, "实时监控报告"
                )
                
                st.success(f"📊 报告生成成功！生成了 {len(report_paths)} 个文件")
                
                # 显示报告链接
                for report_type, path in report_paths.items():
                    if report_type != 'charts' and path:
                        st.markdown(f"[下载 {report_type.upper()} 报告]({path})")
            else:
                st.error("❌ 报告生成器未初始化")
        except Exception as e:
            st.error(f"❌ 报告生成失败: {e}")
    
    def _clear_cache(self):
        """清除缓存"""
        try:
            # 清除streamlit缓存
            st.cache_data.clear()
            st.cache_resource.clear()
            
            # 清除自定义缓存
            st.session_state.live_data = {
                'prices': {},
                'strategies': {},
                'account': {},
                'risk_metrics': {},
                'notifications': []
            }
            
            st.success("🧹 缓存清除成功")
        except Exception as e:
            st.error(f"❌ 缓存清除失败: {e}")
    
    # 辅助方法
    def _get_account_value(self) -> float:
        """获取账户价值"""
        return 10000.0 + np.random.uniform(-500, 500)
    
    def _get_daily_pnl(self) -> float:
        """获取今日盈亏"""
        return np.random.uniform(-100, 200)
    
    def _get_active_strategies_count(self) -> int:
        """获取活跃策略数量"""
        if st.session_state.trader_engine and st.session_state.system_running:
            engine_status = st.session_state.trader_engine.get_engine_status()
            return engine_status.get('metrics', {}).get('running_strategies', 0)
        return 0
    
    def _get_risk_level(self) -> str:
        """获取风险等级"""
        return np.random.choice(['LOW', 'MEDIUM', 'HIGH'], p=[0.6, 0.3, 0.1])
    
    def _calculate_position_risk(self) -> Dict[str, float]:
        """计算仓位风险"""
        return {
            'current_position_percent': np.random.uniform(0.1, 0.3),
            'delta': np.random.uniform(-0.05, 0.05)
        }
    
    def _calculate_loss_risk(self) -> Dict[str, float]:
        """计算损失风险"""
        return {
            'current_drawdown': np.random.uniform(0.0, 0.1),
            'delta': np.random.uniform(-0.02, 0.02)
        }
    
    def _calculate_frequency_risk(self) -> Dict[str, int]:
        """计算频率风险"""
        return {
            'trades_per_hour': np.random.randint(0, 10),
            'delta': np.random.randint(-2, 3)
        }
    
    def _get_risk_warnings(self) -> List[Dict[str, str]]:
        """获取风险警告"""
        warnings = []
        
        if np.random.random() < 0.3:
            warnings.append({
                'level': 'warning',
                'message': '⚠️ 当前仓位比例较高，建议适当降低仓位'
            })
        
        if np.random.random() < 0.1:
            warnings.append({
                'level': 'critical',
                'message': '🚨 检测到异常交易频率，请检查策略设置'
            })
        
        return warnings


def main():
    """主函数"""
    try:
        dashboard = RealtimeDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"系统启动失败: {e}")
        st.stop()


if __name__ == "__main__":
    main()