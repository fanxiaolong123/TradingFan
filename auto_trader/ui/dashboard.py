"""
AutoTrader主仪表板 - Streamlit Web界面

这是系统的主要Web界面，提供：
- 多页面导航结构
- 实时数据展示
- 交互式控制面板
- 与核心系统的集成接口
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入核心模块
from auto_trader.utils import get_config, get_logger
from auto_trader.core.data import DataManager, BinanceDataProvider
from auto_trader.core.broker import SimulatedBroker
from auto_trader.core.account import AccountManager, AccountType
from auto_trader.core.risk import RiskManager, RiskLimits
from auto_trader.strategies.mean_reversion import MeanReversionStrategy
from auto_trader.strategies.base import StrategyConfig

# 导入页面模块
from .pages.strategy_monitor import StrategyMonitor
from .pages.asset_manager import AssetManager
from .pages.backtest_analyzer import BacktestAnalyzer
from .pages.trade_logger import TradeLogger
from .pages.config_manager import ConfigManager


class StreamlitDashboard:
    """
    AutoTrader Streamlit 主仪表板类
    
    负责：
    - 页面路由和导航
    - 系统状态管理
    - 数据缓存和刷新
    - 核心组件初始化
    """
    
    def __init__(self):
        """初始化仪表板"""
        # 设置页面配置
        st.set_page_config(
            page_title="AutoTrader 量化交易系统",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/autotrader',
                'Report a bug': "https://github.com/your-repo/autotrader/issues",
                'About': "AutoTrader - 专业量化交易系统"
            }
        )
        
        # 初始化状态
        if 'system_initialized' not in st.session_state:
            st.session_state.system_initialized = False
            st.session_state.trading_system = None
            st.session_state.config = None
            st.session_state.logger = None
        
        # 初始化系统组件
        self._initialize_system()
        
        # 设置样式
        self._setup_custom_styles()
    
    def _initialize_system(self):
        """初始化交易系统组件"""
        if not st.session_state.system_initialized:
            try:
                # 加载配置
                st.session_state.config = get_config()
                st.session_state.logger = get_logger(__name__)
                
                # 初始化数据管理器
                data_manager = DataManager()
                binance_config = st.session_state.config.get_data_source_config('binance')
                binance_provider = BinanceDataProvider(
                    api_key=binance_config.get('api_key'),
                    api_secret=binance_config.get('api_secret')
                )
                data_manager.add_provider('binance', binance_provider, is_default=True)
                
                # 初始化账户管理器
                account_config = st.session_state.config.get_account_config()
                account_type = AccountType(account_config.get('account_type', 'SPOT'))
                account_manager = AccountManager(account_type)
                account_manager.set_initial_balance(account_config.get('initial_balance', {'USDT': 10000.0}))
                
                # 初始化风险管理器
                risk_config = st.session_state.config.get_risk_management_config()
                if risk_config.get('enabled', True):
                    risk_limits = RiskLimits(
                        max_position_percent=risk_config.get('position_limits', {}).get('max_position_percent', 0.1),
                        max_total_position_percent=risk_config.get('position_limits', {}).get('max_total_position_percent', 0.8),
                        max_daily_loss_percent=risk_config.get('loss_limits', {}).get('max_daily_loss_percent', 0.05),
                        max_total_loss_percent=risk_config.get('loss_limits', {}).get('max_total_loss_percent', 0.20),
                        max_drawdown_percent=risk_config.get('loss_limits', {}).get('max_drawdown_percent', 0.15),
                        max_trades_per_hour=risk_config.get('frequency_limits', {}).get('max_trades_per_hour', 10),
                        max_trades_per_day=risk_config.get('frequency_limits', {}).get('max_trades_per_day', 100),
                    )
                    risk_manager = RiskManager(risk_limits)
                else:
                    risk_manager = None
                
                # 初始化模拟经纪商
                broker = SimulatedBroker(
                    initial_balance=account_config.get('initial_balance', {'USDT': 10000.0}),
                    commission_rate=st.session_state.config.get('trading.default_commission_rate', 0.001)
                )
                
                # 保存到session state
                st.session_state.trading_system = {
                    'data_manager': data_manager,
                    'account_manager': account_manager,
                    'risk_manager': risk_manager,
                    'broker': broker,
                    'strategies': {}
                }
                
                st.session_state.system_initialized = True
                st.session_state.logger.info("UI Dashboard系统初始化完成")
                
            except Exception as e:
                st.error(f"系统初始化失败: {e}")
                st.session_state.logger.error(f"UI Dashboard初始化失败: {e}")
    
    def _setup_custom_styles(self):
        """设置自定义样式"""
        st.markdown("""
        <style>
        /* 主标题样式 */
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            padding: 1rem 0;
            border-bottom: 2px solid #1f77b4;
            margin-bottom: 2rem;
        }
        
        /* 指标卡片样式 */
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin: 0.5rem 0;
        }
        
        /* 成功状态 */
        .status-success {
            color: #28a745;
            font-weight: bold;
        }
        
        /* 警告状态 */
        .status-warning {
            color: #ffc107;
            font-weight: bold;
        }
        
        /* 错误状态 */
        .status-error {
            color: #dc3545;
            font-weight: bold;
        }
        
        /* 侧边栏样式 */
        .sidebar .sidebar-content {
            background-color: #f1f3f4;
        }
        
        /* 隐藏Streamlit菜单和页脚 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """运行主仪表板"""
        # 主标题
        st.markdown('<h1 class="main-header">📈 AutoTrader 量化交易系统</h1>', unsafe_allow_html=True)
        
        # 侧边栏导航
        page = self._render_sidebar()
        
        # 根据选择的页面渲染内容
        if page == "概览":
            self._render_overview_page()
        elif page == "策略监控":
            if st.session_state.system_initialized:
                strategy_monitor = StrategyMonitor(st.session_state.trading_system)
                strategy_monitor.render()
            else:
                st.error("系统未初始化，无法使用策略监控功能")
        elif page == "资产管理":
            if st.session_state.system_initialized:
                asset_manager = AssetManager(st.session_state.trading_system)
                asset_manager.render()
            else:
                st.error("系统未初始化，无法使用资产管理功能")
        elif page == "回测分析":
            if st.session_state.system_initialized:
                backtest_analyzer = BacktestAnalyzer(st.session_state.trading_system)
                backtest_analyzer.render()
            else:
                st.error("系统未初始化，无法使用回测分析功能")
        elif page == "交易日志":
            if st.session_state.system_initialized:
                trade_logger = TradeLogger(st.session_state.trading_system)
                trade_logger.render()
            else:
                st.error("系统未初始化，无法使用交易日志功能")
        elif page == "系统设置":
            if st.session_state.system_initialized:
                config_manager = ConfigManager(st.session_state.trading_system)
                config_manager.render()
            else:
                st.error("系统未初始化，无法使用系统设置功能")
    
    def _render_sidebar(self) -> str:
        """渲染侧边栏导航"""
        st.sidebar.title("🎛️ 控制面板")
        
        # 系统状态显示
        if st.session_state.system_initialized:
            st.sidebar.success("✅ 系统已就绪")
        else:
            st.sidebar.error("❌ 系统未初始化")
        
        # 实时时间显示
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.sidebar.info(f"🕒 当前时间: {current_time}")
        
        # 导航菜单
        st.sidebar.markdown("---")
        page = st.sidebar.selectbox(
            "选择页面",
            ["概览", "策略监控", "资产管理", "回测分析", "交易日志", "系统设置"],
            index=0
        )
        
        # 快速操作
        st.sidebar.markdown("---")
        st.sidebar.markdown("### ⚡ 快速操作")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔄 刷新数据", help="刷新所有数据"):
                st.rerun()
        
        with col2:
            if st.button("⚙️ 重置系统", help="重置系统状态"):
                # 清除session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        return page
    
    def _render_overview_page(self):
        """渲染概览页面"""
        st.header("📊 系统概览")
        
        # 系统状态概览
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="系统状态",
                value="运行中" if st.session_state.system_initialized else "离线",
                delta="正常" if st.session_state.system_initialized else "异常"
            )
        
        with col2:
            st.metric(
                label="活跃策略",
                value=len(st.session_state.trading_system.get('strategies', {})),
                delta="个"
            )
        
        with col3:
            # 获取账户信息
            if st.session_state.system_initialized:
                account_manager = st.session_state.trading_system['account_manager']
                account_summary = account_manager.get_account_summary()
                total_value = account_summary.get('total_value_usdt', 0)
            else:
                total_value = 0
            
            st.metric(
                label="账户价值",
                value=f"{total_value:.2f} USDT",
                delta="实时"
            )
        
        with col4:
            st.metric(
                label="今日收益",
                value="0.00%",
                delta="待实现"
            )
        
        # 图表展示区域
        st.markdown("---")
        
        # 创建示例图表
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 资产趋势")
            
            # 生成示例数据
            dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
            values = 10000 + (dates - dates[0]).days * 5 + pd.Series(range(len(dates))).apply(lambda x: x % 100 - 50)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=values,
                mode='lines',
                name='账户价值',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.update_layout(
                title="账户价值变化趋势",
                xaxis_title="日期",
                yaxis_title="价值 (USDT)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 策略分布")
            
            # 策略分布饼图
            strategy_data = {
                '均值回归': 1,
                '趋势跟随': 0,
                '套利策略': 0,
                '其他': 0
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(strategy_data.keys()),
                values=list(strategy_data.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title="活跃策略分布",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 最近交易记录
        st.markdown("---")
        st.subheader("📋 最近交易记录")
        
        # 示例交易记录
        trade_data = pd.DataFrame({
            '时间': [datetime.now() - timedelta(hours=i) for i in range(5)],
            '策略': ['均值回归'] * 5,
            '交易对': ['BTCUSDT', 'ETHUSDT', 'BTCUSDT', 'ADAUSDT', 'BTCUSDT'],
            '类型': ['买入', '卖出', '买入', '卖出', '买入'],
            '数量': [0.001, 0.1, 0.001, 100, 0.001],
            '价格': [45000, 2500, 44800, 0.45, 45200],
            '状态': ['已成交', '已成交', '已成交', '已成交', '已成交']
        })
        
        st.dataframe(
            trade_data,
            use_container_width=True,
            hide_index=True
        )
    


def main():
    """主函数 - 启动Streamlit应用"""
    dashboard = StreamlitDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()