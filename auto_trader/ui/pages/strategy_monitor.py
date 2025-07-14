"""
策略监控页面

提供实时的策略运行状态监控，包括：
- 策略运行状态展示
- 实时收益率监控
- 交易信号展示
- 策略参数调整
- 策略启停控制
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from auto_trader.strategies.base import StrategyConfig
from auto_trader.strategies.mean_reversion import MeanReversionStrategy


class StrategyMonitor:
    """
    策略监控页面类
    
    负责：
    - 策略状态实时监控
    - 策略性能指标展示
    - 策略参数动态调整
    - 策略启停控制
    """
    
    def __init__(self, trading_system: Dict[str, Any]):
        """
        初始化策略监控器
        
        Args:
            trading_system: 交易系统组件字典
        """
        self.trading_system = trading_system
        self.data_manager = trading_system['data_manager']
        self.account_manager = trading_system['account_manager']
        self.risk_manager = trading_system['risk_manager']
        self.broker = trading_system['broker']
        self.strategies = trading_system.get('strategies', {})
    
    def render(self):
        """渲染策略监控页面"""
        st.header("🎯 策略监控中心")
        
        # 页面布局
        self._render_strategy_overview()
        st.markdown("---")
        self._render_strategy_details()
        st.markdown("---")
        self._render_strategy_controls()
    
    def _render_strategy_overview(self):
        """渲染策略概览部分"""
        st.subheader("📊 策略概览")
        
        # 策略统计指标
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_strategies = len(self.strategies)
            active_strategies = sum(1 for s in self.strategies.values() if s.get('status') == 'running')
            st.metric(
                label="策略总数",
                value=total_strategies,
                delta=f"{active_strategies} 运行中"
            )
        
        with col2:
            # 计算总收益率
            total_pnl = 0.0
            for strategy_name, strategy_info in self.strategies.items():
                total_pnl += strategy_info.get('pnl', 0.0)
            
            st.metric(
                label="总收益率",
                value=f"{total_pnl:.2f}%",
                delta="今日"
            )
        
        with col3:
            # 计算总交易次数
            total_trades = 0
            for strategy_name, strategy_info in self.strategies.items():
                total_trades += strategy_info.get('trade_count', 0)
            
            st.metric(
                label="总交易次数",
                value=total_trades,
                delta="全部策略"
            )
        
        with col4:
            # 计算胜率
            winning_trades = 0
            total_completed_trades = 0
            for strategy_name, strategy_info in self.strategies.items():
                winning_trades += strategy_info.get('winning_trades', 0)
                total_completed_trades += strategy_info.get('completed_trades', 0)
            
            win_rate = (winning_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0
            st.metric(
                label="整体胜率",
                value=f"{win_rate:.1f}%",
                delta="所有策略"
            )
        
        # 策略性能图表
        self._render_performance_chart()
    
    def _render_performance_chart(self):
        """渲染策略性能图表"""
        st.subheader("📈 策略性能趋势")
        
        # 生成示例性能数据
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        
        fig = go.Figure()
        
        # 为每个策略添加性能曲线
        strategy_names = list(self.strategies.keys()) if self.strategies else ['均值回归策略']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, strategy_name in enumerate(strategy_names):
            # 生成示例收益数据
            returns = [0] + [pd.np.random.normal(0.1, 1.0) for _ in range(len(dates)-1)]
            cumulative_returns = pd.Series(returns).cumsum()
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=cumulative_returns,
                mode='lines',
                name=strategy_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title="策略累计收益率对比",
            xaxis_title="日期",
            yaxis_title="累计收益率 (%)",
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategy_details(self):
        """渲染策略详细信息"""
        st.subheader("📋 策略详情")
        
        if not self.strategies:
            st.info("暂无运行中的策略，请在下方添加策略。")
            return
        
        # 策略详情表格
        strategy_data = []
        for strategy_name, strategy_info in self.strategies.items():
            strategy_data.append({
                '策略名称': strategy_name,
                '状态': strategy_info.get('status', 'unknown'),
                '交易对': strategy_info.get('symbol', 'N/A'),
                '时间周期': strategy_info.get('timeframe', 'N/A'),
                '收益率': f"{strategy_info.get('pnl', 0.0):.2f}%",
                '交易次数': strategy_info.get('trade_count', 0),
                '胜率': f"{strategy_info.get('win_rate', 0.0):.1f}%",
                '最大回撤': f"{strategy_info.get('max_drawdown', 0.0):.2f}%",
                '最后更新': strategy_info.get('last_update', 'N/A')
            })
        
        df = pd.DataFrame(strategy_data)
        
        # 使用可编辑的数据框
        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "策略名称": st.column_config.TextColumn("策略名称", width="medium"),
                "状态": st.column_config.SelectboxColumn(
                    "状态",
                    options=["running", "stopped", "paused", "error"],
                    width="small"
                ),
                "收益率": st.column_config.NumberColumn(
                    "收益率",
                    format="%.2f%%",
                    width="small"
                ),
                "胜率": st.column_config.NumberColumn(
                    "胜率", 
                    format="%.1f%%",
                    width="small"
                )
            }
        )
        
        # 检查是否有状态变更
        for i, row in edited_df.iterrows():
            strategy_name = row['策略名称']
            new_status = row['状态']
            if strategy_name in self.strategies:
                old_status = self.strategies[strategy_name].get('status')
                if old_status != new_status:
                    self._handle_strategy_status_change(strategy_name, new_status)
    
    def _render_strategy_controls(self):
        """渲染策略控制面板"""
        st.subheader("🎛️ 策略控制")
        
        # 分为两列：添加策略 和 策略操作
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ➕ 添加新策略")
            
            with st.form("add_strategy_form"):
                strategy_name = st.text_input(
                    "策略名称",
                    placeholder="例如：BTCUSDT_均值回归",
                    help="为策略指定一个唯一的名称"
                )
                
                strategy_type = st.selectbox(
                    "策略类型",
                    options=["均值回归", "趋势跟随", "网格交易", "套利策略"],
                    help="选择要使用的策略类型"
                )
                
                symbol = st.selectbox(
                    "交易对",
                    options=["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "SOLUSDT"],
                    help="选择要交易的币种对"
                )
                
                timeframe = st.selectbox(
                    "时间周期",
                    options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                    index=4,  # 默认选择1h
                    help="选择策略运行的时间周期"
                )
                
                # 策略参数配置
                st.markdown("**策略参数配置**")
                if strategy_type == "均值回归":
                    window_size = st.number_input("移动平均窗口", value=20, min_value=5, max_value=100)
                    std_multiplier = st.number_input("标准差倍数", value=2.0, min_value=0.5, max_value=5.0, step=0.1)
                    
                    strategy_params = {
                        'window_size': window_size,
                        'std_multiplier': std_multiplier
                    }
                else:
                    st.info(f"{strategy_type}策略参数配置待实现")
                    strategy_params = {}
                
                # 风险管理设置
                st.markdown("**风险管理设置**")
                max_position_size = st.number_input(
                    "最大持仓比例 (%)",
                    value=10.0,
                    min_value=1.0,
                    max_value=50.0,
                    step=1.0,
                    help="该策略的最大持仓比例"
                )
                
                stop_loss = st.number_input(
                    "止损比例 (%)",
                    value=5.0,
                    min_value=1.0,
                    max_value=20.0,
                    step=0.5,
                    help="设置止损比例"
                )
                
                submitted = st.form_submit_button("🚀 启动策略", type="primary")
                
                if submitted:
                    if strategy_name and strategy_name not in self.strategies:
                        success = self._add_new_strategy(
                            strategy_name, strategy_type, symbol, timeframe,
                            strategy_params, max_position_size, stop_loss
                        )
                        if success:
                            st.success(f"✅ 策略 '{strategy_name}' 添加成功！")
                            st.rerun()
                        else:
                            st.error("❌ 策略添加失败，请检查参数设置。")
                    elif not strategy_name:
                        st.error("请输入策略名称")
                    else:
                        st.error("策略名称已存在，请使用其他名称")
        
        with col2:
            st.markdown("#### 🔧 策略操作")
            
            if self.strategies:
                selected_strategy = st.selectbox(
                    "选择策略",
                    options=list(self.strategies.keys()),
                    help="选择要操作的策略"
                )
                
                if selected_strategy:
                    strategy_info = self.strategies[selected_strategy]
                    current_status = strategy_info.get('status', 'unknown')
                    
                    st.info(f"当前状态: **{current_status}**")
                    
                    # 策略操作按钮
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    
                    with col_btn1:
                        if st.button("▶️ 启动", disabled=(current_status == 'running')):
                            self._handle_strategy_status_change(selected_strategy, 'running')
                            st.success("策略已启动")
                            st.rerun()
                    
                    with col_btn2:
                        if st.button("⏸️ 暂停", disabled=(current_status != 'running')):
                            self._handle_strategy_status_change(selected_strategy, 'paused')
                            st.success("策略已暂停")
                            st.rerun()
                    
                    with col_btn3:
                        if st.button("⏹️ 停止", disabled=(current_status == 'stopped')):
                            self._handle_strategy_status_change(selected_strategy, 'stopped')
                            st.success("策略已停止")
                            st.rerun()
                    
                    # 删除策略
                    st.markdown("---")
                    if st.button("🗑️ 删除策略", type="secondary", help="永久删除该策略"):
                        if st.session_state.get('confirm_delete') == selected_strategy:
                            self._remove_strategy(selected_strategy)
                            st.success(f"策略 '{selected_strategy}' 已删除")
                            del st.session_state['confirm_delete']
                            st.rerun()
                        else:
                            st.session_state['confirm_delete'] = selected_strategy
                            st.warning("⚠️ 请再次点击确认删除")
            else:
                st.info("暂无策略可操作")
    
    def _add_new_strategy(self, name: str, strategy_type: str, symbol: str, 
                         timeframe: str, params: Dict, max_position: float, 
                         stop_loss: float) -> bool:
        """
        添加新策略
        
        Args:
            name: 策略名称
            strategy_type: 策略类型
            symbol: 交易对
            timeframe: 时间周期
            params: 策略参数
            max_position: 最大持仓比例
            stop_loss: 止损比例
            
        Returns:
            bool: 是否添加成功
        """
        try:
            # 创建策略配置
            config = StrategyConfig(
                name=name,
                symbol=symbol,
                timeframe=timeframe,
                parameters=params
            )
            
            # 根据策略类型创建策略实例
            if strategy_type == "均值回归":
                strategy_instance = MeanReversionStrategy(config)
                strategy_instance.initialize()
            else:
                # 其他策略类型待实现
                st.warning(f"策略类型 '{strategy_type}' 暂未实现")
                return False
            
            # 添加到策略字典
            self.strategies[name] = {
                'instance': strategy_instance,
                'config': config,
                'type': strategy_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'running',
                'pnl': 0.0,
                'trade_count': 0,
                'winning_trades': 0,
                'completed_trades': 0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'max_position': max_position,
                'stop_loss': stop_loss,
                'last_update': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'created_at': datetime.now()
            }
            
            # 更新session state
            st.session_state.trading_system['strategies'] = self.strategies
            
            return True
            
        except Exception as e:
            st.error(f"创建策略失败: {e}")
            return False
    
    def _handle_strategy_status_change(self, strategy_name: str, new_status: str):
        """
        处理策略状态变更
        
        Args:
            strategy_name: 策略名称
            new_status: 新状态
        """
        if strategy_name in self.strategies:
            old_status = self.strategies[strategy_name].get('status')
            self.strategies[strategy_name]['status'] = new_status
            self.strategies[strategy_name]['last_update'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 更新session state
            st.session_state.trading_system['strategies'] = self.strategies
            
            # 记录状态变更日志
            st.info(f"策略 '{strategy_name}' 状态: {old_status} → {new_status}")
    
    def _remove_strategy(self, strategy_name: str):
        """
        移除策略
        
        Args:
            strategy_name: 策略名称
        """
        if strategy_name in self.strategies:
            # 停止策略
            self.strategies[strategy_name]['status'] = 'stopped'
            
            # 从字典中删除
            del self.strategies[strategy_name]
            
            # 更新session state
            st.session_state.trading_system['strategies'] = self.strategies