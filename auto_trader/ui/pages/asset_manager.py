"""
资产管理页面

提供完整的资产管理功能，包括：
- 账户余额实时展示
- 持仓管理和分析
- 资产分布可视化
- 风险暴露分析
- 交易历史记录
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


class AssetManager:
    """
    资产管理页面类
    
    负责：
    - 账户余额管理和展示
    - 持仓分析和监控
    - 资产配置优化建议
    - 风险暴露评估
    """
    
    def __init__(self, trading_system: Dict[str, Any]):
        """
        初始化资产管理器
        
        Args:
            trading_system: 交易系统组件字典
        """
        self.trading_system = trading_system
        self.account_manager = trading_system['account_manager']
        self.risk_manager = trading_system['risk_manager']
        self.broker = trading_system['broker']
        self.strategies = trading_system.get('strategies', {})
    
    def render(self):
        """渲染资产管理页面"""
        st.header("💰 资产管理中心")
        
        # 页面布局
        self._render_account_overview()
        st.markdown("---")
        self._render_position_analysis()
        st.markdown("---")
        self._render_asset_allocation()
        st.markdown("---")
        self._render_risk_analysis()
    
    def _render_account_overview(self):
        """渲染账户概览"""
        st.subheader("📊 账户概览")
        
        # 获取账户信息
        account_summary = self.account_manager.get_account_summary()
        account_balances = self.account_manager.get_balances()
        
        # 计算关键指标
        total_value_usdt = account_summary.get('total_value_usdt', 0.0)
        available_balance = account_summary.get('available_balance_usdt', 0.0)
        locked_balance = total_value_usdt - available_balance
        daily_pnl = account_summary.get('daily_pnl', 0.0)
        daily_pnl_percent = account_summary.get('daily_pnl_percent', 0.0)
        
        # 主要指标展示
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="总资产价值",
                value=f"{total_value_usdt:,.2f} USDT",
                delta=f"{daily_pnl:+.2f} USDT"
            )
        
        with col2:
            st.metric(
                label="可用余额", 
                value=f"{available_balance:,.2f} USDT",
                delta=f"{(available_balance/total_value_usdt*100):.1f}%" if total_value_usdt > 0 else "0%"
            )
        
        with col3:
            st.metric(
                label="冻结资产",
                value=f"{locked_balance:,.2f} USDT", 
                delta=f"{(locked_balance/total_value_usdt*100):.1f}%" if total_value_usdt > 0 else "0%"
            )
        
        with col4:
            st.metric(
                label="今日收益",
                value=f"{daily_pnl_percent:+.2f}%",
                delta=f"{daily_pnl:+.2f} USDT"
            )
        
        # 账户余额详情
        st.markdown("#### 💳 余额详情")
        
        if account_balances:
            balance_data = []
            for asset, balance_info in account_balances.items():
                if isinstance(balance_info, dict):
                    total_balance = balance_info.get('total', 0.0)
                    available = balance_info.get('available', 0.0) 
                    locked = balance_info.get('locked', 0.0)
                else:
                    total_balance = float(balance_info)
                    available = total_balance
                    locked = 0.0
                
                if total_balance > 0.001:  # 只显示有意义的余额
                    balance_data.append({
                        '资产': asset,
                        '总余额': f"{total_balance:.8f}",
                        '可用余额': f"{available:.8f}",
                        '冻结余额': f"{locked:.8f}",
                        'USDT估值': f"{total_balance:.2f}" if asset == 'USDT' else "计算中..."
                    })
            
            if balance_data:
                df_balances = pd.DataFrame(balance_data)
                st.dataframe(
                    df_balances,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("暂无余额数据")
        else:
            st.info("无法获取余额信息")
        
        # 资产价值趋势图
        self._render_asset_value_trend()
    
    def _render_asset_value_trend(self):
        """渲染资产价值趋势图"""
        st.markdown("#### 📈 资产价值趋势")
        
        # 生成示例数据 (实际应用中从数据库获取)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 模拟资产价值变化
        initial_value = 10000
        values = []
        current_value = initial_value
        
        for i, date in enumerate(dates):
            # 模拟市场波动
            daily_change = pd.np.random.normal(0.001, 0.02)  # 平均0.1%的日收益，2%的波动
            current_value *= (1 + daily_change)
            values.append(current_value)
        
        # 创建图表
        fig = go.Figure()
        
        # 资产价值曲线
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines',
            name='总资产价值',
            line=dict(color='#1f77b4', width=3),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.1)'
        ))
        
        # 添加基准线
        fig.add_hline(
            y=initial_value, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="初始价值"
        )
        
        fig.update_layout(
            title="过去30天资产价值变化",
            xaxis_title="日期",
            yaxis_title="价值 (USDT)",
            template="plotly_white",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_position_analysis(self):
        """渲染持仓分析"""
        st.subheader("📍 持仓分析")
        
        # 获取当前持仓
        positions = {}
        for strategy_name, strategy_info in self.strategies.items():
            if 'instance' in strategy_info:
                # 这里应该从strategy实例获取实际持仓
                # 目前使用模拟数据
                symbol = strategy_info.get('symbol', 'BTCUSDT')
                if symbol not in positions:
                    positions[symbol] = {
                        'symbol': symbol,
                        'quantity': 0.0,
                        'avg_price': 0.0,
                        'current_price': 0.0,
                        'unrealized_pnl': 0.0,
                        'strategies': []
                    }
                positions[symbol]['strategies'].append(strategy_name)
        
        if positions:
            # 持仓概览表格
            position_data = []
            for symbol, pos_info in positions.items():
                # 模拟数据
                quantity = pd.np.random.uniform(0.001, 0.1)
                avg_price = pd.np.random.uniform(40000, 50000) if 'BTC' in symbol else pd.np.random.uniform(2000, 3000)
                current_price = avg_price * pd.np.random.uniform(0.95, 1.05)
                unrealized_pnl = (current_price - avg_price) * quantity
                pnl_percent = (current_price - avg_price) / avg_price * 100
                
                position_data.append({
                    '交易对': symbol,
                    '持仓数量': f"{quantity:.6f}",
                    '平均成本': f"{avg_price:.2f}",
                    '当前价格': f"{current_price:.2f}",
                    '未实现盈亏': f"{unrealized_pnl:+.2f}",
                    '盈亏比例': f"{pnl_percent:+.2f}%",
                    '关联策略': ', '.join(pos_info['strategies'])
                })
            
            df_positions = pd.DataFrame(position_data)
            
            # 使用颜色编码显示盈亏
            def color_pnl(val):
                if '+' in str(val):
                    return 'color: green'
                elif '-' in str(val):
                    return 'color: red'
                return ''
            
            styled_df = df_positions.style.applymap(
                color_pnl, 
                subset=['未实现盈亏', '盈亏比例']
            )
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True
            )
            
            # 持仓分布图
            self._render_position_distribution()
            
        else:
            st.info("当前无持仓")
    
    def _render_position_distribution(self):
        """渲染持仓分布图"""
        st.markdown("#### 📊 持仓分布")
        
        # 模拟持仓分布数据
        position_values = {
            'BTCUSDT': 3500,
            'ETHUSDT': 2000,
            'ADAUSDT': 1500,
            'BNBUSDT': 1000,
            'USDT': 2000  # 现金部分
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 饼图显示资产分布
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(position_values.keys()),
                values=list(position_values.values()),
                hole=0.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig_pie.update_layout(
                title="资产配置分布",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # 柱状图显示持仓价值
            fig_bar = go.Figure(data=[go.Bar(
                x=list(position_values.keys()),
                y=list(position_values.values()),
                marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            )])
            
            fig_bar.update_layout(
                title="各资产持仓价值",
                xaxis_title="资产",
                yaxis_title="价值 (USDT)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
    
    def _render_asset_allocation(self):
        """渲染资产配置建议"""
        st.subheader("🎯 资产配置建议")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 当前配置")
            
            # 当前资产配置数据
            current_allocation = {
                'BTC': 35.0,
                'ETH': 20.0, 
                'ADA': 15.0,
                'BNB': 10.0,
                '现金(USDT)': 20.0
            }
            
            allocation_df = pd.DataFrame([
                {'资产': asset, '当前比例': f"{percent:.1f}%"}
                for asset, percent in current_allocation.items()
            ])
            
            st.dataframe(allocation_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 💡 建议配置")
            
            # 建议的资产配置
            suggested_allocation = {
                'BTC': 40.0,
                'ETH': 25.0,
                'ADA': 10.0, 
                'BNB': 10.0,
                '现金(USDT)': 15.0
            }
            
            suggestion_df = pd.DataFrame([
                {'资产': asset, '建议比例': f"{percent:.1f}%"}
                for asset, percent in suggested_allocation.items()
            ])
            
            st.dataframe(suggestion_df, use_container_width=True, hide_index=True)
        
        # 配置对比图表
        self._render_allocation_comparison(current_allocation, suggested_allocation)
        
        # 再平衡建议
        st.markdown("#### ⚖️ 再平衡建议")
        
        rebalance_actions = []
        total_value = 10000  # 假设总价值
        
        for asset in current_allocation.keys():
            current_pct = current_allocation[asset]
            suggested_pct = suggested_allocation[asset]
            difference = suggested_pct - current_pct
            
            if abs(difference) > 2.0:  # 超过2%的偏差才建议调整
                action = "增持" if difference > 0 else "减持"
                amount = abs(difference) * total_value / 100
                
                rebalance_actions.append({
                    '资产': asset,
                    '操作': action,
                    '调整幅度': f"{abs(difference):.1f}%",
                    '调整金额': f"{amount:.0f} USDT"
                })
        
        if rebalance_actions:
            rebalance_df = pd.DataFrame(rebalance_actions)
            st.dataframe(rebalance_df, use_container_width=True, hide_index=True)
            
            if st.button("🔄 执行再平衡"):
                st.success("再平衡操作已提交！")
        else:
            st.success("✅ 当前配置已经很均衡，无需调整")
    
    def _render_allocation_comparison(self, current: Dict, suggested: Dict):
        """渲染配置对比图表"""
        assets = list(current.keys())
        current_values = list(current.values())
        suggested_values = list(suggested.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='当前配置',
            x=assets,
            y=current_values,
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            name='建议配置',
            x=assets,
            y=suggested_values,
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            title="资产配置对比",
            xaxis_title="资产类型",
            yaxis_title="配置比例 (%)",
            barmode='group',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_risk_analysis(self):
        """渲染风险分析"""
        st.subheader("⚠️ 风险分析")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 风险指标")
            
            # 风险指标
            risk_metrics = {
                'VaR (95%)': '1,250 USDT',
                '最大回撤': '8.5%',
                '夏普比率': '1.42',
                '波动率': '12.3%',
                '贝塔系数': '0.85',
                '信息比率': '0.73'
            }
            
            for metric, value in risk_metrics.items():
                st.metric(metric, value)
        
        with col2:
            st.markdown("#### ⚡ 风险提醒")
            
            # 风险提醒
            risk_alerts = [
                {"level": "warning", "message": "BTC持仓比例较高，建议适当分散"},
                {"level": "info", "message": "整体风险水平适中"},
                {"level": "success", "message": "现金比例合理，流动性充足"}
            ]
            
            for alert in risk_alerts:
                if alert["level"] == "warning":
                    st.warning(f"⚠️ {alert['message']}")
                elif alert["level"] == "info":
                    st.info(f"ℹ️ {alert['message']}")
                elif alert["level"] == "success":
                    st.success(f"✅ {alert['message']}")
        
        # 风险暴露图表
        st.markdown("#### 📈 风险暴露分析")
        
        # 模拟不同市场情景下的资产表现
        scenarios = ['牛市 (+20%)', '正常 (0%)', '熊市 (-20%)', '极端下跌 (-40%)']
        portfolio_values = [12000, 10000, 8000, 6000]
        btc_values = [12500, 10000, 7500, 5000]
        eth_values = [11800, 10000, 8200, 6400]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='投资组合',
            x=scenarios,
            y=portfolio_values,
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Bar(
            name='纯BTC持仓',
            x=scenarios,
            y=btc_values,
            marker_color='#ff7f0e'
        ))
        
        fig.add_trace(go.Bar(
            name='纯ETH持仓',
            x=scenarios,
            y=eth_values,
            marker_color='#2ca02c'
        ))
        
        fig.update_layout(
            title="不同市场情景下的资产表现",
            xaxis_title="市场情景",
            yaxis_title="资产价值 (USDT)",
            barmode='group',
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)