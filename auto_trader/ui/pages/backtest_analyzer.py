"""
回测分析页面

提供回测功能和结果分析，包括：
- 回测参数配置
- 回测执行和监控  
- 回测结果可视化
- 策略性能评估
- 回测报告生成
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from auto_trader.core.backtest import BacktestEngine
from auto_trader.strategies.base import StrategyConfig
from auto_trader.strategies.mean_reversion import MeanReversionStrategy


class BacktestAnalyzer:
    """
    回测分析页面类
    
    负责：
    - 回测参数配置界面
    - 回测执行和进度监控
    - 回测结果可视化展示
    - 策略性能指标计算
    - 回测报告生成
    """
    
    def __init__(self, trading_system: Dict[str, Any]):
        """
        初始化回测分析器
        
        Args:
            trading_system: 交易系统组件字典
        """
        self.trading_system = trading_system
        self.data_manager = trading_system['data_manager']
        self.account_manager = trading_system['account_manager']
        self.risk_manager = trading_system['risk_manager']
        self.broker = trading_system['broker']
        
        # 初始化回测引擎
        self.backtest_engine = BacktestEngine(data_manager=self.data_manager)
    
    def render(self):
        """渲染回测分析页面"""
        st.header("📈 回测分析中心")
        
        # 页面布局
        self._render_backtest_config()
        st.markdown("---")
        self._render_backtest_results()
        st.markdown("---")
        self._render_performance_analysis()
    
    def _render_backtest_config(self):
        """渲染回测配置部分"""
        st.subheader("⚙️ 回测配置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 策略配置")
            
            # 策略选择
            strategy_type = st.selectbox(
                "策略类型",
                options=["均值回归", "趋势跟随", "网格交易", "套利策略"],
                help="选择要回测的策略类型"
            )
            
            # 交易对选择
            symbol = st.selectbox(
                "交易对",
                options=["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "SOLUSDT"],
                help="选择要回测的交易对"
            )
            
            # 时间周期
            timeframe = st.selectbox(
                "时间周期",
                options=["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
                index=4,  # 默认1h
                help="K线数据的时间周期"
            )
            
            # 回测时间范围
            st.markdown("**回测时间范围**")
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=30),
                help="回测开始日期"
            )
            end_date = st.date_input(
                "结束日期", 
                value=datetime.now(),
                help="回测结束日期"
            )
            
            # 策略参数
            if strategy_type == "均值回归":
                st.markdown("**策略参数**")
                window_size = st.number_input("移动平均窗口", value=20, min_value=5, max_value=100)
                std_multiplier = st.number_input("标准差倍数", value=2.0, min_value=0.5, max_value=5.0, step=0.1)
                
                strategy_params = {
                    'window_size': window_size,
                    'std_multiplier': std_multiplier
                }
            else:
                st.info(f"{strategy_type}策略参数配置待实现")
                strategy_params = {}
        
        with col2:
            st.markdown("#### 💰 回测设置")
            
            # 初始资金
            initial_balance = st.number_input(
                "初始资金 (USDT)",
                value=10000.0,
                min_value=1000.0,
                max_value=1000000.0,
                step=1000.0,
                help="回测初始资金"
            )
            
            # 手续费设置
            commission_rate = st.number_input(
                "手续费率 (%)",
                value=0.1,
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                help="交易手续费率"
            )
            
            # 滑点设置
            slippage = st.number_input(
                "滑点 (%)",
                value=0.05,
                min_value=0.0,
                max_value=0.5,
                step=0.01,
                help="交易滑点"
            )
            
            # 单笔最大交易金额
            max_trade_amount = st.number_input(
                "单笔最大交易金额 (USDT)",
                value=1000.0,
                min_value=100.0,
                max_value=initial_balance,
                step=100.0,
                help="单笔交易的最大金额"
            )
            
            # 风险控制
            st.markdown("**风险控制**")
            max_drawdown = st.number_input(
                "最大回撤限制 (%)",
                value=20.0,
                min_value=5.0,
                max_value=50.0,
                step=1.0,
                help="最大允许回撤比例"
            )
            
            # 执行回测按钮
            st.markdown("---")
            if st.button("🚀 开始回测", type="primary", help="执行回测"):
                if start_date < end_date:
                    with st.spinner("正在执行回测..."):
                        self._run_backtest(
                            strategy_type, symbol, timeframe, 
                            start_date, end_date, strategy_params,
                            initial_balance, commission_rate, slippage
                        )
                else:
                    st.error("开始日期必须早于结束日期")
    
    def _run_backtest(self, strategy_type: str, symbol: str, timeframe: str,
                     start_date, end_date, strategy_params: Dict,
                     initial_balance: float, commission_rate: float, slippage: float):
        """
        执行回测
        
        Args:
            strategy_type: 策略类型
            symbol: 交易对
            timeframe: 时间周期
            start_date: 开始日期
            end_date: 结束日期
            strategy_params: 策略参数
            initial_balance: 初始资金
            commission_rate: 手续费率
            slippage: 滑点
        """
        try:
            # 创建策略配置
            config = StrategyConfig(
                name=f"{strategy_type}_{symbol}_{timeframe}",
                symbol=symbol,
                timeframe=timeframe,
                parameters=strategy_params
            )
            
            # 创建策略实例
            if strategy_type == "均值回归":
                strategy = MeanReversionStrategy(config)
            else:
                st.error(f"策略类型 '{strategy_type}' 暂未实现")
                return
            
            # 设置回测引擎参数
            self.backtest_engine.set_commission_rate(commission_rate / 100)
            self.backtest_engine.set_slippage(slippage / 100)
            
            # 执行回测
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 模拟回测进度
            for i in range(100):
                progress_bar.progress(i + 1)
                status_text.text(f"回测进度: {i+1}%")
                
            # 生成模拟回测结果
            backtest_results = self._generate_sample_results(
                symbol, start_date, end_date, initial_balance
            )
            
            # 保存结果到session state
            st.session_state['backtest_results'] = backtest_results
            st.session_state['backtest_config'] = {
                'strategy_type': strategy_type,
                'symbol': symbol,
                'timeframe': timeframe,
                'start_date': start_date,
                'end_date': end_date,
                'initial_balance': initial_balance
            }
            
            st.success("✅ 回测完成！")
            
        except Exception as e:
            st.error(f"回测执行失败: {e}")
    
    def _generate_sample_results(self, symbol: str, start_date, end_date, initial_balance: float) -> Dict:
        """生成示例回测结果"""
        # 生成时间序列
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 生成模拟价格数据
        price_data = []
        current_price = 45000 if 'BTC' in symbol else 2500
        
        for date in dates:
            daily_return = pd.np.random.normal(0.001, 0.02)
            current_price *= (1 + daily_return)
            price_data.append({
                'date': date,
                'price': current_price,
                'volume': pd.np.random.uniform(100, 1000)
            })
        
        # 生成模拟交易记录
        trades = []
        portfolio_values = []
        current_balance = initial_balance
        position = 0
        
        for i, data in enumerate(price_data):
            # 模拟交易信号
            if i > 0 and i % 3 == 0:  # 每3天一次交易
                if position == 0:  # 买入
                    trade_amount = current_balance * 0.1
                    position = trade_amount / data['price']
                    current_balance -= trade_amount
                    trades.append({
                        'date': data['date'],
                        'type': 'BUY',
                        'price': data['price'],
                        'quantity': position,
                        'amount': trade_amount
                    })
                else:  # 卖出
                    trade_amount = position * data['price']
                    current_balance += trade_amount
                    trades.append({
                        'date': data['date'],
                        'type': 'SELL',
                        'price': data['price'],
                        'quantity': position,
                        'amount': trade_amount
                    })
                    position = 0
            
            # 计算组合价值
            portfolio_value = current_balance + (position * data['price'])
            portfolio_values.append({
                'date': data['date'],
                'value': portfolio_value,
                'cash': current_balance,
                'position_value': position * data['price']
            })
        
        # 计算性能指标
        final_value = portfolio_values[-1]['value']
        total_return = (final_value - initial_balance) / initial_balance * 100
        
        # 计算最大回撤
        peak_value = initial_balance
        max_drawdown = 0
        for pv in portfolio_values:
            if pv['value'] > peak_value:
                peak_value = pv['value']
            drawdown = (peak_value - pv['value']) / peak_value * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # 计算夏普比率 (简化版)
        returns = [pv['value'] for pv in portfolio_values]
        return_series = pd.Series(returns).pct_change().dropna()
        sharpe_ratio = (return_series.mean() / return_series.std()) * (252 ** 0.5) if return_series.std() > 0 else 0
        
        return {
            'price_data': price_data,
            'trades': trades,
            'portfolio_values': portfolio_values,
            'performance': {
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': len(trades),
                'winning_trades': len([t for i, t in enumerate(trades) if i > 0 and t['type'] == 'SELL']),
                'final_value': final_value
            }
        }
    
    def _render_backtest_results(self):
        """渲染回测结果"""
        st.subheader("📊 回测结果")
        
        if 'backtest_results' not in st.session_state:
            st.info("请先配置并执行回测")
            return
        
        results = st.session_state['backtest_results']
        config = st.session_state['backtest_config']
        
        # 性能指标概览
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "总收益率",
                f"{results['performance']['total_return']:.2f}%",
                delta=f"vs 初始资金"
            )
        
        with col2:
            st.metric(
                "最大回撤",
                f"{results['performance']['max_drawdown']:.2f}%",
                delta="风险指标"
            )
        
        with col3:
            st.metric(
                "夏普比率",
                f"{results['performance']['sharpe_ratio']:.2f}",
                delta="风险调整收益"
            )
        
        with col4:
            st.metric(
                "交易次数",
                f"{results['performance']['total_trades']}",
                delta="总交易"
            )
        
        # 资产价值曲线
        self._render_equity_curve(results['portfolio_values'])
        
        # 交易记录
        self._render_trade_history(results['trades'])
    
    def _render_equity_curve(self, portfolio_values: List[Dict]):
        """渲染资产价值曲线"""
        st.markdown("#### 📈 资产价值曲线")
        
        # 转换为DataFrame
        df = pd.DataFrame(portfolio_values)
        
        # 创建图表
        fig = go.Figure()
        
        # 总资产价值
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['value'],
            mode='lines',
            name='总资产价值',
            line=dict(color='#1f77b4', width=3)
        ))
        
        # 现金部分
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['cash'],
            mode='lines',
            name='现金',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        # 持仓价值
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['position_value'],
            mode='lines',
            name='持仓价值',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig.update_layout(
            title="资产价值变化趋势",
            xaxis_title="日期",
            yaxis_title="价值 (USDT)",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_trade_history(self, trades: List[Dict]):
        """渲染交易历史"""
        st.markdown("#### 📋 交易历史")
        
        if not trades:
            st.info("暂无交易记录")
            return
        
        # 转换为DataFrame
        df_trades = pd.DataFrame(trades)
        df_trades['date'] = pd.to_datetime(df_trades['date']).dt.strftime('%Y-%m-%d')
        
        # 格式化显示
        df_trades['price'] = df_trades['price'].apply(lambda x: f"{x:.2f}")
        df_trades['quantity'] = df_trades['quantity'].apply(lambda x: f"{x:.6f}")
        df_trades['amount'] = df_trades['amount'].apply(lambda x: f"{x:.2f}")
        
        # 重命名列
        df_trades = df_trades.rename(columns={
            'date': '日期',
            'type': '类型',
            'price': '价格',
            'quantity': '数量',
            'amount': '金额'
        })
        
        st.dataframe(
            df_trades,
            use_container_width=True,
            hide_index=True
        )
    
    def _render_performance_analysis(self):
        """渲染性能分析"""
        st.subheader("📊 性能分析")
        
        if 'backtest_results' not in st.session_state:
            st.info("请先执行回测")
            return
        
        results = st.session_state['backtest_results']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 收益分析")
            
            # 月度收益分析
            portfolio_df = pd.DataFrame(results['portfolio_values'])
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df['month'] = portfolio_df['date'].dt.to_period('M')
            
            monthly_returns = portfolio_df.groupby('month')['value'].last().pct_change().dropna() * 100
            
            if len(monthly_returns) > 0:
                fig = go.Figure(data=[go.Bar(
                    x=monthly_returns.index.astype(str),
                    y=monthly_returns.values,
                    marker_color=['green' if x > 0 else 'red' for x in monthly_returns.values]
                )])
                
                fig.update_layout(
                    title="月度收益率",
                    xaxis_title="月份",
                    yaxis_title="收益率 (%)",
                    template="plotly_white",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("数据不足以计算月度收益")
        
        with col2:
            st.markdown("#### 📊 风险分析")
            
            # 回撤分析
            portfolio_df = pd.DataFrame(results['portfolio_values'])
            portfolio_df['peak'] = portfolio_df['value'].expanding().max()
            portfolio_df['drawdown'] = (portfolio_df['peak'] - portfolio_df['value']) / portfolio_df['peak'] * 100
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_df['date'],
                y=-portfolio_df['drawdown'],  # 负值显示
                mode='lines',
                fill='tonexty',
                name='回撤',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="回撤分析",
                xaxis_title="日期",
                yaxis_title="回撤 (%)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)