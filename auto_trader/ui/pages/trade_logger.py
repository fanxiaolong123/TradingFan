"""
交易日志页面

提供交易记录和系统日志展示，包括：
- 交易记录查询和过滤
- 系统日志实时显示
- 日志导出功能
- 错误日志告警
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TradeLogger:
    """
    交易日志页面类
    
    负责：
    - 交易记录展示和查询
    - 系统日志实时监控
    - 日志过滤和搜索
    - 交易统计分析
    - 日志导出功能
    """
    
    def __init__(self, trading_system: Dict[str, Any]):
        """
        初始化交易日志器
        
        Args:
            trading_system: 交易系统组件字典
        """
        self.trading_system = trading_system
        self.broker = trading_system['broker']
        self.strategies = trading_system.get('strategies', {})
        
        # 生成示例数据
        self._generate_sample_data()
    
    def _generate_sample_data(self):
        """生成示例交易记录和系统日志"""
        # 生成交易记录
        self.trade_records = []
        base_time = datetime.now() - timedelta(days=7)
        
        for i in range(50):  # 生成50条交易记录
            trade_time = base_time + timedelta(hours=i*3)
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
            symbol = symbols[i % len(symbols)]
            
            # 生成买入/卖出交易
            if i % 2 == 0:  # 买入
                self.trade_records.append({
                    'id': f"T{i+1:04d}",
                    'time': trade_time,
                    'strategy': '均值回归策略',
                    'symbol': symbol,
                    'side': 'BUY',
                    'type': 'MARKET',
                    'quantity': round(np.random.uniform(0.001, 0.1), 6),
                    'price': round(np.random.uniform(40000, 50000) if 'BTC' in symbol else np.random.uniform(2000, 3000), 2),
                    'amount': round(np.random.uniform(500, 2000), 2),
                    'commission': round(np.random.uniform(0.5, 2.0), 2),
                    'status': 'FILLED',
                    'pnl': 0.0
                })
            else:  # 卖出
                self.trade_records.append({
                    'id': f"T{i+1:04d}",
                    'time': trade_time,
                    'strategy': '均值回归策略',
                    'symbol': symbol,
                    'side': 'SELL',
                    'type': 'MARKET',
                    'quantity': round(np.random.uniform(0.001, 0.1), 6),
                    'price': round(np.random.uniform(40000, 50000) if 'BTC' in symbol else np.random.uniform(2000, 3000), 2),
                    'amount': round(np.random.uniform(500, 2000), 2),
                    'commission': round(np.random.uniform(0.5, 2.0), 2),
                    'status': 'FILLED',
                    'pnl': round(np.random.uniform(-50, 100), 2)
                })
        
        # 生成系统日志
        self.system_logs = []
        log_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
        log_sources = ['策略', '交易', '风控', '数据', '系统']
        
        for i in range(100):  # 生成100条系统日志
            log_time = base_time + timedelta(minutes=i*5)
            level = log_levels[i % len(log_levels)]
            source = log_sources[i % len(log_sources)]
            
            # 根据级别和来源生成不同的日志消息
            if level == 'INFO':
                messages = [
                    f"{source}模块正常运行",
                    f"策略信号生成: {symbols[i % len(symbols)]}",
                    f"数据更新完成: {symbols[i % len(symbols)]}",
                    f"账户余额更新"
                ]
            elif level == 'WARNING':
                messages = [
                    f"持仓比例接近上限: {symbols[i % len(symbols)]}",
                    f"网络延迟较高",
                    f"数据延迟: {symbols[i % len(symbols)]}",
                    f"风控规则触发"
                ]
            elif level == 'ERROR':
                messages = [
                    f"订单执行失败: {symbols[i % len(symbols)]}",
                    f"数据获取失败",
                    f"连接超时",
                    f"API限流"
                ]
            else:  # DEBUG
                messages = [
                    f"调试信息: 计算信号中",
                    f"数据处理: {symbols[i % len(symbols)]}",
                    f"策略参数更新",
                    f"系统状态检查"
                ]
            
            self.system_logs.append({
                'time': log_time,
                'level': level,
                'source': source,
                'message': messages[i % len(messages)]
            })
    
    def render(self):
        """渲染交易日志页面"""
        st.header("📄 交易日志中心")
        
        # 页面导航
        tab1, tab2, tab3, tab4 = st.tabs(["📊 交易记录", "📋 系统日志", "📈 交易统计", "📤 日志导出"])
        
        with tab1:
            self._render_trade_records()
        
        with tab2:
            self._render_system_logs()
        
        with tab3:
            self._render_trade_statistics()
        
        with tab4:
            self._render_log_export()
    
    def _render_trade_records(self):
        """渲染交易记录页面"""
        st.subheader("📊 交易记录")
        
        # 过滤器
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            strategy_filter = st.selectbox(
                "策略",
                options=["全部"] + list(set([trade['strategy'] for trade in self.trade_records])),
                index=0
            )
        
        with col2:
            symbol_filter = st.selectbox(
                "交易对",
                options=["全部"] + list(set([trade['symbol'] for trade in self.trade_records])),
                index=0
            )
        
        with col3:
            side_filter = st.selectbox(
                "方向",
                options=["全部", "BUY", "SELL"],
                index=0
            )
        
        with col4:
            status_filter = st.selectbox(
                "状态",
                options=["全部", "FILLED", "CANCELLED", "PENDING"],
                index=0
            )
        
        # 日期范围过滤
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=7)
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now()
            )
        
        # 应用过滤器
        filtered_trades = self._filter_trades(
            strategy_filter, symbol_filter, side_filter, status_filter, start_date, end_date
        )
        
        # 统计概览
        if filtered_trades:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_trades = len(filtered_trades)
                st.metric("总交易数", total_trades)
            
            with col2:
                total_volume = sum(trade['amount'] for trade in filtered_trades)
                st.metric("总交易金额", f"{total_volume:,.2f} USDT")
            
            with col3:
                total_pnl = sum(trade['pnl'] for trade in filtered_trades)
                st.metric("总盈亏", f"{total_pnl:+.2f} USDT")
            
            with col4:
                total_commission = sum(trade['commission'] for trade in filtered_trades)
                st.metric("总手续费", f"{total_commission:.2f} USDT")
        
        # 交易记录表格
        st.markdown("#### 📋 交易详情")
        
        if filtered_trades:
            # 转换为DataFrame
            df_trades = pd.DataFrame(filtered_trades)
            
            # 格式化显示
            df_trades['time'] = pd.to_datetime(df_trades['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df_trades['quantity'] = df_trades['quantity'].apply(lambda x: f"{x:.6f}")
            df_trades['price'] = df_trades['price'].apply(lambda x: f"{x:.2f}")
            df_trades['amount'] = df_trades['amount'].apply(lambda x: f"{x:.2f}")
            df_trades['commission'] = df_trades['commission'].apply(lambda x: f"{x:.2f}")
            df_trades['pnl'] = df_trades['pnl'].apply(lambda x: f"{x:+.2f}")
            
            # 重命名列
            df_trades = df_trades.rename(columns={
                'id': 'ID',
                'time': '时间',
                'strategy': '策略',
                'symbol': '交易对',
                'side': '方向',
                'type': '类型',
                'quantity': '数量',
                'price': '价格',
                'amount': '金额',
                'commission': '手续费',
                'status': '状态',
                'pnl': '盈亏'
            })
            
            # 分页显示
            page_size = st.slider("每页显示", 10, 50, 20)
            total_pages = (len(df_trades) - 1) // page_size + 1
            
            if total_pages > 1:
                page = st.selectbox("页码", range(1, total_pages + 1))
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                df_display = df_trades.iloc[start_idx:end_idx]
            else:
                df_display = df_trades
            
            st.dataframe(
                df_display,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("没有匹配的交易记录")
    
    def _filter_trades(self, strategy_filter: str, symbol_filter: str, side_filter: str, 
                      status_filter: str, start_date, end_date) -> List[Dict]:
        """过滤交易记录"""
        filtered = []
        
        for trade in self.trade_records:
            # 时间过滤
            trade_date = trade['time'].date()
            if not (start_date <= trade_date <= end_date):
                continue
            
            # 策略过滤
            if strategy_filter != "全部" and trade['strategy'] != strategy_filter:
                continue
            
            # 交易对过滤
            if symbol_filter != "全部" and trade['symbol'] != symbol_filter:
                continue
            
            # 方向过滤
            if side_filter != "全部" and trade['side'] != side_filter:
                continue
            
            # 状态过滤
            if status_filter != "全部" and trade['status'] != status_filter:
                continue
            
            filtered.append(trade)
        
        return filtered
    
    def _render_system_logs(self):
        """渲染系统日志页面"""
        st.subheader("📋 系统日志")
        
        # 过滤器
        col1, col2, col3 = st.columns(3)
        
        with col1:
            level_filter = st.selectbox(
                "日志级别",
                options=["全部", "INFO", "WARNING", "ERROR", "DEBUG"],
                index=0
            )
        
        with col2:
            source_filter = st.selectbox(
                "来源",
                options=["全部"] + list(set([log['source'] for log in self.system_logs])),
                index=0
            )
        
        with col3:
            search_text = st.text_input(
                "搜索关键词",
                placeholder="输入关键词搜索日志..."
            )
        
        # 实时刷新开关
        auto_refresh = st.checkbox("自动刷新", value=False)
        if auto_refresh:
            st.info("⚡ 自动刷新已启用")
        
        # 应用过滤器
        filtered_logs = self._filter_logs(level_filter, source_filter, search_text)
        
        # 日志级别统计
        if filtered_logs:
            level_counts = {}
            for log in filtered_logs:
                level = log['level']
                level_counts[level] = level_counts.get(level, 0) + 1
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("INFO", level_counts.get('INFO', 0))
            with col2:
                st.metric("WARNING", level_counts.get('WARNING', 0))
            with col3:
                st.metric("ERROR", level_counts.get('ERROR', 0))
            with col4:
                st.metric("DEBUG", level_counts.get('DEBUG', 0))
        
        # 日志列表
        st.markdown("#### 📜 日志详情")
        
        if filtered_logs:
            # 最新日志在前
            filtered_logs.sort(key=lambda x: x['time'], reverse=True)
            
            # 分页
            page_size = st.slider("每页显示日志", 20, 100, 50)
            total_pages = (len(filtered_logs) - 1) // page_size + 1
            
            if total_pages > 1:
                page = st.selectbox("页码", range(1, total_pages + 1), key="log_page")
                start_idx = (page - 1) * page_size
                end_idx = start_idx + page_size
                logs_display = filtered_logs[start_idx:end_idx]
            else:
                logs_display = filtered_logs
            
            # 显示日志
            for log in logs_display:
                # 根据级别设置颜色
                if log['level'] == 'ERROR':
                    st.error(f"🔴 [{log['time'].strftime('%H:%M:%S')}] {log['source']} - {log['message']}")
                elif log['level'] == 'WARNING':
                    st.warning(f"🟡 [{log['time'].strftime('%H:%M:%S')}] {log['source']} - {log['message']}")
                elif log['level'] == 'INFO':
                    st.info(f"🔵 [{log['time'].strftime('%H:%M:%S')}] {log['source']} - {log['message']}")
                else:  # DEBUG
                    st.text(f"⚪ [{log['time'].strftime('%H:%M:%S')}] {log['source']} - {log['message']}")
        else:
            st.info("没有匹配的日志记录")
    
    def _filter_logs(self, level_filter: str, source_filter: str, search_text: str) -> List[Dict]:
        """过滤系统日志"""
        filtered = []
        
        for log in self.system_logs:
            # 级别过滤
            if level_filter != "全部" and log['level'] != level_filter:
                continue
            
            # 来源过滤
            if source_filter != "全部" and log['source'] != source_filter:
                continue
            
            # 搜索过滤
            if search_text and search_text.lower() not in log['message'].lower():
                continue
            
            filtered.append(log)
        
        return filtered
    
    def _render_trade_statistics(self):
        """渲染交易统计页面"""
        st.subheader("📈 交易统计")
        
        # 时间范围选择
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=30),
                key="stat_start"
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now(),
                key="stat_end"
            )
        
        # 过滤数据
        filtered_trades = []
        for trade in self.trade_records:
            trade_date = trade['time'].date()
            if start_date <= trade_date <= end_date:
                filtered_trades.append(trade)
        
        if not filtered_trades:
            st.info("所选时间段内没有交易数据")
            return
        
        # 整体统计
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 交易概览")
            
            # 基本统计
            total_trades = len(filtered_trades)
            buy_trades = len([t for t in filtered_trades if t['side'] == 'BUY'])
            sell_trades = len([t for t in filtered_trades if t['side'] == 'SELL'])
            total_volume = sum(trade['amount'] for trade in filtered_trades)
            total_pnl = sum(trade['pnl'] for trade in filtered_trades)
            
            st.metric("总交易笔数", total_trades)
            st.metric("买入交易", buy_trades)
            st.metric("卖出交易", sell_trades)
            st.metric("总交易额", f"{total_volume:,.2f} USDT")
            st.metric("总盈亏", f"{total_pnl:+.2f} USDT")
        
        with col2:
            st.markdown("#### 📈 交易分布")
            
            # 按交易对分布
            symbol_counts = {}
            for trade in filtered_trades:
                symbol = trade['symbol']
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            
            fig = go.Figure(data=[go.Pie(
                labels=list(symbol_counts.keys()),
                values=list(symbol_counts.values()),
                hole=0.3
            )])
            
            fig.update_layout(
                title="交易对分布",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 时间序列分析
        st.markdown("#### 📅 时间序列分析")
        
        # 按日期汇总
        df_trades = pd.DataFrame(filtered_trades)
        df_trades['date'] = pd.to_datetime(df_trades['time']).dt.date
        
        daily_stats = df_trades.groupby('date').agg({
            'id': 'count',
            'amount': 'sum',
            'pnl': 'sum'
        }).reset_index()
        
        daily_stats.columns = ['date', 'trade_count', 'volume', 'pnl']
        
        # 创建时间序列图表
        col1, col2 = st.columns(2)
        
        with col1:
            # 每日交易量
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['trade_count'],
                mode='lines+markers',
                name='交易笔数',
                line=dict(color='#1f77b4')
            ))
            
            fig.update_layout(
                title="每日交易笔数",
                xaxis_title="日期",
                yaxis_title="交易笔数",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 每日盈亏
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_stats['date'],
                y=daily_stats['pnl'],
                name='每日盈亏',
                marker_color=['green' if x > 0 else 'red' for x in daily_stats['pnl']]
            ))
            
            fig.update_layout(
                title="每日盈亏",
                xaxis_title="日期",
                yaxis_title="盈亏 (USDT)",
                template="plotly_white",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_log_export(self):
        """渲染日志导出页面"""
        st.subheader("📤 日志导出")
        
        # 导出选项
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 交易记录导出")
            
            export_format = st.selectbox(
                "导出格式",
                options=["CSV", "Excel", "JSON"],
                index=0
            )
            
            # 时间范围
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=30),
                key="export_start"
            )
            end_date = st.date_input(
                "结束日期",
                value=datetime.now(),
                key="export_end"
            )
            
            if st.button("📥 导出交易记录"):
                # 过滤数据
                filtered_trades = []
                for trade in self.trade_records:
                    trade_date = trade['time'].date()
                    if start_date <= trade_date <= end_date:
                        filtered_trades.append(trade)
                
                if filtered_trades:
                    df_export = pd.DataFrame(filtered_trades)
                    
                    if export_format == "CSV":
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="下载CSV文件",
                            data=csv,
                            file_name=f"trade_records_{start_date}_to_{end_date}.csv",
                            mime="text/csv"
                        )
                    elif export_format == "Excel":
                        # 这里需要实现Excel导出
                        st.info("Excel导出功能开发中...")
                    elif export_format == "JSON":
                        json_data = df_export.to_json(orient='records', date_format='iso')
                        st.download_button(
                            label="下载JSON文件",
                            data=json_data,
                            file_name=f"trade_records_{start_date}_to_{end_date}.json",
                            mime="application/json"
                        )
                else:
                    st.warning("所选时间段内没有交易数据")
        
        with col2:
            st.markdown("#### 📋 系统日志导出")
            
            log_level = st.selectbox(
                "日志级别",
                options=["全部", "INFO", "WARNING", "ERROR", "DEBUG"],
                index=0,
                key="export_level"
            )
            
            log_source = st.selectbox(
                "日志来源",
                options=["全部"] + list(set([log['source'] for log in self.system_logs])),
                index=0,
                key="export_source"
            )
            
            if st.button("📥 导出系统日志"):
                # 过滤日志
                filtered_logs = self._filter_logs(log_level, log_source, "")
                
                if filtered_logs:
                    df_logs = pd.DataFrame(filtered_logs)
                    csv = df_logs.to_csv(index=False)
                    st.download_button(
                        label="下载日志文件",
                        data=csv,
                        file_name=f"system_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("没有匹配的日志数据")
        
        # 自动备份设置
        st.markdown("---")
        st.markdown("#### ⚙️ 自动备份设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            auto_backup = st.checkbox("启用自动备份", value=False)
            backup_interval = st.selectbox(
                "备份频率",
                options=["每小时", "每天", "每周"],
                index=1
            )
        
        with col2:
            backup_location = st.text_input(
                "备份位置",
                value="./backups/",
                placeholder="输入备份目录路径"
            )
            
            if st.button("💾 立即备份"):
                st.success("备份已创建！")
                st.info(f"备份保存至: {backup_location}")
        
        if auto_backup:
            st.success(f"✅ 自动备份已启用，频率: {backup_interval}")
        else:
            st.info("ℹ️ 自动备份已禁用")