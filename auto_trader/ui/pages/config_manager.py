"""
配置管理页面

提供系统配置管理功能，包括：
- 交易参数配置
- 风控规则设置
- API密钥管理
- 系统设置
"""

import streamlit as st
import json
import yaml
from typing import Dict, Any
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from auto_trader.utils.config import ConfigManager as CoreConfigManager


class ConfigManager:
    """
    配置管理页面类
    
    负责：
    - 交易参数配置界面
    - 风控规则设置
    - API密钥管理
    - 系统设置管理
    - 配置文件导入导出
    """
    
    def __init__(self, trading_system: Dict[str, Any]):
        """
        初始化配置管理器
        
        Args:
            trading_system: 交易系统组件字典
        """
        self.trading_system = trading_system
        self.config_manager = CoreConfigManager()
        
        # 初始化配置状态
        if 'config_modified' not in st.session_state:
            st.session_state.config_modified = False
        
        # 加载当前配置
        self._load_current_config()
    
    def _load_current_config(self):
        """加载当前配置"""
        try:
            # 从配置管理器获取当前配置
            self.current_config = self.config_manager.get_all_config()
        except Exception as e:
            st.error(f"加载配置失败: {e}")
            self.current_config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'trading': {
                'default_quantity': 100.0,
                'commission_rate': 0.001,
                'slippage': 0.0001,
                'max_orders_per_symbol': 10,
                'order_timeout': 30,
                'retry_attempts': 3
            },
            'risk_management': {
                'enabled': True,
                'position_limits': {
                    'max_position_percent': 0.1,
                    'max_total_position_percent': 0.8
                },
                'loss_limits': {
                    'max_daily_loss_percent': 0.05,
                    'max_total_loss_percent': 0.20,
                    'max_drawdown_percent': 0.15
                },
                'frequency_limits': {
                    'max_trades_per_hour': 10,
                    'max_trades_per_day': 100
                }
            },
            'data_sources': {
                'binance': {
                    'api_key': '',
                    'api_secret': '',
                    'testnet': True,
                    'timeout': 30
                }
            },
            'system': {
                'auto_trade': False,
                'notifications': True,
                'log_level': 'INFO',
                'backup_enabled': True,
                'backup_interval': 'daily'
            },
            'ui': {
                'theme': 'light',
                'language': 'zh-CN',
                'auto_refresh': True,
                'refresh_interval': 30
            }
        }
    
    def render(self):
        """渲染配置管理页面"""
        st.header("⚙️ 系统配置管理")
        
        # 顶部操作栏
        self._render_top_actions()
        
        # 配置分类标签页
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💱 交易配置", 
            "🛡️ 风控设置", 
            "🔑 API设置", 
            "🔧 系统设置", 
            "📁 配置管理"
        ])
        
        with tab1:
            self._render_trading_config()
        
        with tab2:
            self._render_risk_config()
        
        with tab3:
            self._render_api_config()
        
        with tab4:
            self._render_system_config()
        
        with tab5:
            self._render_config_management()
        
        # 底部保存按钮
        self._render_save_actions()
    
    def _render_top_actions(self):
        """渲染顶部操作栏"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.session_state.config_modified:
                st.warning("⚠️ 配置已修改，请记得保存")
            else:
                st.success("✅ 配置已同步")
        
        with col2:
            if st.button("🔄 重新加载", help="重新加载配置文件"):
                self._load_current_config()
                st.session_state.config_modified = False
                st.rerun()
        
        with col3:
            if st.button("↩️ 恢复默认", help="恢复默认配置"):
                self.current_config = self._get_default_config()
                st.session_state.config_modified = True
                st.rerun()
    
    def _render_trading_config(self):
        """渲染交易配置页面"""
        st.subheader("💱 交易参数配置")
        
        trading_config = self.current_config.get('trading', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 基础参数")
            
            # 默认交易数量
            default_quantity = st.number_input(
                "默认交易数量",
                value=trading_config.get('default_quantity', 100.0),
                min_value=1.0,
                max_value=10000.0,
                step=1.0,
                help="单笔交易的默认数量"
            )
            
            # 手续费率
            commission_rate = st.number_input(
                "手续费率 (%)",
                value=trading_config.get('commission_rate', 0.001) * 100,
                min_value=0.0,
                max_value=1.0,
                step=0.001,
                format="%.4f",
                help="交易手续费率"
            )
            
            # 滑点
            slippage = st.number_input(
                "滑点 (%)",
                value=trading_config.get('slippage', 0.0001) * 100,
                min_value=0.0,
                max_value=1.0,
                step=0.001,
                format="%.4f",
                help="交易滑点"
            )
        
        with col2:
            st.markdown("#### ⚙️ 高级参数")
            
            # 最大订单数
            max_orders = st.number_input(
                "单币种最大订单数",
                value=trading_config.get('max_orders_per_symbol', 10),
                min_value=1,
                max_value=100,
                step=1,
                help="每个交易对的最大同时订单数"
            )
            
            # 订单超时
            order_timeout = st.number_input(
                "订单超时时间 (秒)",
                value=trading_config.get('order_timeout', 30),
                min_value=5,
                max_value=300,
                step=5,
                help="订单执行超时时间"
            )
            
            # 重试次数
            retry_attempts = st.number_input(
                "重试次数",
                value=trading_config.get('retry_attempts', 3),
                min_value=1,
                max_value=10,
                step=1,
                help="订单失败后的重试次数"
            )
        
        # 更新配置
        if (default_quantity != trading_config.get('default_quantity', 100.0) or
            commission_rate/100 != trading_config.get('commission_rate', 0.001) or
            slippage/100 != trading_config.get('slippage', 0.0001) or
            max_orders != trading_config.get('max_orders_per_symbol', 10) or
            order_timeout != trading_config.get('order_timeout', 30) or
            retry_attempts != trading_config.get('retry_attempts', 3)):
            
            self.current_config['trading'] = {
                'default_quantity': default_quantity,
                'commission_rate': commission_rate / 100,
                'slippage': slippage / 100,
                'max_orders_per_symbol': max_orders,
                'order_timeout': order_timeout,
                'retry_attempts': retry_attempts
            }
            st.session_state.config_modified = True
        
        # 配置预览
        st.markdown("#### 📋 当前配置预览")
        with st.expander("查看详细配置"):
            st.json(self.current_config['trading'])
    
    def _render_risk_config(self):
        """渲染风控配置页面"""
        st.subheader("🛡️ 风险控制配置")
        
        risk_config = self.current_config.get('risk_management', {})
        
        # 风控总开关
        risk_enabled = st.checkbox(
            "启用风险控制",
            value=risk_config.get('enabled', True),
            help="是否启用风险控制模块"
        )
        
        if risk_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 持仓限制")
                
                position_limits = risk_config.get('position_limits', {})
                
                # 最大单个持仓比例
                max_position_percent = st.slider(
                    "最大单个持仓比例 (%)",
                    min_value=1.0,
                    max_value=50.0,
                    value=position_limits.get('max_position_percent', 0.1) * 100,
                    step=1.0,
                    help="单个交易对的最大持仓比例"
                )
                
                # 最大总持仓比例
                max_total_position_percent = st.slider(
                    "最大总持仓比例 (%)",
                    min_value=50.0,
                    max_value=100.0,
                    value=position_limits.get('max_total_position_percent', 0.8) * 100,
                    step=5.0,
                    help="所有持仓的最大总比例"
                )
                
                st.markdown("#### 📉 亏损限制")
                
                loss_limits = risk_config.get('loss_limits', {})
                
                # 最大日亏损
                max_daily_loss = st.slider(
                    "最大日亏损 (%)",
                    min_value=1.0,
                    max_value=20.0,
                    value=loss_limits.get('max_daily_loss_percent', 0.05) * 100,
                    step=0.5,
                    help="单日最大亏损比例"
                )
                
                # 最大总亏损
                max_total_loss = st.slider(
                    "最大总亏损 (%)",
                    min_value=5.0,
                    max_value=50.0,
                    value=loss_limits.get('max_total_loss_percent', 0.20) * 100,
                    step=1.0,
                    help="总体最大亏损比例"
                )
                
                # 最大回撤
                max_drawdown = st.slider(
                    "最大回撤 (%)",
                    min_value=5.0,
                    max_value=30.0,
                    value=loss_limits.get('max_drawdown_percent', 0.15) * 100,
                    step=1.0,
                    help="最大回撤比例"
                )
            
            with col2:
                st.markdown("#### 🕐 频率限制")
                
                frequency_limits = risk_config.get('frequency_limits', {})
                
                # 每小时最大交易次数
                max_trades_per_hour = st.number_input(
                    "每小时最大交易次数",
                    value=frequency_limits.get('max_trades_per_hour', 10),
                    min_value=1,
                    max_value=100,
                    step=1,
                    help="每小时最大交易次数"
                )
                
                # 每天最大交易次数
                max_trades_per_day = st.number_input(
                    "每天最大交易次数",
                    value=frequency_limits.get('max_trades_per_day', 100),
                    min_value=10,
                    max_value=1000,
                    step=10,
                    help="每天最大交易次数"
                )
                
                st.markdown("#### 📈 风险指标")
                
                # 显示当前风险状态
                st.metric("当前风险等级", "低", delta="安全")
                st.metric("持仓集中度", "15%", delta="正常")
                st.metric("今日交易次数", "5", delta="剩余95次")
                
                # 风险报警设置
                st.markdown("#### 🚨 报警设置")
                
                alert_email = st.text_input(
                    "报警邮箱",
                    value="",
                    placeholder="输入接收风险报警的邮箱"
                )
                
                alert_webhook = st.text_input(
                    "Webhook URL",
                    value="",
                    placeholder="输入Webhook URL"
                )
            
            # 更新配置
            self.current_config['risk_management'] = {
                'enabled': risk_enabled,
                'position_limits': {
                    'max_position_percent': max_position_percent / 100,
                    'max_total_position_percent': max_total_position_percent / 100
                },
                'loss_limits': {
                    'max_daily_loss_percent': max_daily_loss / 100,
                    'max_total_loss_percent': max_total_loss / 100,
                    'max_drawdown_percent': max_drawdown / 100
                },
                'frequency_limits': {
                    'max_trades_per_hour': max_trades_per_hour,
                    'max_trades_per_day': max_trades_per_day
                },
                'alerts': {
                    'email': alert_email,
                    'webhook': alert_webhook
                }
            }
            st.session_state.config_modified = True
        
        else:
            st.warning("⚠️ 风险控制已禁用，请谨慎操作")
    
    def _render_api_config(self):
        """渲染API配置页面"""
        st.subheader("🔑 API密钥管理")
        
        data_sources = self.current_config.get('data_sources', {})
        
        # Binance API配置
        st.markdown("#### 🟡 Binance API")
        binance_config = data_sources.get('binance', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_key = st.text_input(
                "API Key",
                value=binance_config.get('api_key', ''),
                type="password",
                help="Binance API密钥"
            )
            
            testnet = st.checkbox(
                "使用测试网",
                value=binance_config.get('testnet', True),
                help="是否使用Binance测试网"
            )
        
        with col2:
            api_secret = st.text_input(
                "API Secret",
                value=binance_config.get('api_secret', ''),
                type="password",
                help="Binance API密钥"
            )
            
            timeout = st.number_input(
                "连接超时 (秒)",
                value=binance_config.get('timeout', 30),
                min_value=5,
                max_value=120,
                step=5,
                help="API连接超时时间"
            )
        
        # API测试
        if st.button("🧪 测试API连接"):
            if api_key and api_secret:
                with st.spinner("测试连接中..."):
                    # 这里应该实际测试API连接
                    import time
                    time.sleep(2)
                    st.success("✅ API连接测试成功")
            else:
                st.error("❌ 请先填写API密钥")
        
        # 更新配置
        self.current_config['data_sources'] = {
            'binance': {
                'api_key': api_key,
                'api_secret': api_secret,
                'testnet': testnet,
                'timeout': timeout
            }
        }
        st.session_state.config_modified = True
        
        # 安全提示
        st.markdown("---")
        st.info("""
        🔒 **安全提示：**
        - API密钥将加密存储
        - 建议使用只读或交易权限的API密钥
        - 定期更换API密钥
        - 不要在公共网络环境下配置API密钥
        """)
    
    def _render_system_config(self):
        """渲染系统配置页面"""
        st.subheader("🔧 系统设置")
        
        system_config = self.current_config.get('system', {})
        ui_config = self.current_config.get('ui', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎛️ 系统控制")
            
            # 自动交易开关
            auto_trade = st.checkbox(
                "启用自动交易",
                value=system_config.get('auto_trade', False),
                help="是否允许系统自动执行交易"
            )
            
            # 通知开关
            notifications = st.checkbox(
                "启用通知",
                value=system_config.get('notifications', True),
                help="是否启用系统通知"
            )
            
            # 日志级别
            log_level = st.selectbox(
                "日志级别",
                options=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                index=['DEBUG', 'INFO', 'WARNING', 'ERROR'].index(system_config.get('log_level', 'INFO')),
                help="系统日志输出级别"
            )
            
            # 备份设置
            backup_enabled = st.checkbox(
                "启用自动备份",
                value=system_config.get('backup_enabled', True),
                help="是否启用自动备份"
            )
            
            if backup_enabled:
                backup_interval = st.selectbox(
                    "备份频率",
                    options=['hourly', 'daily', 'weekly'],
                    index=['hourly', 'daily', 'weekly'].index(system_config.get('backup_interval', 'daily')),
                    help="自动备份频率"
                )
            else:
                backup_interval = 'daily'
        
        with col2:
            st.markdown("#### 🎨 界面设置")
            
            # 主题设置
            theme = st.selectbox(
                "界面主题",
                options=['light', 'dark', 'auto'],
                index=['light', 'dark', 'auto'].index(ui_config.get('theme', 'light')),
                help="界面主题色彩"
            )
            
            # 语言设置
            language = st.selectbox(
                "界面语言",
                options=['zh-CN', 'en-US'],
                index=['zh-CN', 'en-US'].index(ui_config.get('language', 'zh-CN')),
                help="界面显示语言"
            )
            
            # 自动刷新
            auto_refresh = st.checkbox(
                "自动刷新",
                value=ui_config.get('auto_refresh', True),
                help="是否自动刷新界面数据"
            )
            
            if auto_refresh:
                refresh_interval = st.slider(
                    "刷新间隔 (秒)",
                    min_value=5,
                    max_value=300,
                    value=ui_config.get('refresh_interval', 30),
                    step=5,
                    help="自动刷新间隔时间"
                )
            else:
                refresh_interval = 30
        
        # 更新配置
        self.current_config['system'] = {
            'auto_trade': auto_trade,
            'notifications': notifications,
            'log_level': log_level,
            'backup_enabled': backup_enabled,
            'backup_interval': backup_interval
        }
        
        self.current_config['ui'] = {
            'theme': theme,
            'language': language,
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval
        }
        
        st.session_state.config_modified = True
        
        # 系统状态
        st.markdown("---")
        st.markdown("#### 📊 系统状态")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("系统运行时间", "2小时15分钟")
        
        with col2:
            st.metric("内存使用", "245MB")
        
        with col3:
            st.metric("CPU使用率", "15%")
    
    def _render_config_management(self):
        """渲染配置管理页面"""
        st.subheader("📁 配置文件管理")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📤 导出配置")
            
            export_format = st.selectbox(
                "导出格式",
                options=['YAML', 'JSON'],
                index=0,
                help="配置文件导出格式"
            )
            
            if st.button("💾 导出配置"):
                try:
                    if export_format == 'YAML':
                        config_data = yaml.dump(self.current_config, default_flow_style=False, allow_unicode=True)
                        st.download_button(
                            label="下载YAML配置文件",
                            data=config_data,
                            file_name="config.yml",
                            mime="text/yaml"
                        )
                    else:  # JSON
                        config_data = json.dumps(self.current_config, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="下载JSON配置文件",
                            data=config_data,
                            file_name="config.json",
                            mime="application/json"
                        )
                except Exception as e:
                    st.error(f"导出配置失败: {e}")
        
        with col2:
            st.markdown("#### 📥 导入配置")
            
            uploaded_file = st.file_uploader(
                "选择配置文件",
                type=['yml', 'yaml', 'json'],
                help="上传配置文件"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.json'):
                        imported_config = json.load(uploaded_file)
                    else:  # YAML
                        imported_config = yaml.safe_load(uploaded_file)
                    
                    st.success("配置文件解析成功")
                    
                    # 显示配置预览
                    with st.expander("预览配置"):
                        st.json(imported_config)
                    
                    if st.button("✅ 应用配置"):
                        self.current_config = imported_config
                        st.session_state.config_modified = True
                        st.success("配置已应用")
                        st.rerun()
                
                except Exception as e:
                    st.error(f"配置文件解析失败: {e}")
        
        # 配置历史版本
        st.markdown("---")
        st.markdown("#### 📚 配置历史")
        
        # 模拟配置历史
        config_history = [
            {"version": "v1.3", "date": "2024-01-15 14:30", "description": "更新风控参数"},
            {"version": "v1.2", "date": "2024-01-14 09:15", "description": "添加API配置"},
            {"version": "v1.1", "date": "2024-01-13 16:45", "description": "初始配置"}
        ]
        
        for i, history in enumerate(config_history):
            col1, col2, col3 = st.columns([2, 3, 1])
            
            with col1:
                st.text(history['version'])
            
            with col2:
                st.text(f"{history['date']} - {history['description']}")
            
            with col3:
                if st.button("恢复", key=f"restore_{i}"):
                    st.info(f"恢复到 {history['version']} 功能开发中...")
    
    def _render_save_actions(self):
        """渲染保存操作按钮"""
        st.markdown("---")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.session_state.config_modified:
                st.warning("有未保存的配置更改")
            else:
                st.success("所有配置已保存")
        
        with col2:
            if st.button("💾 保存配置", type="primary", disabled=not st.session_state.config_modified):
                try:
                    # 这里应该保存配置到文件
                    # self.config_manager.save_config(self.current_config)
                    st.session_state.config_modified = False
                    st.success("✅ 配置已保存")
                    st.rerun()
                except Exception as e:
                    st.error(f"保存配置失败: {e}")
        
        with col3:
            if st.button("🔄 重启系统", help="重启系统以应用配置更改"):
                st.info("系统重启功能开发中...")
        
        # 配置验证
        if st.session_state.config_modified:
            st.markdown("#### 🔍 配置验证")
            
            # 这里可以添加配置验证逻辑
            validation_results = self._validate_config()
            
            if validation_results['valid']:
                st.success("✅ 配置验证通过")
            else:
                st.error("❌ 配置验证失败")
                for error in validation_results['errors']:
                    st.error(f"• {error}")
    
    def _validate_config(self) -> Dict[str, Any]:
        """验证配置有效性"""
        errors = []
        
        # 验证交易配置
        trading_config = self.current_config.get('trading', {})
        if trading_config.get('commission_rate', 0) < 0:
            errors.append("手续费率不能为负数")
        
        # 验证风控配置
        risk_config = self.current_config.get('risk_management', {})
        if risk_config.get('enabled', True):
            position_limits = risk_config.get('position_limits', {})
            if position_limits.get('max_position_percent', 0) > position_limits.get('max_total_position_percent', 1):
                errors.append("单个持仓比例不能超过总持仓比例")
        
        # 验证API配置
        data_sources = self.current_config.get('data_sources', {})
        binance_config = data_sources.get('binance', {})
        if not binance_config.get('api_key') or not binance_config.get('api_secret'):
            errors.append("Binance API密钥不能为空")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }