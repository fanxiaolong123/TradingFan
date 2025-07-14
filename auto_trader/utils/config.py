"""
配置管理工具模块

这个模块负责加载和管理系统配置，支持：
- YAML配置文件加载
- 环境变量覆盖
- 敏感信息安全管理
- 配置验证和默认值处理
"""

import os
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import json

# 延迟导入logger以避免循环导入
logger = None

def _get_logger():
    global logger
    if logger is None:
        from .logger import get_logger
        logger = get_logger(__name__)
    return logger


@dataclass
class ConfigManager:
    """配置管理器"""
    
    # 配置文件路径
    config_file: str = "config.yml"
    secrets_file: str = "secrets.yml"
    
    # 配置数据
    config: Dict[str, Any] = field(default_factory=dict)
    secrets: Dict[str, Any] = field(default_factory=dict)
    
    # 环境变量前缀
    env_prefix: str = "AUTOTRADER_"
    
    def __post_init__(self):
        """初始化后加载配置"""
        self.load_config()
        self.load_secrets()
        self.apply_env_overrides()
        self.validate_config()
    
    def load_config(self) -> None:
        """加载主配置文件"""
        try:
            config_path = Path(self.config_file)
            
            if not config_path.exists():
                _get_logger().warning(f"配置文件不存在: {self.config_file}，使用默认配置")
                self.config = self._get_default_config()
                return
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            
            _get_logger().info(f"配置文件加载成功: {self.config_file}")
            
        except Exception as e:
            _get_logger().error(f"加载配置文件失败: {e}")
            self.config = self._get_default_config()
    
    def load_secrets(self) -> None:
        """加载敏感信息配置文件"""
        try:
            secrets_path = Path(self.secrets_file)
            
            if not secrets_path.exists():
                _get_logger().warning(f"敏感信息文件不存在: {self.secrets_file}")
                self.secrets = {}
                return
            
            with open(secrets_path, 'r', encoding='utf-8') as f:
                self.secrets = yaml.safe_load(f) or {}
            
            _get_logger().info(f"敏感信息文件加载成功: {self.secrets_file}")
            
        except Exception as e:
            _get_logger().error(f"加载敏感信息文件失败: {e}")
            self.secrets = {}
    
    def apply_env_overrides(self) -> None:
        """应用环境变量覆盖"""
        env_count = 0
        
        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                # 移除前缀并转换为小写
                config_key = key[len(self.env_prefix):].lower()
                
                # 将环境变量键转换为配置路径
                config_path = config_key.replace('_', '.')
                
                # 设置配置值
                self._set_nested_value(self.config, config_path, value)
                env_count += 1
        
        if env_count > 0:
            _get_logger().info(f"应用了 {env_count} 个环境变量覆盖")
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: str) -> None:
        """设置嵌套配置值"""
        keys = path.split('.')
        current = config
        
        # 导航到最后一级
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # 设置最终值（尝试自动类型转换）
        final_key = keys[-1]
        current[final_key] = self._convert_value(value)
    
    def _convert_value(self, value: str) -> Union[str, int, float, bool]:
        """自动转换值类型"""
        # 布尔值转换
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # 数字转换
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # 保持字符串
        return value
    
    def validate_config(self) -> None:
        """验证配置有效性"""
        errors = []
        
        # 验证必需的配置项
        required_keys = [
            'system.name',
            'system.version',
            'data_sources.default',
            'trading.default_commission_rate',
            'account.initial_balance'
        ]
        
        for key in required_keys:
            if not self.get(key):
                errors.append(f"缺少必需的配置项: {key}")
        
        # 验证数值范围
        numeric_validations = [
            ('trading.default_commission_rate', 0, 1, "手续费率必须在0-1之间"),
            ('trading.default_slippage', 0, 1, "滑点必须在0-1之间"),
            ('risk_management.loss_limits.max_daily_loss_percent', 0, 1, "每日最大损失比例必须在0-1之间"),
            ('risk_management.position_limits.max_position_percent', 0, 1, "最大仓位比例必须在0-1之间"),
        ]
        
        for key, min_val, max_val, message in numeric_validations:
            value = self.get(key)
            if value is not None and not (min_val <= value <= max_val):
                errors.append(f"{message}: {key} = {value}")
        
        # 验证支持的交易对
        supported_symbols = self.get('trading.supported_symbols', [])
        if not supported_symbols:
            errors.append("未配置支持的交易对")
        
        # 验证支持的时间周期
        supported_timeframes = self.get('trading.supported_timeframes', [])
        if not supported_timeframes:
            errors.append("未配置支持的时间周期")
        
        # 如果有错误，记录并抛出异常
        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(f"- {error}" for error in errors)
            _get_logger().error(error_msg)
            raise ValueError(error_msg)
        
        _get_logger().info("配置验证通过")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        # 首先尝试从敏感信息中获取
        value = self._get_nested_value(self.secrets, key)
        if value is not None:
            return value
        
        # 然后从主配置中获取
        value = self._get_nested_value(self.config, key)
        if value is not None:
            return value
        
        return default
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """获取嵌套配置值"""
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        self._set_nested_value(self.config, key, value)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置段
        
        Args:
            section: 配置段名称
            
        Returns:
            配置段字典
        """
        return self.get(section, {})
    
    def get_data_source_config(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取数据源配置
        
        Args:
            name: 数据源名称，如果为None则返回默认数据源配置
            
        Returns:
            数据源配置字典
        """
        if name is None:
            name = self.get('data_sources.default', 'binance')
        
        return self.get(f'data_sources.{name}', {})
    
    def get_strategy_config(self, name: str) -> Dict[str, Any]:
        """
        获取策略配置
        
        Args:
            name: 策略名称
            
        Returns:
            策略配置字典
        """
        return self.get(f'strategies.{name}', {})
    
    def get_risk_management_config(self) -> Dict[str, Any]:
        """
        获取风险管理配置
        
        Returns:
            风险管理配置字典
        """
        return self.get('risk_management', {})
    
    def get_account_config(self) -> Dict[str, Any]:
        """
        获取账户配置
        
        Returns:
            账户配置字典
        """
        return self.get('account', {})
    
    def get_backtest_config(self) -> Dict[str, Any]:
        """
        获取回测配置
        
        Returns:
            回测配置字典
        """
        return self.get('backtest', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        获取日志配置
        
        Returns:
            日志配置字典
        """
        return self.get('logging', {})
    
    def get_web_ui_config(self) -> Dict[str, Any]:
        """
        获取Web UI配置
        
        Returns:
            Web UI配置字典
        """
        return self.get('web_ui', {})
    
    def is_debug_mode(self) -> bool:
        """
        检查是否为调试模式
        
        Returns:
            是否为调试模式
        """
        return self.get('development.debug', False)
    
    def is_production_mode(self) -> bool:
        """
        检查是否为生产模式
        
        Returns:
            是否为生产模式
        """
        return self.get('system.environment', 'development') == 'production'
    
    def save_config(self, file_path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            file_path: 文件路径，如果为None则保存到原文件
        """
        try:
            save_path = file_path or self.config_file
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            _get_logger().info(f"配置已保存到: {save_path}")
            
        except Exception as e:
            _get_logger().error(f"保存配置失败: {e}")
            raise
    
    def create_secrets_template(self) -> None:
        """创建敏感信息模板文件"""
        secrets_template = {
            'data_sources': {
                'binance': {
                    'api_key': 'your_binance_api_key_here',
                    'api_secret': 'your_binance_api_secret_here'
                }
            },
            'database': {
                'postgresql': {
                    'username': 'your_db_username',
                    'password': 'your_db_password'
                }
            },
            'notifications': {
                'email': {
                    'username': 'your_email_username',
                    'password': 'your_email_password',
                    'from_email': 'your_email@example.com',
                    'to_emails': ['recipient@example.com']
                },
                'telegram': {
                    'bot_token': 'your_telegram_bot_token',
                    'chat_id': 'your_telegram_chat_id'
                }
            },
            'web_ui': {
                'authentication': {
                    'password': 'your_web_ui_password'
                }
            }
        }
        
        try:
            with open(self.secrets_file, 'w', encoding='utf-8') as f:
                yaml.dump(secrets_template, f, default_flow_style=False, allow_unicode=True)
            
            _get_logger().info(f"敏感信息模板已创建: {self.secrets_file}")
            _get_logger().warning("请编辑secrets.yml文件并填入真实的敏感信息")
            
        except Exception as e:
            _get_logger().error(f"创建敏感信息模板失败: {e}")
            raise
    
    def export_config(self, file_path: str, format: str = 'yaml') -> None:
        """
        导出配置到指定格式的文件
        
        Args:
            file_path: 导出文件路径
            format: 导出格式 ('yaml', 'json')
        """
        try:
            export_data = {
                'config': self.config,
                'export_time': datetime.now().isoformat(),
                'version': self.get('system.version', '1.0.0')
            }
            
            if format.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
            
            _get_logger().info(f"配置已导出到: {file_path}")
            
        except Exception as e:
            _get_logger().error(f"导出配置失败: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'system': {
                'name': 'AutoTrader',
                'version': '1.0.0',
                'environment': 'development',
                'timezone': 'Asia/Shanghai',
                'log_level': 'INFO'
            },
            'data_sources': {
                'default': 'binance',
                'binance': {
                    'enabled': True,
                    'testnet': False,
                    'timeout': 30,
                    'retry_times': 3
                }
            },
            'trading': {
                'default_commission_rate': 0.001,
                'default_slippage': 0.0001,
                'min_order_value': 10.0,
                'max_order_value': 50000.0,
                'supported_symbols': ['BTCUSDT', 'ETHUSDT'],
                'supported_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            },
            'account': {
                'initial_balance': {'USDT': 10000.0},
                'account_type': 'SPOT'
            },
            'risk_management': {
                'enabled': True,
                'position_limits': {
                    'max_position_percent': 0.1,
                    'max_total_position_percent': 0.8
                },
                'loss_limits': {
                    'max_daily_loss_percent': 0.05,
                    'max_total_loss_percent': 0.20
                }
            },
            'logging': {
                'level': 'INFO',
                'file_config': {
                    'enabled': True,
                    'file_path': 'logs/trading.log'
                },
                'console_config': {
                    'enabled': True
                }
            }
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"ConfigManager(config_file={self.config_file}, keys={len(self.config)})"
    
    def __repr__(self) -> str:
        """对象表示"""
        return self.__str__()


# 全局配置实例
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """
    获取全局配置管理器实例
    
    Returns:
        ConfigManager: 配置管理器实例
    """
    return config_manager


def reload_config() -> None:
    """重新加载配置"""
    global config_manager
    config_manager = ConfigManager()
    _get_logger().info("配置已重新加载")