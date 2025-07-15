"""
通知系统模块

该模块提供了多种通知方式，包括：
- Telegram机器人通知
- 邮件通知
- Webhook通知
- 本地日志通知
- 推送通知（可扩展）

支持不同类型的通知：
- 交易信号通知
- 风险警告通知
- 系统状态通知
- 错误报告通知
"""

import asyncio
import smtplib
import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
import os
import time
from pathlib import Path


class NotificationType(Enum):
    """通知类型枚举"""
    TRADE_SIGNAL = "trade_signal"          # 交易信号
    TRADE_EXECUTED = "trade_executed"      # 交易执行
    RISK_WARNING = "risk_warning"          # 风险警告
    SYSTEM_STATUS = "system_status"        # 系统状态
    ERROR_REPORT = "error_report"          # 错误报告
    PERFORMANCE_REPORT = "performance_report"  # 性能报告
    MARKET_ALERT = "market_alert"          # 市场提醒
    EMERGENCY = "emergency"                # 紧急情况


class NotificationLevel(Enum):
    """通知级别枚举"""
    LOW = "low"           # 低级别
    MEDIUM = "medium"     # 中级别
    HIGH = "high"         # 高级别
    CRITICAL = "critical" # 关键级别


@dataclass
class NotificationMessage:
    """通知消息"""
    title: str                              # 标题
    content: str                           # 内容
    notification_type: NotificationType    # 通知类型
    level: NotificationLevel = NotificationLevel.MEDIUM  # 级别
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'title': self.title,
            'content': self.content,
            'type': self.notification_type.value,
            'level': self.level.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


class NotificationProvider:
    """通知提供者基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化通知提供者
        
        Args:
            name: 提供者名称
            config: 配置信息
        """
        self.name = name
        self.config = config
        self.enabled = config.get('enabled', True)
        self.logger = logging.getLogger(f"Notification.{name}")
        
        # 通知历史
        self.notification_history: List[Dict] = []
        
        # 限流设置
        self.rate_limit = config.get('rate_limit', 10)  # 每分钟最大通知数
        self.recent_notifications: List[float] = []
        
        self.logger.info(f"通知提供者 {name} 初始化完成")
    
    def send_notification(self, message: NotificationMessage) -> bool:
        """
        发送通知（子类需要实现）
        
        Args:
            message: 通知消息
            
        Returns:
            bool: 是否发送成功
        """
        raise NotImplementedError("子类必须实现 send_notification 方法")
    
    def can_send_notification(self, message: NotificationMessage) -> bool:
        """
        检查是否可以发送通知（限流检查）
        
        Args:
            message: 通知消息
            
        Returns:
            bool: 是否可以发送
        """
        if not self.enabled:
            return False
        
        # 检查限流
        now = time.time()
        # 清理1分钟前的记录
        self.recent_notifications = [t for t in self.recent_notifications if now - t < 60]
        
        if len(self.recent_notifications) >= self.rate_limit:
            self.logger.warning(f"通知发送频率过高，已达到限制: {self.rate_limit}/分钟")
            return False
        
        return True
    
    def record_notification(self, message: NotificationMessage, success: bool):
        """
        记录通知历史
        
        Args:
            message: 通知消息
            success: 是否成功
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'message': message.to_dict(),
            'success': success,
            'provider': self.name
        }
        
        self.notification_history.append(record)
        self.recent_notifications.append(time.time())
        
        # 保留最近1000条记录
        if len(self.notification_history) > 1000:
            self.notification_history = self.notification_history[-1000:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取通知统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_notifications = len(self.notification_history)
        successful_notifications = sum(1 for record in self.notification_history if record['success'])
        
        # 按类型统计
        type_stats = {}
        for record in self.notification_history:
            msg_type = record['message']['type']
            if msg_type not in type_stats:
                type_stats[msg_type] = {'total': 0, 'success': 0}
            type_stats[msg_type]['total'] += 1
            if record['success']:
                type_stats[msg_type]['success'] += 1
        
        return {
            'provider': self.name,
            'enabled': self.enabled,
            'total_notifications': total_notifications,
            'successful_notifications': successful_notifications,
            'success_rate': successful_notifications / total_notifications if total_notifications > 0 else 0,
            'type_statistics': type_stats,
            'recent_notifications_count': len(self.recent_notifications),
            'rate_limit': self.rate_limit
        }


class TelegramNotificationProvider(NotificationProvider):
    """Telegram通知提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Telegram", config)
        
        self.bot_token = config.get('bot_token', '')
        self.chat_id = config.get('chat_id', '')
        self.parse_mode = config.get('parse_mode', 'HTML')
        self.disable_web_page_preview = config.get('disable_web_page_preview', True)
        
        if not self.bot_token or not self.chat_id:
            self.logger.warning("Telegram配置不完整，通知功能将被禁用")
            self.enabled = False
    
    def send_notification(self, message: NotificationMessage) -> bool:
        """
        发送Telegram通知
        
        Args:
            message: 通知消息
            
        Returns:
            bool: 是否发送成功
        """
        if not self.can_send_notification(message):
            return False
        
        try:
            # 格式化消息
            formatted_message = self._format_message(message)
            
            # 发送消息
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': formatted_message,
                'parse_mode': self.parse_mode,
                'disable_web_page_preview': self.disable_web_page_preview
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Telegram通知发送成功: {message.title}")
                self.record_notification(message, True)
                return True
            else:
                self.logger.error(f"Telegram通知发送失败: {response.status_code} - {response.text}")
                self.record_notification(message, False)
                return False
                
        except Exception as e:
            self.logger.error(f"Telegram通知发送异常: {e}")
            self.record_notification(message, False)
            return False
    
    def _format_message(self, message: NotificationMessage) -> str:
        """
        格式化Telegram消息
        
        Args:
            message: 通知消息
            
        Returns:
            str: 格式化后的消息
        """
        # 根据级别选择图标
        level_icons = {
            NotificationLevel.LOW: "🔵",
            NotificationLevel.MEDIUM: "🟡",
            NotificationLevel.HIGH: "🟠",
            NotificationLevel.CRITICAL: "🔴"
        }
        
        # 根据类型选择图标
        type_icons = {
            NotificationType.TRADE_SIGNAL: "📊",
            NotificationType.TRADE_EXECUTED: "✅",
            NotificationType.RISK_WARNING: "⚠️",
            NotificationType.SYSTEM_STATUS: "🔧",
            NotificationType.ERROR_REPORT: "❌",
            NotificationType.PERFORMANCE_REPORT: "📈",
            NotificationType.MARKET_ALERT: "📢",
            NotificationType.EMERGENCY: "🚨"
        }
        
        level_icon = level_icons.get(message.level, "🔵")
        type_icon = type_icons.get(message.notification_type, "📝")
        
        formatted = f"{level_icon} {type_icon} <b>{message.title}</b>\n\n"
        formatted += f"{message.content}\n\n"
        
        # 添加元数据
        if message.metadata:
            formatted += "<b>详细信息:</b>\n"
            for key, value in message.metadata.items():
                formatted += f"• {key}: {value}\n"
            formatted += "\n"
        
        formatted += f"<i>时间: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</i>"
        
        return formatted
    
    def send_chart(self, chart_path: str, caption: str = "") -> bool:
        """
        发送图表到Telegram
        
        Args:
            chart_path: 图表文件路径
            caption: 图表说明
            
        Returns:
            bool: 是否发送成功
        """
        if not self.enabled or not os.path.exists(chart_path):
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
            
            with open(chart_path, 'rb') as photo:
                files = {'photo': photo}
                data = {
                    'chat_id': self.chat_id,
                    'caption': caption
                }
                
                response = requests.post(url, files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    self.logger.info(f"Telegram图表发送成功: {chart_path}")
                    return True
                else:
                    self.logger.error(f"Telegram图表发送失败: {response.status_code} - {response.text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Telegram图表发送异常: {e}")
            return False


class EmailNotificationProvider(NotificationProvider):
    """邮件通知提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Email", config)
        
        self.smtp_server = config.get('smtp_server', '')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_email = config.get('from_email', '')
        self.to_emails = config.get('to_emails', [])
        self.use_tls = config.get('use_tls', True)
        
        if not all([self.smtp_server, self.username, self.password, self.from_email, self.to_emails]):
            self.logger.warning("邮件配置不完整，通知功能将被禁用")
            self.enabled = False
    
    def send_notification(self, message: NotificationMessage) -> bool:
        """
        发送邮件通知
        
        Args:
            message: 通知消息
            
        Returns:
            bool: 是否发送成功
        """
        if not self.can_send_notification(message):
            return False
        
        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = ', '.join(self.to_emails)
            msg['Subject'] = f"[{message.level.value.upper()}] {message.title}"
            
            # 邮件正文
            body = self._format_email_body(message)
            msg.attach(MIMEText(body, 'html'))
            
            # 发送邮件
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            if self.use_tls:
                server.starttls()
            server.login(self.username, self.password)
            
            text = msg.as_string()
            server.sendmail(self.from_email, self.to_emails, text)
            server.quit()
            
            self.logger.info(f"邮件通知发送成功: {message.title}")
            self.record_notification(message, True)
            return True
            
        except Exception as e:
            self.logger.error(f"邮件通知发送异常: {e}")
            self.record_notification(message, False)
            return False
    
    def _format_email_body(self, message: NotificationMessage) -> str:
        """
        格式化邮件正文
        
        Args:
            message: 通知消息
            
        Returns:
            str: 格式化后的邮件正文
        """
        body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .content {{ margin: 20px 0; }}
                .metadata {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                .footer {{ color: #666; font-size: 12px; margin-top: 20px; }}
                .level-high {{ color: #ff9800; }}
                .level-critical {{ color: #f44336; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2 class="level-{message.level.value}">{message.title}</h2>
                <p>通知类型: {message.notification_type.value}</p>
                <p>级别: {message.level.value.upper()}</p>
                <p>时间: {message.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                <p>{message.content}</p>
            </div>
        """
        
        if message.metadata:
            body += """
            <div class="metadata">
                <h3>详细信息:</h3>
                <ul>
            """
            for key, value in message.metadata.items():
                body += f"<li><strong>{key}:</strong> {value}</li>"
            body += """
                </ul>
            </div>
            """
        
        body += """
            <div class="footer">
                <p>此邮件由量化交易系统自动发送</p>
            </div>
        </body>
        </html>
        """
        
        return body


class WebhookNotificationProvider(NotificationProvider):
    """Webhook通知提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Webhook", config)
        
        self.webhook_url = config.get('webhook_url', '')
        self.headers = config.get('headers', {})
        self.method = config.get('method', 'POST')
        self.timeout = config.get('timeout', 10)
        
        if not self.webhook_url:
            self.logger.warning("Webhook URL未配置，通知功能将被禁用")
            self.enabled = False
    
    def send_notification(self, message: NotificationMessage) -> bool:
        """
        发送Webhook通知
        
        Args:
            message: 通知消息
            
        Returns:
            bool: 是否发送成功
        """
        if not self.can_send_notification(message):
            return False
        
        try:
            payload = {
                'title': message.title,
                'content': message.content,
                'type': message.notification_type.value,
                'level': message.level.value,
                'timestamp': message.timestamp.isoformat(),
                'metadata': message.metadata
            }
            
            response = requests.request(
                method=self.method,
                url=self.webhook_url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                self.logger.info(f"Webhook通知发送成功: {message.title}")
                self.record_notification(message, True)
                return True
            else:
                self.logger.error(f"Webhook通知发送失败: {response.status_code} - {response.text}")
                self.record_notification(message, False)
                return False
                
        except Exception as e:
            self.logger.error(f"Webhook通知发送异常: {e}")
            self.record_notification(message, False)
            return False


class LogNotificationProvider(NotificationProvider):
    """日志通知提供者"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Log", config)
        
        self.log_file = config.get('log_file', 'notifications.log')
        self.log_level = config.get('log_level', 'INFO')
        
        # 创建专门的通知日志记录器
        self.notification_logger = logging.getLogger('notifications')
        self.notification_logger.setLevel(getattr(logging, self.log_level))
        
        # 添加文件处理器
        if self.log_file:
            handler = logging.FileHandler(self.log_file, encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.notification_logger.addHandler(handler)
    
    def send_notification(self, message: NotificationMessage) -> bool:
        """
        发送日志通知
        
        Args:
            message: 通知消息
            
        Returns:
            bool: 是否发送成功
        """
        if not self.can_send_notification(message):
            return False
        
        try:
            # 格式化日志消息
            log_message = f"[{message.level.value.upper()}] [{message.notification_type.value}] {message.title}: {message.content}"
            
            if message.metadata:
                log_message += f" | 元数据: {json.dumps(message.metadata, ensure_ascii=False)}"
            
            # 根据级别选择日志级别
            if message.level == NotificationLevel.CRITICAL:
                self.notification_logger.critical(log_message)
            elif message.level == NotificationLevel.HIGH:
                self.notification_logger.warning(log_message)
            elif message.level == NotificationLevel.MEDIUM:
                self.notification_logger.info(log_message)
            else:
                self.notification_logger.debug(log_message)
            
            self.record_notification(message, True)
            return True
            
        except Exception as e:
            self.logger.error(f"日志通知发送异常: {e}")
            self.record_notification(message, False)
            return False


class NotificationManager:
    """通知管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化通知管理器
        
        Args:
            config: 通知配置
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 通知提供者
        self.providers: Dict[str, NotificationProvider] = {}
        
        # 通知过滤器
        self.filters = config.get('filters', {})
        
        # 初始化通知提供者
        self._initialize_providers()
        
        # 通知队列（用于异步发送）
        self.notification_queue: List[NotificationMessage] = []
        
        self.logger.info("通知管理器初始化完成")
    
    def _initialize_providers(self):
        """初始化通知提供者"""
        providers_config = self.config.get('providers', {})
        
        # Telegram
        if 'telegram' in providers_config:
            self.providers['telegram'] = TelegramNotificationProvider(providers_config['telegram'])
        
        # Email
        if 'email' in providers_config:
            self.providers['email'] = EmailNotificationProvider(providers_config['email'])
        
        # Webhook
        if 'webhook' in providers_config:
            self.providers['webhook'] = WebhookNotificationProvider(providers_config['webhook'])
        
        # Log
        if 'log' in providers_config:
            self.providers['log'] = LogNotificationProvider(providers_config['log'])
        
        # 如果没有配置任何提供者，使用默认日志提供者
        if not self.providers:
            self.providers['log'] = LogNotificationProvider({'enabled': True})
        
        self.logger.info(f"初始化了 {len(self.providers)} 个通知提供者")
    
    def send_notification(self, 
                         title: str,
                         content: str,
                         notification_type: NotificationType,
                         level: NotificationLevel = NotificationLevel.MEDIUM,
                         metadata: Optional[Dict[str, Any]] = None,
                         providers: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        发送通知
        
        Args:
            title: 通知标题
            content: 通知内容
            notification_type: 通知类型
            level: 通知级别
            metadata: 元数据
            providers: 指定的提供者列表，如果为None则使用所有启用的提供者
            
        Returns:
            Dict[str, bool]: 各提供者的发送结果
        """
        message = NotificationMessage(
            title=title,
            content=content,
            notification_type=notification_type,
            level=level,
            metadata=metadata or {}
        )
        
        return self.send_notification_message(message, providers)
    
    def send_notification_message(self, 
                                 message: NotificationMessage,
                                 providers: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        发送通知消息
        
        Args:
            message: 通知消息
            providers: 指定的提供者列表
            
        Returns:
            Dict[str, bool]: 各提供者的发送结果
        """
        # 应用过滤器
        if not self._should_send_notification(message):
            self.logger.debug(f"通知被过滤器拒绝: {message.title}")
            return {}
        
        # 确定要使用的提供者
        if providers is None:
            providers = [name for name, provider in self.providers.items() if provider.enabled]
        
        results = {}
        
        for provider_name in providers:
            if provider_name in self.providers:
                provider = self.providers[provider_name]
                try:
                    result = provider.send_notification(message)
                    results[provider_name] = result
                except Exception as e:
                    self.logger.error(f"提供者 {provider_name} 发送通知失败: {e}")
                    results[provider_name] = False
            else:
                self.logger.warning(f"未找到通知提供者: {provider_name}")
                results[provider_name] = False
        
        return results
    
    def _should_send_notification(self, message: NotificationMessage) -> bool:
        """
        检查是否应该发送通知（过滤器）
        
        Args:
            message: 通知消息
            
        Returns:
            bool: 是否应该发送
        """
        # 检查级别过滤器
        min_level = self.filters.get('min_level', NotificationLevel.LOW)
        if isinstance(min_level, str):
            min_level = NotificationLevel(min_level)
        
        level_order = [NotificationLevel.LOW, NotificationLevel.MEDIUM, 
                      NotificationLevel.HIGH, NotificationLevel.CRITICAL]
        
        if level_order.index(message.level) < level_order.index(min_level):
            return False
        
        # 检查类型过滤器
        allowed_types = self.filters.get('allowed_types', [])
        if allowed_types and message.notification_type not in allowed_types:
            return False
        
        # 检查禁用类型过滤器
        disabled_types = self.filters.get('disabled_types', [])
        if message.notification_type in disabled_types:
            return False
        
        return True
    
    def send_trade_signal(self, 
                         symbol: str,
                         signal_type: str,
                         price: float,
                         confidence: float,
                         strategy: str,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        发送交易信号通知
        
        Args:
            symbol: 交易对
            signal_type: 信号类型
            price: 价格
            confidence: 信心水平
            strategy: 策略名称
            metadata: 额外元数据
        """
        title = f"交易信号 - {symbol}"
        content = f"策略 {strategy} 产生了 {signal_type} 信号\n价格: {price:.2f}\n信心水平: {confidence:.2%}"
        
        notification_metadata = {
            'symbol': symbol,
            'signal_type': signal_type,
            'price': price,
            'confidence': confidence,
            'strategy': strategy
        }
        
        if metadata:
            notification_metadata.update(metadata)
        
        self.send_notification(
            title=title,
            content=content,
            notification_type=NotificationType.TRADE_SIGNAL,
            level=NotificationLevel.MEDIUM,
            metadata=notification_metadata
        )
    
    def send_trade_executed(self, 
                           symbol: str,
                           side: str,
                           quantity: float,
                           price: float,
                           pnl: Optional[float] = None,
                           strategy: str = ""):
        """
        发送交易执行通知
        
        Args:
            symbol: 交易对
            side: 交易方向
            quantity: 数量
            price: 价格
            pnl: 盈亏
            strategy: 策略名称
        """
        title = f"交易执行 - {symbol}"
        content = f"执行了 {side} 交易\n数量: {quantity:.6f}\n价格: {price:.2f}"
        
        if pnl is not None:
            content += f"\n盈亏: {pnl:.2f} USDT"
        
        if strategy:
            content += f"\n策略: {strategy}"
        
        self.send_notification(
            title=title,
            content=content,
            notification_type=NotificationType.TRADE_EXECUTED,
            level=NotificationLevel.MEDIUM,
            metadata={
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'pnl': pnl,
                'strategy': strategy
            }
        )
    
    def send_risk_warning(self, 
                         warning_type: str,
                         description: str,
                         severity: str = "medium",
                         metadata: Optional[Dict[str, Any]] = None):
        """
        发送风险警告通知
        
        Args:
            warning_type: 警告类型
            description: 描述
            severity: 严重程度
            metadata: 额外元数据
        """
        severity_levels = {
            'low': NotificationLevel.LOW,
            'medium': NotificationLevel.MEDIUM,
            'high': NotificationLevel.HIGH,
            'critical': NotificationLevel.CRITICAL
        }
        
        title = f"风险警告 - {warning_type}"
        content = description
        
        self.send_notification(
            title=title,
            content=content,
            notification_type=NotificationType.RISK_WARNING,
            level=severity_levels.get(severity, NotificationLevel.MEDIUM),
            metadata=metadata or {}
        )
    
    def send_system_status(self, 
                          status: str,
                          description: str,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        发送系统状态通知
        
        Args:
            status: 状态
            description: 描述
            metadata: 额外元数据
        """
        title = f"系统状态 - {status}"
        content = description
        
        self.send_notification(
            title=title,
            content=content,
            notification_type=NotificationType.SYSTEM_STATUS,
            level=NotificationLevel.MEDIUM,
            metadata=metadata or {}
        )
    
    def send_error_report(self, 
                         error_type: str,
                         error_message: str,
                         traceback: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        发送错误报告通知
        
        Args:
            error_type: 错误类型
            error_message: 错误消息
            traceback: 错误堆栈
            metadata: 额外元数据
        """
        title = f"错误报告 - {error_type}"
        content = f"错误消息: {error_message}"
        
        if traceback:
            content += f"\n\n堆栈信息:\n{traceback}"
        
        self.send_notification(
            title=title,
            content=content,
            notification_type=NotificationType.ERROR_REPORT,
            level=NotificationLevel.HIGH,
            metadata=metadata or {}
        )
    
    def send_performance_report(self, 
                              report_data: Dict[str, Any],
                              chart_path: Optional[str] = None):
        """
        发送性能报告通知
        
        Args:
            report_data: 报告数据
            chart_path: 图表路径
        """
        title = "性能报告"
        content = f"总收益率: {report_data.get('total_return', 0):.2%}\n"
        content += f"胜率: {report_data.get('win_rate', 0):.1%}\n"
        content += f"最大回撤: {report_data.get('max_drawdown', 0):.2%}\n"
        content += f"夏普比率: {report_data.get('sharpe_ratio', 0):.3f}"
        
        self.send_notification(
            title=title,
            content=content,
            notification_type=NotificationType.PERFORMANCE_REPORT,
            level=NotificationLevel.MEDIUM,
            metadata=report_data
        )
        
        # 如果有图表且有Telegram提供者，发送图表
        if chart_path and 'telegram' in self.providers:
            telegram_provider = self.providers['telegram']
            telegram_provider.send_chart(chart_path, "性能报告图表")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取通知统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        stats = {
            'total_providers': len(self.providers),
            'enabled_providers': sum(1 for p in self.providers.values() if p.enabled),
            'providers_statistics': {}
        }
        
        for name, provider in self.providers.items():
            stats['providers_statistics'][name] = provider.get_statistics()
        
        return stats
    
    def test_notifications(self) -> Dict[str, bool]:
        """
        测试所有通知提供者
        
        Returns:
            Dict[str, bool]: 各提供者的测试结果
        """
        test_message = NotificationMessage(
            title="通知系统测试",
            content="这是一条测试消息，用于验证通知系统是否正常工作。",
            notification_type=NotificationType.SYSTEM_STATUS,
            level=NotificationLevel.LOW,
            metadata={'test': True}
        )
        
        return self.send_notification_message(test_message)