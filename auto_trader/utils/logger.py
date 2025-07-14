"""
日志管理工具模块

这个模块提供了统一的日志管理功能，支持：
- 多种日志级别
- 文件和控制台双重输出
- 日志轮转和压缩
- 彩色控制台输出
- 性能日志记录
"""

import logging
import logging.handlers
import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json
import threading
from functools import wraps
import time
import traceback

# 彩色输出支持
try:
    import colorama
    from colorama import Fore, Back, Style
    colorama.init()
    COLOR_SUPPORT = True
except ImportError:
    COLOR_SUPPORT = False
    Fore = Back = Style = None


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 定义颜色映射
        self.colors = {
            'DEBUG': Fore.CYAN if COLOR_SUPPORT else '',
            'INFO': Fore.GREEN if COLOR_SUPPORT else '',
            'WARNING': Fore.YELLOW if COLOR_SUPPORT else '',
            'ERROR': Fore.RED if COLOR_SUPPORT else '',
            'CRITICAL': Fore.RED + Style.BRIGHT if COLOR_SUPPORT else '',
        }
        
        # 重置颜色
        self.reset = Style.RESET_ALL if COLOR_SUPPORT else ''
    
    def format(self, record):
        """格式化日志记录"""
        # 添加颜色
        if record.levelname in self.colors:
            record.levelname = f"{self.colors[record.levelname]}{record.levelname}{self.reset}"
        
        # 格式化消息
        formatted = super().format(record)
        
        return formatted


class JsonFormatter(logging.Formatter):
    """JSON格式化器"""
    
    def format(self, record):
        """格式化为JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class TradingLoggerAdapter(logging.LoggerAdapter):
    """交易日志适配器"""
    
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})
    
    def process(self, msg, kwargs):
        """处理日志消息"""
        # 添加额外上下文信息
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        
        kwargs['extra'].update(self.extra)
        
        # 添加性能信息
        if hasattr(self, '_start_time'):
            kwargs['extra']['elapsed_time'] = time.time() - self._start_time
        
        return msg, kwargs
    
    def start_timer(self):
        """开始计时"""
        self._start_time = time.time()
    
    def log_performance(self, operation: str, duration: float = None):
        """记录性能信息"""
        if duration is None and hasattr(self, '_start_time'):
            duration = time.time() - self._start_time
        
        self.info(f"Performance: {operation} completed in {duration:.4f}s", 
                 extra={'performance_metric': True, 'operation': operation, 'duration': duration})


class LoggerManager:
    """日志管理器"""
    
    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        self.config: Dict[str, Any] = {}
        self.lock = threading.Lock()
    
    def setup_logging(self, config: Dict[str, Any]) -> None:
        """
        设置日志配置
        
        Args:
            config: 日志配置字典
        """
        with self.lock:
            self.config = config
            
            # 设置根日志级别
            root_logger = logging.getLogger()
            root_logger.setLevel(getattr(logging, config.get('level', 'INFO')))
            
            # 清除现有处理器
            root_logger.handlers.clear()
            
            # 创建文件处理器
            if config.get('file_config', {}).get('enabled', True):
                self._setup_file_handler(config['file_config'])
            
            # 创建控制台处理器
            if config.get('console_config', {}).get('enabled', True):
                self._setup_console_handler(config['console_config'])
    
    def _setup_file_handler(self, file_config: Dict[str, Any]) -> None:
        """设置文件处理器"""
        try:
            # 创建日志目录
            log_file = file_config.get('file_path', 'logs/trading.log')
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建轮转文件处理器
            max_size = self._parse_size(file_config.get('max_size', '10MB'))
            backup_count = file_config.get('backup_count', 5)
            
            handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_size,
                backupCount=backup_count,
                encoding=file_config.get('encoding', 'utf-8')
            )
            
            # 设置格式
            format_str = file_config.get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            if file_config.get('json_format', False):
                formatter = JsonFormatter()
            else:
                formatter = logging.Formatter(format_str)
            
            handler.setFormatter(formatter)
            
            # 设置级别
            level = file_config.get('level', self.config.get('level', 'INFO'))
            handler.setLevel(getattr(logging, level))
            
            # 添加到根日志器
            logging.getLogger().addHandler(handler)
            self.handlers['file'] = handler
            
        except Exception as e:
            print(f"设置文件日志处理器失败: {e}")
    
    def _setup_console_handler(self, console_config: Dict[str, Any]) -> None:
        """设置控制台处理器"""
        try:
            # 创建控制台处理器
            handler = logging.StreamHandler(sys.stdout)
            
            # 设置格式
            format_str = console_config.get('format', 
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            if console_config.get('color', True) and COLOR_SUPPORT:
                formatter = ColoredFormatter(format_str)
            else:
                formatter = logging.Formatter(format_str)
            
            handler.setFormatter(formatter)
            
            # 设置级别
            level = console_config.get('level', self.config.get('level', 'INFO'))
            handler.setLevel(getattr(logging, level))
            
            # 添加到根日志器
            logging.getLogger().addHandler(handler)
            self.handlers['console'] = handler
            
        except Exception as e:
            print(f"设置控制台日志处理器失败: {e}")
    
    def _parse_size(self, size_str: str) -> int:
        """解析文件大小字符串"""
        size_str = size_str.upper()
        
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_logger(self, name: str, extra: Optional[Dict[str, Any]] = None) -> TradingLoggerAdapter:
        """
        获取日志器
        
        Args:
            name: 日志器名称
            extra: 额外上下文信息
            
        Returns:
            TradingLoggerAdapter: 日志器适配器
        """
        with self.lock:
            if name not in self.loggers:
                self.loggers[name] = logging.getLogger(name)
            
            logger = self.loggers[name]
            return TradingLoggerAdapter(logger, extra)
    
    def set_level(self, level: str) -> None:
        """
        设置日志级别
        
        Args:
            level: 日志级别
        """
        numeric_level = getattr(logging, level.upper())
        
        # 设置根日志器级别
        logging.getLogger().setLevel(numeric_level)
        
        # 设置所有处理器级别
        for handler in self.handlers.values():
            handler.setLevel(numeric_level)
    
    def add_file_handler(self, name: str, file_path: str, level: str = 'INFO') -> None:
        """
        添加文件处理器
        
        Args:
            name: 处理器名称
            file_path: 文件路径
            level: 日志级别
        """
        try:
            # 创建目录
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 创建处理器
            handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            
            # 设置格式和级别
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            handler.setLevel(getattr(logging, level))
            
            # 添加到根日志器
            logging.getLogger().addHandler(handler)
            self.handlers[name] = handler
            
        except Exception as e:
            print(f"添加文件处理器失败: {e}")
    
    def remove_handler(self, name: str) -> None:
        """
        移除处理器
        
        Args:
            name: 处理器名称
        """
        if name in self.handlers:
            handler = self.handlers[name]
            logging.getLogger().removeHandler(handler)
            handler.close()
            del self.handlers[name]
    
    def flush_all(self) -> None:
        """刷新所有处理器"""
        for handler in self.handlers.values():
            if hasattr(handler, 'flush'):
                handler.flush()
    
    def close_all(self) -> None:
        """关闭所有处理器"""
        for handler in self.handlers.values():
            handler.close()
        self.handlers.clear()
        self.loggers.clear()


# 全局日志管理器
logger_manager = LoggerManager()


def setup_logging(config: Dict[str, Any]) -> None:
    """
    设置日志配置
    
    Args:
        config: 日志配置字典
    """
    logger_manager.setup_logging(config)


def get_logger(name: str, extra: Optional[Dict[str, Any]] = None) -> TradingLoggerAdapter:
    """
    获取日志器
    
    Args:
        name: 日志器名称
        extra: 额外上下文信息
        
    Returns:
        TradingLoggerAdapter: 日志器适配器
    """
    return logger_manager.get_logger(name, extra)


def log_function_call(func):
    """
    装饰器：记录函数调用
    
    Args:
        func: 被装饰的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        # 记录函数调用开始
        logger.start_timer()
        logger.debug(f"Calling function: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            
            # 记录函数调用结束
            logger.log_performance(f"Function {func.__name__}")
            logger.debug(f"Function {func.__name__} completed successfully")
            
            return result
            
        except Exception as e:
            # 记录异常
            logger.error(f"Function {func.__name__} failed: {str(e)}")
            logger.debug(f"Function {func.__name__} traceback: {traceback.format_exc()}")
            raise
    
    return wrapper


def log_execution_time(operation_name: str = None):
    """
    装饰器：记录执行时间
    
    Args:
        operation_name: 操作名称
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            start_time = time.time()
            op_name = operation_name or func.__name__
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.log_performance(op_name, duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Operation {op_name} failed after {duration:.4f}s: {str(e)}")
                raise
        
        return wrapper
    
    return decorator


class LogContext:
    """日志上下文管理器"""
    
    def __init__(self, logger: TradingLoggerAdapter, operation: str, level: str = 'INFO'):
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(getattr(logging, self.level), f"开始 {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(getattr(logging, self.level), 
                          f"完成 {self.operation}，耗时 {duration:.4f}s")
        else:
            self.logger.error(f"操作 {self.operation} 失败，耗时 {duration:.4f}s: {str(exc_val)}")
        
        return False  # 不抑制异常


def with_logging(operation: str, level: str = 'INFO'):
    """
    上下文管理器工厂函数
    
    Args:
        operation: 操作名称
        level: 日志级别
    
    Returns:
        LogContext: 上下文管理器
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            with LogContext(logger, operation, level):
                return func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# 默认日志配置
DEFAULT_LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file_config': {
        'enabled': True,
        'file_path': 'logs/trading.log',
        'max_size': '10MB',
        'backup_count': 5,
        'encoding': 'utf-8'
    },
    'console_config': {
        'enabled': True,
        'color': True
    }
}

# 初始化默认日志配置
setup_logging(DEFAULT_LOG_CONFIG)