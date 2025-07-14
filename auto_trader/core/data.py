"""
数据模块 - 负责行情数据的获取和管理

这个模块提供了统一的数据接口，支持：
- 历史K线数据获取
- 实时行情数据订阅
- 多种数据源支持（Binance、CCXT等）
- 数据缓存和本地存储
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import asyncio
import websocket
import json
import requests

# 可选导入，避免缺少依赖时导入失败
try:
    from binance.client import Client
    from binance import ThreadedWebsocketManager
    BINANCE_AVAILABLE = True
except ImportError:
    Client = None
    ThreadedWebsocketManager = None
    BINANCE_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    ccxt = None
    CCXT_AVAILABLE = False


class DataSourceType(Enum):
    """数据源类型枚举"""
    BINANCE = "binance"          # 币安交易所
    CCXT = "ccxt"               # CCXT统一接口
    LOCAL = "local"             # 本地数据文件
    MOCK = "mock"               # 模拟数据（用于测试）


@dataclass
class KlineData:
    """K线数据结构"""
    symbol: str                  # 交易对符号
    timestamp: datetime          # 时间戳
    open: float                 # 开盘价
    high: float                 # 最高价
    low: float                  # 最低价
    close: float                # 收盘价
    volume: float               # 成交量
    quote_volume: float = 0.0   # 计价货币成交量
    trades_count: int = 0       # 成交笔数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'quote_volume': self.quote_volume,
            'trades_count': self.trades_count
        }


@dataclass
class MarketTicker:
    """市场行情数据结构"""
    symbol: str                  # 交易对符号
    timestamp: datetime          # 时间戳
    price: float                # 当前价格
    bid: float                  # 买一价
    ask: float                  # 卖一价
    high_24h: float             # 24小时最高价
    low_24h: float              # 24小时最低价
    volume_24h: float           # 24小时成交量
    change_24h: float           # 24小时涨跌幅
    

class DataProvider(ABC):
    """数据提供者抽象基类"""
    
    @abstractmethod
    def get_historical_klines(self, 
                            symbol: str, 
                            interval: str, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 1000) -> pd.DataFrame:
        """
        获取历史K线数据
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔（1m, 5m, 1h, 1d等）
            start_time: 开始时间
            end_time: 结束时间
            limit: 数据条数限制
            
        Returns:
            pd.DataFrame: 包含OHLCV数据的DataFrame
        """
        pass
    
    @abstractmethod
    def get_latest_price(self, symbol: str) -> float:
        """获取最新价格"""
        pass
    
    @abstractmethod
    def get_market_ticker(self, symbol: str) -> MarketTicker:
        """获取市场行情"""
        pass
    
    @abstractmethod
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable) -> None:
        """订阅K线数据"""
        pass
    
    @abstractmethod
    def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """订阅行情数据"""
        pass


class BinanceDataProvider(DataProvider):
    """币安数据提供者"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 base_url: str = "https://api.binance.com", testnet: bool = False):
        """
        初始化币安数据提供者
        
        Args:
            api_key: API密钥（可选，用于获取更高频率限制）
            api_secret: API密钥（可选）
            base_url: API基础URL（默认为主网）
            testnet: 是否使用测试网络（默认为False）
        """
        if not BINANCE_AVAILABLE:
            print("警告: python-binance 库未安装，将使用REST API方式")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url  # 存储API基础URL
        self.testnet = testnet    # 存储测试网络标志
        
        # 不立即初始化Client，避免启动时的API调用
        # 延迟到实际需要时再初始化
        self.client = None
        self._client_initialized = False
        
        # WebSocket管理器
        self.socket_manager = None
        self.active_streams: Dict[str, Any] = {}  # 活跃的数据流
        
        # 时间间隔映射
        if BINANCE_AVAILABLE and Client:
            self.interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '3m': Client.KLINE_INTERVAL_3MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '30m': Client.KLINE_INTERVAL_30MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '2h': Client.KLINE_INTERVAL_2HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '6h': Client.KLINE_INTERVAL_6HOUR,
                '8h': Client.KLINE_INTERVAL_8HOUR,
                '12h': Client.KLINE_INTERVAL_12HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY,
                '3d': Client.KLINE_INTERVAL_3DAY,
                '1w': Client.KLINE_INTERVAL_1WEEK,
                '1M': Client.KLINE_INTERVAL_1MONTH
            }
        else:
            # 备用映射（直接使用字符串）
            self.interval_map = {
                '1m': '1m',
                '3m': '3m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '2h': '2h',
                '4h': '4h',
                '6h': '6h',
                '8h': '8h',
                '12h': '12h',
                '1d': '1d',
                '3d': '3d',
                '1w': '1w',
                '1M': '1M'
            }
    
    def _initialize_client(self) -> None:
        """延迟初始化币安客户端"""
        if self._client_initialized:
            return
            
        if BINANCE_AVAILABLE and self.api_key:
            try:
                # 根据testnet参数选择不同的客户端配置
                if self.testnet:
                    self.client = Client(api_key=self.api_key, api_secret=self.api_secret, testnet=True)
                    print("初始化测试网Client成功")
                else:
                    self.client = Client(api_key=self.api_key, api_secret=self.api_secret)
                    print("初始化主网Client成功")
            except Exception as e:
                print(f"Client初始化失败，将使用REST API: {e}")
                self.client = None
        
        self._client_initialized = True
        self.socket_manager = None
        self.active_streams: Dict[str, Any] = {}  # 活跃的数据流
        
        # 时间间隔映射
        if BINANCE_AVAILABLE and Client:
            self.interval_map = {
                '1m': Client.KLINE_INTERVAL_1MINUTE,
                '3m': Client.KLINE_INTERVAL_3MINUTE,
                '5m': Client.KLINE_INTERVAL_5MINUTE,
                '15m': Client.KLINE_INTERVAL_15MINUTE,
                '30m': Client.KLINE_INTERVAL_30MINUTE,
                '1h': Client.KLINE_INTERVAL_1HOUR,
                '2h': Client.KLINE_INTERVAL_2HOUR,
                '4h': Client.KLINE_INTERVAL_4HOUR,
                '6h': Client.KLINE_INTERVAL_6HOUR,
                '8h': Client.KLINE_INTERVAL_8HOUR,
                '12h': Client.KLINE_INTERVAL_12HOUR,
                '1d': Client.KLINE_INTERVAL_1DAY,
                '3d': Client.KLINE_INTERVAL_3DAY,
                '1w': Client.KLINE_INTERVAL_1WEEK,
                '1M': Client.KLINE_INTERVAL_1MONTH
            }
        else:
            # 备用映射（直接使用字符串）
            self.interval_map = {
                '1m': '1m',
                '3m': '3m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '2h': '2h',
                '4h': '4h',
                '6h': '6h',
                '8h': '8h',
                '12h': '12h',
                '1d': '1d',
                '3d': '3d',
                '1w': '1w',
                '1M': '1M'
            }
    
    def get_historical_klines(self, 
                            symbol: str, 
                            interval: str, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 1000) -> pd.DataFrame:
        """获取历史K线数据"""
        try:
            # 转换时间间隔格式
            binance_interval = self.interval_map.get(interval)
            if not binance_interval:
                raise ValueError(f"不支持的时间间隔: {interval}")
            
            # 延迟初始化Client（如果需要）
            self._initialize_client()
            
            # 使用公共接口获取数据（无需API密钥）
            if self.client:
                # 有API密钥，使用官方客户端
                klines = self.client.get_historical_klines(
                    symbol=symbol,
                    interval=binance_interval,
                    start_str=start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else None,
                    end_str=end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else None,
                    limit=limit
                )
            else:
                # 无API密钥，使用REST API
                klines = self._get_klines_rest(symbol, binance_interval, start_time, end_time, limit)
            
            if not klines:
                print(f"警告: 未获取到 {symbol} 的历史数据")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 'taker_buy_base_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # 数据类型转换
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['quote_volume'] = df['quote_volume'].astype(float)
            df['trades_count'] = df['trades_count'].astype(int)
            
            # 只保留需要的列
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades_count']]
            
            # 添加symbol列
            df['symbol'] = symbol
            
            return df
            
        except Exception as e:
            print(f"获取历史K线数据失败: {e}")
            return pd.DataFrame()
    
    def _get_klines_rest(self, symbol: str, interval: str, start_time: Optional[datetime], 
                        end_time: Optional[datetime], limit: int) -> List[List]:
        """使用REST API获取K线数据"""
        import time
        
        # 使用配置的base_url构建完整的API端点URL
        url = f"{self.base_url}/api/v3/klines"
        
        # 限制每次请求的数量，避免触发限制
        actual_limit = min(limit, 500)
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': actual_limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        # 添加请求头以避免418错误
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive'
        }
        
        # 如果有API密钥，添加到请求头中（用于提高请求限制）
        if self.api_key:
            headers['X-MBX-APIKEY'] = self.api_key
        
        # 添加延时避免频率限制
        time.sleep(0.1)
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API请求失败: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"响应状态码: {e.response.status_code}")
                print(f"响应内容: {e.response.text}")
            raise
    
    def get_latest_price(self, symbol: str) -> float:
        """获取最新价格"""
        try:
            # 尝试延迟初始化Client
            self._initialize_client()
            
            if self.client:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            else:
                # 使用REST API，使用配置的base_url
                url = f"{self.base_url}/api/v3/ticker/price"
                params = {'symbol': symbol}
                
                # 添加请求头
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                    'Accept': 'application/json'
                }
                
                # 如果有API密钥，添加到请求头中
                if self.api_key:
                    headers['X-MBX-APIKEY'] = self.api_key
                
                response = requests.get(url, params=params, headers=headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                return float(data['price'])
        except Exception as e:
            print(f"获取最新价格失败: {e}")
            return 0.0
    
    def get_market_ticker(self, symbol: str) -> MarketTicker:
        """获取市场行情"""
        try:
            if self.client:
                ticker = self.client.get_ticker(symbol=symbol)
            else:
                # 使用REST API
                url = f"https://api.binance.com/api/v3/ticker/24hr"
                params = {'symbol': symbol}
                response = requests.get(url, params=params)
                response.raise_for_status()
                ticker = response.json()
            
            return MarketTicker(
                symbol=symbol,
                timestamp=datetime.now(),
                price=float(ticker['lastPrice']),
                bid=float(ticker['bidPrice']),
                ask=float(ticker['askPrice']),
                high_24h=float(ticker['highPrice']),
                low_24h=float(ticker['lowPrice']),
                volume_24h=float(ticker['volume']),
                change_24h=float(ticker['priceChangePercent']) / 100
            )
        except Exception as e:
            print(f"获取市场行情失败: {e}")
            return MarketTicker(symbol, datetime.now(), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable) -> None:
        """订阅K线数据"""
        if not BINANCE_AVAILABLE:
            print("WebSocket订阅需要python-binance库")
            return
            
        if not self.client:
            print("WebSocket订阅需要API密钥")
            return
        
        try:
            if not self.socket_manager:
                self.socket_manager = ThreadedWebsocketManager(api_key=self.api_key, api_secret=self.api_secret)
                self.socket_manager.start()
            
            # 创建K线数据流
            stream_name = f"{symbol.lower()}@kline_{interval}"
            
            def handle_socket_message(msg):
                """处理WebSocket消息"""
                try:
                    kline_data = msg['k']
                    
                    # 创建KlineData对象
                    kline = KlineData(
                        symbol=kline_data['s'],
                        timestamp=pd.to_datetime(kline_data['t'], unit='ms'),
                        open=float(kline_data['o']),
                        high=float(kline_data['h']),
                        low=float(kline_data['l']),
                        close=float(kline_data['c']),
                        volume=float(kline_data['v']),
                        quote_volume=float(kline_data['q']),
                        trades_count=int(kline_data['n'])
                    )
                    
                    # 调用回调函数
                    callback(kline)
                    
                except Exception as e:
                    print(f"处理K线数据失败: {e}")
            
            # 启动数据流
            conn_key = self.socket_manager.start_kline_socket(
                callback=handle_socket_message,
                symbol=symbol,
                interval=interval
            )
            
            # 记录活跃流
            self.active_streams[stream_name] = conn_key
            
            print(f"开始订阅K线数据: {symbol} - {interval}")
            
        except Exception as e:
            print(f"订阅K线数据失败: {e}")
    
    def subscribe_ticker(self, symbol: str, callback: Callable) -> None:
        """订阅行情数据"""
        if not BINANCE_AVAILABLE:
            print("WebSocket订阅需要python-binance库")
            return
            
        if not self.client:
            print("WebSocket订阅需要API密钥")
            return
        
        try:
            if not self.socket_manager:
                self.socket_manager = ThreadedWebsocketManager(api_key=self.api_key, api_secret=self.api_secret)
                self.socket_manager.start()
            
            def handle_socket_message(msg):
                """处理WebSocket消息"""
                try:
                    ticker = MarketTicker(
                        symbol=msg['s'],
                        timestamp=datetime.now(),
                        price=float(msg['c']),
                        bid=float(msg['b']),
                        ask=float(msg['a']),
                        high_24h=float(msg['h']),
                        low_24h=float(msg['l']),
                        volume_24h=float(msg['v']),
                        change_24h=float(msg['P']) / 100
                    )
                    
                    callback(ticker)
                    
                except Exception as e:
                    print(f"处理行情数据失败: {e}")
            
            # 启动数据流
            conn_key = self.socket_manager.start_symbol_ticker_socket(
                callback=handle_socket_message,
                symbol=symbol
            )
            
            # 记录活跃流
            stream_name = f"{symbol.lower()}@ticker"
            self.active_streams[stream_name] = conn_key
            
            print(f"开始订阅行情数据: {symbol}")
            
        except Exception as e:
            print(f"订阅行情数据失败: {e}")
    
    def close_all_streams(self) -> None:
        """关闭所有数据流"""
        if self.socket_manager:
            for stream_name, conn_key in self.active_streams.items():
                try:
                    self.socket_manager.stop_socket(conn_key)
                    print(f"关闭数据流: {stream_name}")
                except Exception as e:
                    print(f"关闭数据流失败: {e}")
            
            self.active_streams.clear()
            self.socket_manager.stop()
            self.socket_manager = None


class DataManager:
    """数据管理器 - 统一管理多个数据源"""
    
    def __init__(self):
        """初始化数据管理器"""
        self.providers: Dict[str, DataProvider] = {}    # 数据提供者
        self.default_provider: Optional[str] = None      # 默认数据提供者
        self.cache: Dict[str, pd.DataFrame] = {}        # 数据缓存
        self.cache_expire_time = 60  # 缓存过期时间（秒）
        self.last_cache_time: Dict[str, datetime] = {}  # 最后缓存时间
    
    def add_provider(self, name: str, provider: DataProvider, is_default: bool = False) -> None:
        """
        添加数据提供者
        
        Args:
            name: 提供者名称
            provider: 数据提供者实例
            is_default: 是否设为默认提供者
        """
        self.providers[name] = provider
        if is_default or not self.default_provider:
            self.default_provider = name
        print(f"添加数据提供者: {name}")
    
    def get_provider(self, name: Optional[str] = None) -> Optional[DataProvider]:
        """
        获取数据提供者
        
        Args:
            name: 提供者名称，如果为None则返回默认提供者
            
        Returns:
            DataProvider: 数据提供者实例
        """
        provider_name = name or self.default_provider
        return self.providers.get(provider_name)
    
    def get_historical_klines(self, 
                            symbol: str, 
                            interval: str, 
                            start_time: Optional[datetime] = None,
                            end_time: Optional[datetime] = None,
                            limit: int = 1000,
                            provider: Optional[str] = None,
                            use_cache: bool = True) -> pd.DataFrame:
        """
        获取历史K线数据（支持缓存）
        
        Args:
            symbol: 交易对符号
            interval: 时间间隔
            start_time: 开始时间
            end_time: 结束时间
            limit: 数据条数限制
            provider: 数据提供者名称
            use_cache: 是否使用缓存
            
        Returns:
            pd.DataFrame: K线数据
        """
        # 生成缓存键
        cache_key = f"{symbol}_{interval}_{start_time}_{end_time}_{limit}"
        
        # 检查缓存
        if use_cache and cache_key in self.cache:
            last_time = self.last_cache_time.get(cache_key)
            if last_time and (datetime.now() - last_time).seconds < self.cache_expire_time:
                return self.cache[cache_key].copy()
        
        # 获取数据提供者
        data_provider = self.get_provider(provider)
        if not data_provider:
            print(f"未找到数据提供者: {provider}")
            return pd.DataFrame()
        
        # 获取数据
        df = data_provider.get_historical_klines(symbol, interval, start_time, end_time, limit)
        
        # 缓存数据
        if use_cache and not df.empty:
            self.cache[cache_key] = df.copy()
            self.last_cache_time[cache_key] = datetime.now()
        
        return df
    
    def get_latest_price(self, symbol: str, provider: Optional[str] = None) -> float:
        """获取最新价格"""
        data_provider = self.get_provider(provider)
        if not data_provider:
            return 0.0
        return data_provider.get_latest_price(symbol)
    
    def get_market_ticker(self, symbol: str, provider: Optional[str] = None) -> Optional[MarketTicker]:
        """获取市场行情"""
        data_provider = self.get_provider(provider)
        if not data_provider:
            return None
        return data_provider.get_market_ticker(symbol)
    
    def subscribe_kline(self, symbol: str, interval: str, callback: Callable, provider: Optional[str] = None) -> None:
        """订阅K线数据"""
        data_provider = self.get_provider(provider)
        if not data_provider:
            print(f"未找到数据提供者: {provider}")
            return
        data_provider.subscribe_kline(symbol, interval, callback)
    
    def subscribe_ticker(self, symbol: str, callback: Callable, provider: Optional[str] = None) -> None:
        """订阅行情数据"""
        data_provider = self.get_provider(provider)
        if not data_provider:
            print(f"未找到数据提供者: {provider}")
            return
        data_provider.subscribe_ticker(symbol, callback)
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self.cache.clear()
        self.last_cache_time.clear()
        print("数据缓存已清除")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        return {
            'cache_count': len(self.cache),
            'cache_keys': list(self.cache.keys()),
            'cache_expire_time': self.cache_expire_time
        }