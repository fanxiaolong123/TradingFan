"""
时间工具模块

这个模块提供了时间处理的各种工具函数，包括：
- 时区转换和处理
- 时间格式化和解析
- 交易时间判断
- 时间范围计算
"""

import pytz
from datetime import datetime, timedelta, time
from typing import Optional, Union, List, Tuple
import pandas as pd
import calendar

from .logger import get_logger

logger = get_logger(__name__)


class TimeUtils:
    """时间工具类"""
    
    # 主要时区
    TIMEZONES = {
        'UTC': pytz.UTC,
        'US/Eastern': pytz.timezone('US/Eastern'),
        'US/Central': pytz.timezone('US/Central'),
        'US/Mountain': pytz.timezone('US/Mountain'),
        'US/Pacific': pytz.timezone('US/Pacific'),
        'Europe/London': pytz.timezone('Europe/London'),
        'Europe/Paris': pytz.timezone('Europe/Paris'),
        'Asia/Tokyo': pytz.timezone('Asia/Tokyo'),
        'Asia/Shanghai': pytz.timezone('Asia/Shanghai'),
        'Asia/Hong_Kong': pytz.timezone('Asia/Hong_Kong'),
        'Australia/Sydney': pytz.timezone('Australia/Sydney'),
    }
    
    # 主要股票市场交易时间
    MARKET_HOURS = {
        'NYSE': {
            'timezone': 'US/Eastern',
            'open': time(9, 30),
            'close': time(16, 0),
            'weekdays': [0, 1, 2, 3, 4]  # 周一到周五
        },
        'NASDAQ': {
            'timezone': 'US/Eastern',
            'open': time(9, 30),
            'close': time(16, 0),
            'weekdays': [0, 1, 2, 3, 4]
        },
        'LSE': {  # 伦敦证券交易所
            'timezone': 'Europe/London',
            'open': time(8, 0),
            'close': time(16, 30),
            'weekdays': [0, 1, 2, 3, 4]
        },
        'TSE': {  # 东京证券交易所
            'timezone': 'Asia/Tokyo',
            'open': time(9, 0),
            'close': time(15, 0),
            'weekdays': [0, 1, 2, 3, 4]
        },
        'SSE': {  # 上海证券交易所
            'timezone': 'Asia/Shanghai',
            'open': time(9, 30),
            'close': time(15, 0),
            'weekdays': [0, 1, 2, 3, 4]
        },
        'CRYPTO': {  # 加密货币市场（24/7）
            'timezone': 'UTC',
            'open': time(0, 0),
            'close': time(23, 59),
            'weekdays': [0, 1, 2, 3, 4, 5, 6]  # 全周
        }
    }
    
    @staticmethod
    def get_current_time(timezone: str = 'UTC') -> datetime:
        """
        获取当前时间
        
        Args:
            timezone: 时区名称
            
        Returns:
            datetime: 当前时间
        """
        tz = TimeUtils.TIMEZONES.get(timezone, pytz.UTC)
        return datetime.now(tz)
    
    @staticmethod
    def convert_timezone(dt: datetime, 
                        from_tz: str = 'UTC', 
                        to_tz: str = 'UTC') -> datetime:
        """
        转换时区
        
        Args:
            dt: 要转换的时间
            from_tz: 源时区
            to_tz: 目标时区
            
        Returns:
            datetime: 转换后的时间
        """
        # 获取时区对象
        from_timezone = TimeUtils.TIMEZONES.get(from_tz, pytz.UTC)
        to_timezone = TimeUtils.TIMEZONES.get(to_tz, pytz.UTC)
        
        # 如果输入时间没有时区信息，假设为源时区
        if dt.tzinfo is None:
            dt = from_timezone.localize(dt)
        
        # 转换时区
        return dt.astimezone(to_timezone)
    
    @staticmethod
    def parse_time_string(time_str: str, 
                         format: str = '%Y-%m-%d %H:%M:%S',
                         timezone: str = 'UTC') -> datetime:
        """
        解析时间字符串
        
        Args:
            time_str: 时间字符串
            format: 时间格式
            timezone: 时区
            
        Returns:
            datetime: 解析后的时间
        """
        dt = datetime.strptime(time_str, format)
        tz = TimeUtils.TIMEZONES.get(timezone, pytz.UTC)
        return tz.localize(dt)
    
    @staticmethod
    def format_time(dt: datetime, 
                   format: str = '%Y-%m-%d %H:%M:%S',
                   timezone: str = None) -> str:
        """
        格式化时间
        
        Args:
            dt: 要格式化的时间
            format: 时间格式
            timezone: 目标时区
            
        Returns:
            str: 格式化后的时间字符串
        """
        if timezone:
            dt = TimeUtils.convert_timezone(dt, to_tz=timezone)
        
        return dt.strftime(format)
    
    @staticmethod
    def is_market_open(market: str = 'CRYPTO', 
                      check_time: Optional[datetime] = None) -> bool:
        """
        检查市场是否开盘
        
        Args:
            market: 市场名称
            check_time: 检查时间，None表示当前时间
            
        Returns:
            bool: 是否开盘
        """
        if market not in TimeUtils.MARKET_HOURS:
            logger.warning(f"未知市场: {market}")
            return False
        
        market_info = TimeUtils.MARKET_HOURS[market]
        
        # 获取检查时间
        if check_time is None:
            check_time = TimeUtils.get_current_time(market_info['timezone'])
        else:
            check_time = TimeUtils.convert_timezone(check_time, to_tz=market_info['timezone'])
        
        # 检查是否为工作日
        if check_time.weekday() not in market_info['weekdays']:
            return False
        
        # 检查是否在交易时间内
        current_time = check_time.time()
        return market_info['open'] <= current_time <= market_info['close']
    
    @staticmethod
    def get_market_hours(market: str = 'CRYPTO',
                        date: Optional[datetime] = None) -> Optional[Tuple[datetime, datetime]]:
        """
        获取市场交易时间
        
        Args:
            market: 市场名称
            date: 日期，None表示今天
            
        Returns:
            Tuple[datetime, datetime]: (开盘时间, 收盘时间)
        """
        if market not in TimeUtils.MARKET_HOURS:
            logger.warning(f"未知市场: {market}")
            return None
        
        market_info = TimeUtils.MARKET_HOURS[market]
        
        # 获取日期
        if date is None:
            date = TimeUtils.get_current_time(market_info['timezone'])
        else:
            date = TimeUtils.convert_timezone(date, to_tz=market_info['timezone'])
        
        # 检查是否为工作日
        if date.weekday() not in market_info['weekdays']:
            return None
        
        # 构建开盘和收盘时间
        open_time = datetime.combine(date.date(), market_info['open'])
        close_time = datetime.combine(date.date(), market_info['close'])
        
        # 添加时区信息
        tz = TimeUtils.TIMEZONES[market_info['timezone']]
        open_time = tz.localize(open_time)
        close_time = tz.localize(close_time)
        
        return open_time, close_time
    
    @staticmethod
    def get_next_market_open(market: str = 'CRYPTO') -> Optional[datetime]:
        """
        获取下一个市场开盘时间
        
        Args:
            market: 市场名称
            
        Returns:
            datetime: 下一个开盘时间
        """
        if market not in TimeUtils.MARKET_HOURS:
            logger.warning(f"未知市场: {market}")
            return None
        
        market_info = TimeUtils.MARKET_HOURS[market]
        current_time = TimeUtils.get_current_time(market_info['timezone'])
        
        # 对于24/7市场，总是开盘
        if market == 'CRYPTO':
            return current_time
        
        # 从今天开始检查
        check_date = current_time.date()
        
        for i in range(7):  # 最多检查一周
            check_datetime = datetime.combine(check_date, market_info['open'])
            check_datetime = TimeUtils.TIMEZONES[market_info['timezone']].localize(check_datetime)
            
            # 如果是工作日且时间未过
            if (check_datetime.weekday() in market_info['weekdays'] and 
                check_datetime > current_time):
                return check_datetime
            
            # 检查下一天
            check_date += timedelta(days=1)
        
        return None
    
    @staticmethod
    def time_until_market_open(market: str = 'CRYPTO') -> Optional[timedelta]:
        """
        计算距离市场开盘的时间
        
        Args:
            market: 市场名称
            
        Returns:
            timedelta: 距离开盘的时间
        """
        if TimeUtils.is_market_open(market):
            return timedelta(0)
        
        next_open = TimeUtils.get_next_market_open(market)
        if next_open is None:
            return None
        
        current_time = TimeUtils.get_current_time()
        return next_open - current_time
    
    @staticmethod
    def get_trading_days(start_date: datetime, 
                        end_date: datetime,
                        market: str = 'CRYPTO') -> List[datetime]:
        """
        获取交易日列表
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            market: 市场名称
            
        Returns:
            List[datetime]: 交易日列表
        """
        if market not in TimeUtils.MARKET_HOURS:
            logger.warning(f"未知市场: {market}")
            return []
        
        market_info = TimeUtils.MARKET_HOURS[market]
        trading_days = []
        
        current_date = start_date.date()
        end_date = end_date.date()
        
        while current_date <= end_date:
            if current_date.weekday() in market_info['weekdays']:
                trading_days.append(datetime.combine(current_date, time()))
            
            current_date += timedelta(days=1)
        
        return trading_days
    
    @staticmethod
    def get_time_range(start_time: datetime,
                      end_time: datetime,
                      interval: str = '1H') -> List[datetime]:
        """
        生成时间范围
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            interval: 时间间隔
            
        Returns:
            List[datetime]: 时间点列表
        """
        return pd.date_range(start=start_time, end=end_time, freq=interval).tolist()
    
    @staticmethod
    def round_time(dt: datetime, 
                   interval: str = '1H',
                   method: str = 'round') -> datetime:
        """
        时间取整
        
        Args:
            dt: 要取整的时间
            interval: 时间间隔
            method: 取整方法 ('round', 'floor', 'ceil')
            
        Returns:
            datetime: 取整后的时间
        """
        # 解析时间间隔
        if interval.endswith('min'):
            minutes = int(interval[:-3])
            freq = f'{minutes}T'
        elif interval.endswith('H'):
            hours = int(interval[:-1])
            freq = f'{hours}H'
        elif interval.endswith('D'):
            days = int(interval[:-1])
            freq = f'{days}D'
        else:
            freq = interval
        
        # 创建时间序列并取整
        ts = pd.Timestamp(dt)
        
        if method == 'round':
            rounded_ts = ts.round(freq)
        elif method == 'floor':
            rounded_ts = ts.floor(freq)
        elif method == 'ceil':
            rounded_ts = ts.ceil(freq)
        else:
            raise ValueError(f"不支持的取整方法: {method}")
        
        return rounded_ts.to_pydatetime()
    
    @staticmethod
    def get_time_difference(time1: datetime, 
                           time2: datetime,
                           unit: str = 'seconds') -> float:
        """
        计算时间差
        
        Args:
            time1: 时间1
            time2: 时间2
            unit: 时间单位 ('seconds', 'minutes', 'hours', 'days')
            
        Returns:
            float: 时间差
        """
        diff = time2 - time1
        
        if unit == 'seconds':
            return diff.total_seconds()
        elif unit == 'minutes':
            return diff.total_seconds() / 60
        elif unit == 'hours':
            return diff.total_seconds() / 3600
        elif unit == 'days':
            return diff.total_seconds() / 86400
        else:
            raise ValueError(f"不支持的时间单位: {unit}")
    
    @staticmethod
    def is_weekend(dt: datetime) -> bool:
        """
        判断是否为周末
        
        Args:
            dt: 时间
            
        Returns:
            bool: 是否为周末
        """
        return dt.weekday() >= 5  # 周六和周日
    
    @staticmethod
    def get_month_range(year: int, month: int) -> Tuple[datetime, datetime]:
        """
        获取月份时间范围
        
        Args:
            year: 年份
            month: 月份
            
        Returns:
            Tuple[datetime, datetime]: (月初, 月末)
        """
        start_date = datetime(year, month, 1)
        
        # 获取月份的最后一天
        last_day = calendar.monthrange(year, month)[1]
        end_date = datetime(year, month, last_day, 23, 59, 59)
        
        return start_date, end_date
    
    @staticmethod
    def get_quarter_range(year: int, quarter: int) -> Tuple[datetime, datetime]:
        """
        获取季度时间范围
        
        Args:
            year: 年份
            quarter: 季度 (1-4)
            
        Returns:
            Tuple[datetime, datetime]: (季度开始, 季度结束)
        """
        if quarter not in [1, 2, 3, 4]:
            raise ValueError("季度必须在1-4之间")
        
        # 季度开始月份
        start_month = (quarter - 1) * 3 + 1
        start_date = datetime(year, start_month, 1)
        
        # 季度结束月份
        end_month = quarter * 3
        last_day = calendar.monthrange(year, end_month)[1]
        end_date = datetime(year, end_month, last_day, 23, 59, 59)
        
        return start_date, end_date
    
    @staticmethod
    def get_year_range(year: int) -> Tuple[datetime, datetime]:
        """
        获取年份时间范围
        
        Args:
            year: 年份
            
        Returns:
            Tuple[datetime, datetime]: (年初, 年末)
        """
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31, 23, 59, 59)
        
        return start_date, end_date
    
    @staticmethod
    def add_business_days(dt: datetime, days: int) -> datetime:
        """
        添加工作日
        
        Args:
            dt: 基准时间
            days: 工作日数量
            
        Returns:
            datetime: 结果时间
        """
        current_date = dt.date()
        
        while days > 0:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:  # 周一到周五
                days -= 1
        
        return datetime.combine(current_date, dt.time())
    
    @staticmethod
    def get_age_in_units(dt: datetime, 
                        unit: str = 'days',
                        reference_time: Optional[datetime] = None) -> float:
        """
        计算时间年龄
        
        Args:
            dt: 时间点
            unit: 时间单位
            reference_time: 参考时间，None表示当前时间
            
        Returns:
            float: 年龄
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        return TimeUtils.get_time_difference(dt, reference_time, unit)


# 便捷函数
def now(timezone: str = 'UTC') -> datetime:
    """获取当前时间"""
    return TimeUtils.get_current_time(timezone)


def to_utc(dt: datetime, from_tz: str = 'UTC') -> datetime:
    """转换为UTC时间"""
    return TimeUtils.convert_timezone(dt, from_tz, 'UTC')


def from_utc(dt: datetime, to_tz: str = 'Asia/Shanghai') -> datetime:
    """从UTC时间转换"""
    return TimeUtils.convert_timezone(dt, 'UTC', to_tz)


def is_trading_time(market: str = 'CRYPTO') -> bool:
    """检查是否为交易时间"""
    return TimeUtils.is_market_open(market)


def format_duration(seconds: float) -> str:
    """格式化持续时间"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def parse_timeframe(timeframe: str) -> timedelta:
    """解析时间框架为timedelta"""
    if timeframe.endswith('s'):
        seconds = int(timeframe[:-1])
        return timedelta(seconds=seconds)
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
        return timedelta(minutes=minutes)
    elif timeframe.endswith('h'):
        hours = int(timeframe[:-1])
        return timedelta(hours=hours)
    elif timeframe.endswith('d'):
        days = int(timeframe[:-1])
        return timedelta(days=days)
    else:
        raise ValueError(f"无法解析时间框架: {timeframe}")


def get_period_start(dt: datetime, period: str) -> datetime:
    """获取周期开始时间"""
    if period == 'day':
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'week':
        # 周一为一周的开始
        days_since_monday = dt.weekday()
        start_of_week = dt - timedelta(days=days_since_monday)
        return start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'month':
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    elif period == 'year':
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError(f"不支持的周期: {period}")


def get_period_end(dt: datetime, period: str) -> datetime:
    """获取周期结束时间"""
    if period == 'day':
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == 'week':
        # 周日为一周的结束
        days_until_sunday = 6 - dt.weekday()
        end_of_week = dt + timedelta(days=days_until_sunday)
        return end_of_week.replace(hour=23, minute=59, second=59, microsecond=999999)
    elif period == 'month':
        # 下个月的第一天减去1微秒
        if dt.month == 12:
            next_month = dt.replace(year=dt.year + 1, month=1, day=1)
        else:
            next_month = dt.replace(month=dt.month + 1, day=1)
        return next_month - timedelta(microseconds=1)
    elif period == 'year':
        # 下一年的第一天减去1微秒
        next_year = dt.replace(year=dt.year + 1, month=1, day=1)
        return next_year - timedelta(microseconds=1)
    else:
        raise ValueError(f"不支持的周期: {period}")