"""
数据质量保障模块

提供数据质量检查、清洗和修复功能：
- 数据完整性检查
- 缺失值处理
- 异常值检测和处理
- 数据一致性验证
- 数据质量报告
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings


class DataQualityIssue(Enum):
    """数据质量问题类型"""
    MISSING_VALUES = "missing_values"           # 缺失值
    OUTLIERS = "outliers"                      # 异常值
    DUPLICATES = "duplicates"                  # 重复数据
    TIMESTAMP_GAPS = "timestamp_gaps"          # 时间戳间隔
    INVALID_OHLC = "invalid_ohlc"             # 无效OHLC关系
    NEGATIVE_VALUES = "negative_values"        # 负值
    ZERO_VOLUME = "zero_volume"               # 零成交量
    PRICE_JUMPS = "price_jumps"               # 价格跳跃
    INCONSISTENT_PRECISION = "precision"       # 精度不一致


@dataclass
class QualityIssue:
    """数据质量问题记录"""
    issue_type: DataQualityIssue
    severity: str                              # 严重程度: critical, warning, info
    description: str                           # 问题描述
    affected_rows: List[int]                   # 受影响的行
    column: Optional[str] = None               # 受影响的列
    details: Optional[Dict[str, Any]] = None   # 详细信息


@dataclass 
class QualityReport:
    """数据质量报告"""
    symbol: str
    timeframe: str
    total_rows: int
    processed_rows: int
    issues: List[QualityIssue]
    score: float                               # 质量评分 (0-100)
    processing_time: float                     # 处理时间(秒)
    timestamp: datetime
    
    def get_critical_issues(self) -> List[QualityIssue]:
        """获取严重问题"""
        return [issue for issue in self.issues if issue.severity == 'critical']
    
    def get_warning_issues(self) -> List[QualityIssue]:
        """获取警告问题"""
        return [issue for issue in self.issues if issue.severity == 'warning']


class DataValidator:
    """数据验证器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据验证器
        
        Args:
            config: 验证配置参数
        """
        # 默认配置
        self.config = {
            'max_price_jump_pct': 0.1,           # 最大价格跳跃百分比
            'max_volume_multiplier': 10.0,       # 最大成交量倍数
            'min_volume_threshold': 0.0,         # 最小成交量阈值
            'outlier_std_threshold': 3.0,        # 异常值标准差阈值
            'required_columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
            'allow_zero_volume': False,          # 是否允许零成交量
            'timestamp_tolerance_seconds': 60,   # 时间戳容忍度(秒)
        }
        
        # 更新配置
        if config:
            self.config.update(config)
    
    def validate_data_structure(self, data: pd.DataFrame) -> List[QualityIssue]:
        """
        验证数据结构
        
        Args:
            data: 要验证的数据
            
        Returns:
            List[QualityIssue]: 发现的问题列表
        """
        issues = []
        
        # 检查必需列
        missing_columns = []
        for col in self.config['required_columns']:
            if col not in data.columns:
                missing_columns.append(col)
        
        if missing_columns:
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.MISSING_VALUES,
                severity='critical',
                description=f"缺少必需列: {missing_columns}",
                affected_rows=[],
                details={'missing_columns': missing_columns}
            ))
        
        # 检查数据为空
        if len(data) == 0:
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.MISSING_VALUES,
                severity='critical',
                description="数据集为空",
                affected_rows=[]
            ))
        
        return issues
    
    def check_missing_values(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查缺失值"""
        issues = []
        
        for column in data.columns:
            if column in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                missing_mask = data[column].isna()
                missing_count = missing_mask.sum()
                
                if missing_count > 0:
                    missing_rows = data.index[missing_mask].tolist()
                    severity = 'critical' if column in ['close', 'timestamp'] else 'warning'
                    
                    issues.append(QualityIssue(
                        issue_type=DataQualityIssue.MISSING_VALUES,
                        severity=severity,
                        description=f"列 {column} 有 {missing_count} 个缺失值",
                        affected_rows=missing_rows,
                        column=column,
                        details={'missing_count': missing_count, 'missing_ratio': missing_count / len(data)}
                    ))
        
        return issues
    
    def check_ohlc_consistency(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查OHLC数据一致性"""
        issues = []
        
        if not all(col in data.columns for col in ['open', 'high', 'low', 'close']):
            return issues
        
        # 检查 high >= max(open, close) 和 low <= min(open, close)
        invalid_high = (data['high'] < data[['open', 'close']].max(axis=1))
        invalid_low = (data['low'] > data[['open', 'close']].min(axis=1))
        
        if invalid_high.any():
            invalid_rows = data.index[invalid_high].tolist()
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.INVALID_OHLC,
                severity='critical',
                description=f"高价低于开盘价或收盘价的行数: {len(invalid_rows)}",
                affected_rows=invalid_rows,
                details={'invalid_high_count': len(invalid_rows)}
            ))
        
        if invalid_low.any():
            invalid_rows = data.index[invalid_low].tolist()
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.INVALID_OHLC,
                severity='critical',
                description=f"低价高于开盘价或收盘价的行数: {len(invalid_rows)}",
                affected_rows=invalid_rows,
                details={'invalid_low_count': len(invalid_rows)}
            ))
        
        return issues
    
    def check_negative_values(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查负值"""
        issues = []
        
        price_columns = ['open', 'high', 'low', 'close']
        for column in price_columns:
            if column in data.columns:
                negative_mask = data[column] <= 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    negative_rows = data.index[negative_mask].tolist()
                    issues.append(QualityIssue(
                        issue_type=DataQualityIssue.NEGATIVE_VALUES,
                        severity='critical',
                        description=f"列 {column} 有 {negative_count} 个非正值",
                        affected_rows=negative_rows,
                        column=column,
                        details={'negative_count': negative_count}
                    ))
        
        # 检查成交量
        if 'volume' in data.columns:
            if not self.config['allow_zero_volume']:
                zero_volume_mask = data['volume'] <= 0
                zero_volume_count = zero_volume_mask.sum()
                
                if zero_volume_count > 0:
                    zero_volume_rows = data.index[zero_volume_mask].tolist()
                    issues.append(QualityIssue(
                        issue_type=DataQualityIssue.ZERO_VOLUME,
                        severity='warning',
                        description=f"零成交量行数: {zero_volume_count}",
                        affected_rows=zero_volume_rows,
                        column='volume',
                        details={'zero_volume_count': zero_volume_count}
                    ))
            
            negative_volume_mask = data['volume'] < 0
            negative_volume_count = negative_volume_mask.sum()
            
            if negative_volume_count > 0:
                negative_volume_rows = data.index[negative_volume_mask].tolist()
                issues.append(QualityIssue(
                    issue_type=DataQualityIssue.NEGATIVE_VALUES,
                    severity='critical',
                    description=f"负成交量行数: {negative_volume_count}",
                    affected_rows=negative_volume_rows,
                    column='volume',
                    details={'negative_volume_count': negative_volume_count}
                ))
        
        return issues
    
    def check_duplicates(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查重复数据"""
        issues = []
        
        if 'timestamp' in data.columns:
            # 检查时间戳重复
            duplicate_timestamps = data['timestamp'].duplicated()
            duplicate_count = duplicate_timestamps.sum()
            
            if duplicate_count > 0:
                duplicate_rows = data.index[duplicate_timestamps].tolist()
                issues.append(QualityIssue(
                    issue_type=DataQualityIssue.DUPLICATES,
                    severity='warning',
                    description=f"重复时间戳行数: {duplicate_count}",
                    affected_rows=duplicate_rows,
                    column='timestamp',
                    details={'duplicate_count': duplicate_count}
                ))
        
        # 检查完全重复的行
        completely_duplicate = data.duplicated()
        completely_duplicate_count = completely_duplicate.sum()
        
        if completely_duplicate_count > 0:
            duplicate_rows = data.index[completely_duplicate].tolist()
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.DUPLICATES,
                severity='warning',
                description=f"完全重复行数: {completely_duplicate_count}",
                affected_rows=duplicate_rows,
                details={'completely_duplicate_count': completely_duplicate_count}
            ))
        
        return issues
    
    def check_outliers(self, data: pd.DataFrame) -> List[QualityIssue]:
        """检查异常值"""
        issues = []
        
        # 检查价格跳跃
        if 'close' in data.columns and len(data) > 1:
            price_changes = data['close'].pct_change().abs()
            large_jumps = price_changes > self.config['max_price_jump_pct']
            jump_count = large_jumps.sum()
            
            if jump_count > 0:
                jump_rows = data.index[large_jumps].tolist()
                issues.append(QualityIssue(
                    issue_type=DataQualityIssue.PRICE_JUMPS,
                    severity='warning',
                    description=f"大幅价格跳跃行数: {jump_count}",
                    affected_rows=jump_rows,
                    column='close',
                    details={'jump_count': jump_count, 'max_jump': price_changes.max()}
                ))
        
        # 检查成交量异常
        if 'volume' in data.columns and len(data) > 1:
            volume_median = data['volume'].median()
            if volume_median > 0:
                volume_ratio = data['volume'] / volume_median
                high_volume = volume_ratio > self.config['max_volume_multiplier']
                high_volume_count = high_volume.sum()
                
                if high_volume_count > 0:
                    high_volume_rows = data.index[high_volume].tolist()
                    issues.append(QualityIssue(
                        issue_type=DataQualityIssue.OUTLIERS,
                        severity='info',
                        description=f"异常高成交量行数: {high_volume_count}",
                        affected_rows=high_volume_rows,
                        column='volume',
                        details={'high_volume_count': high_volume_count, 'max_ratio': volume_ratio.max()}
                    ))
        
        # 使用统计方法检查价格异常值
        price_columns = ['open', 'high', 'low', 'close']
        for column in price_columns:
            if column in data.columns and len(data) > 10:
                z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
                outliers = z_scores > self.config['outlier_std_threshold']
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    outlier_rows = data.index[outliers].tolist()
                    issues.append(QualityIssue(
                        issue_type=DataQualityIssue.OUTLIERS,
                        severity='info',
                        description=f"列 {column} 统计异常值行数: {outlier_count}",
                        affected_rows=outlier_rows,
                        column=column,
                        details={'outlier_count': outlier_count, 'max_z_score': z_scores.max()}
                    ))
        
        return issues
    
    def check_timestamp_gaps(self, data: pd.DataFrame, expected_interval: Optional[str] = None) -> List[QualityIssue]:
        """检查时间戳间隔"""
        issues = []
        
        if 'timestamp' not in data.columns or len(data) < 2:
            return issues
        
        # 确保时间戳已排序
        if not data['timestamp'].is_monotonic_increasing:
            issues.append(QualityIssue(
                issue_type=DataQualityIssue.TIMESTAMP_GAPS,
                severity='warning',
                description="时间戳未按升序排列",
                affected_rows=[],
                column='timestamp'
            ))
        
        # 计算时间间隔
        time_diffs = data['timestamp'].diff().dt.total_seconds()
        
        # 如果指定了期望间隔，检查是否符合
        if expected_interval:
            interval_map = {
                '1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '2h': 7200, '4h': 14400, '6h': 21600,
                '8h': 28800, '12h': 43200, '1d': 86400
            }
            
            expected_seconds = interval_map.get(expected_interval)
            if expected_seconds:
                tolerance = self.config['timestamp_tolerance_seconds']
                irregular_gaps = np.abs(time_diffs - expected_seconds) > tolerance
                # 排除第一行（没有前一行进行比较）
                irregular_gaps.iloc[0] = False
                
                gap_count = irregular_gaps.sum()
                if gap_count > 0:
                    gap_rows = data.index[irregular_gaps].tolist()
                    issues.append(QualityIssue(
                        issue_type=DataQualityIssue.TIMESTAMP_GAPS,
                        severity='warning',
                        description=f"时间间隔不规律的行数: {gap_count}",
                        affected_rows=gap_rows,
                        column='timestamp',
                        details={
                            'expected_interval': expected_interval,
                            'expected_seconds': expected_seconds,
                            'gap_count': gap_count
                        }
                    ))
        
        return issues


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据清洗器
        
        Args:
            config: 清洗配置参数
        """
        # 默认配置
        self.config = {
            'fill_method': 'forward',                # 缺失值填充方法: forward, backward, interpolate, drop
            'outlier_method': 'clip',                # 异常值处理方法: clip, remove, ignore
            'outlier_quantile_low': 0.01,            # 异常值下分位数
            'outlier_quantile_high': 0.99,           # 异常值上分位数
            'duplicate_keep': 'first',               # 重复值保留策略: first, last, False(删除所有)
            'interpolate_method': 'linear',          # 插值方法
            'max_consecutive_missing': 5,            # 最大连续缺失值
        }
        
        # 更新配置
        if config:
            self.config.update(config)
    
    def clean_data(self, data: pd.DataFrame, issues: List[QualityIssue]) -> pd.DataFrame:
        """
        根据质量问题清洗数据
        
        Args:
            data: 原始数据
            issues: 质量问题列表
            
        Returns:
            pd.DataFrame: 清洗后的数据
        """
        cleaned_data = data.copy()
        
        # 按严重程度排序，优先处理严重问题
        sorted_issues = sorted(issues, key=lambda x: {'critical': 0, 'warning': 1, 'info': 2}[x.severity])
        
        for issue in sorted_issues:
            if issue.issue_type == DataQualityIssue.MISSING_VALUES:
                cleaned_data = self._handle_missing_values(cleaned_data, issue)
            elif issue.issue_type == DataQualityIssue.DUPLICATES:
                cleaned_data = self._handle_duplicates(cleaned_data, issue)
            elif issue.issue_type == DataQualityIssue.OUTLIERS:
                cleaned_data = self._handle_outliers(cleaned_data, issue)
            elif issue.issue_type == DataQualityIssue.INVALID_OHLC:
                cleaned_data = self._handle_invalid_ohlc(cleaned_data, issue)
            elif issue.issue_type == DataQualityIssue.NEGATIVE_VALUES:
                cleaned_data = self._handle_negative_values(cleaned_data, issue)
            elif issue.issue_type == DataQualityIssue.ZERO_VOLUME:
                cleaned_data = self._handle_zero_volume(cleaned_data, issue)
            elif issue.issue_type == DataQualityIssue.PRICE_JUMPS:
                cleaned_data = self._handle_price_jumps(cleaned_data, issue)
        
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame, issue: QualityIssue) -> pd.DataFrame:
        """处理缺失值"""
        if not issue.column:
            return data
        
        column = issue.column
        
        if self.config['fill_method'] == 'forward':
            data[column] = data[column].fillna(method='ffill')
        elif self.config['fill_method'] == 'backward':
            data[column] = data[column].fillna(method='bfill')
        elif self.config['fill_method'] == 'interpolate':
            if column in ['open', 'high', 'low', 'close', 'volume']:
                data[column] = data[column].interpolate(method=self.config['interpolate_method'])
        elif self.config['fill_method'] == 'drop':
            data = data.dropna(subset=[column])
        
        return data
    
    def _handle_duplicates(self, data: pd.DataFrame, issue: QualityIssue) -> pd.DataFrame:
        """处理重复值"""
        if issue.column == 'timestamp':
            # 处理时间戳重复
            data = data.drop_duplicates(subset=['timestamp'], keep=self.config['duplicate_keep'])
        else:
            # 处理完全重复的行
            data = data.drop_duplicates(keep=self.config['duplicate_keep'])
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, issue: QualityIssue) -> pd.DataFrame:
        """处理异常值"""
        if not issue.column:
            return data
        
        column = issue.column
        
        if self.config['outlier_method'] == 'clip':
            # 使用分位数截断
            q_low = data[column].quantile(self.config['outlier_quantile_low'])
            q_high = data[column].quantile(self.config['outlier_quantile_high'])
            data[column] = data[column].clip(lower=q_low, upper=q_high)
        elif self.config['outlier_method'] == 'remove':
            # 移除异常值行
            data = data.drop(issue.affected_rows)
        # 'ignore' 选项不做处理
        
        return data
    
    def _handle_invalid_ohlc(self, data: pd.DataFrame, issue: QualityIssue) -> pd.DataFrame:
        """处理无效OHLC数据"""
        # 修正高价和低价
        for idx in issue.affected_rows:
            if idx in data.index:
                row = data.loc[idx]
                # 修正高价：设为开盘价和收盘价的最大值
                data.loc[idx, 'high'] = max(row['open'], row['close'], row['high'])
                # 修正低价：设为开盘价和收盘价的最小值
                data.loc[idx, 'low'] = min(row['open'], row['close'], row['low'])
        
        return data
    
    def _handle_negative_values(self, data: pd.DataFrame, issue: QualityIssue) -> pd.DataFrame:
        """处理负值"""
        if not issue.column:
            return data
        
        column = issue.column
        
        # 移除负值行或用前向填充替换
        if self.config['fill_method'] == 'drop':
            data = data[data[column] > 0]
        else:
            # 用前一个有效值填充
            mask = data[column] <= 0
            data.loc[mask, column] = np.nan
            data[column] = data[column].fillna(method='ffill')
        
        return data
    
    def _handle_zero_volume(self, data: pd.DataFrame, issue: QualityIssue) -> pd.DataFrame:
        """处理零成交量"""
        # 用最小正值替换零成交量
        min_volume = data[data['volume'] > 0]['volume'].min()
        if pd.notna(min_volume):
            data.loc[data['volume'] <= 0, 'volume'] = min_volume * 0.1
        
        return data
    
    def _handle_price_jumps(self, data: pd.DataFrame, issue: QualityIssue) -> pd.DataFrame:
        """处理价格跳跃"""
        if not issue.column or len(data) < 2:
            return data
        
        column = issue.column
        
        # 使用移动平均平滑价格跳跃
        for idx in issue.affected_rows:
            if idx in data.index and idx > 0:
                # 使用前后值的平均
                prev_idx = data.index[data.index.get_loc(idx) - 1]
                if data.index.get_loc(idx) < len(data) - 1:
                    next_idx = data.index[data.index.get_loc(idx) + 1]
                    data.loc[idx, column] = (data.loc[prev_idx, column] + data.loc[next_idx, column]) / 2
                else:
                    # 如果是最后一行，使用前一行的值
                    data.loc[idx, column] = data.loc[prev_idx, column]
        
        return data


class DataQualityManager:
    """数据质量管理器"""
    
    def __init__(self, validator_config: Optional[Dict[str, Any]] = None,
                 cleaner_config: Optional[Dict[str, Any]] = None):
        """
        初始化数据质量管理器
        
        Args:
            validator_config: 验证器配置
            cleaner_config: 清洗器配置
        """
        self.validator = DataValidator(validator_config)
        self.cleaner = DataCleaner(cleaner_config)
        self.reports: List[QualityReport] = []
    
    def process_data(self, data: pd.DataFrame, symbol: str = "UNKNOWN", 
                    timeframe: str = "UNKNOWN", expected_interval: Optional[str] = None,
                    auto_clean: bool = True) -> Tuple[pd.DataFrame, QualityReport]:
        """
        处理数据质量
        
        Args:
            data: 原始数据
            symbol: 交易对符号
            timeframe: 时间框架
            expected_interval: 期望的时间间隔
            auto_clean: 是否自动清洗
            
        Returns:
            Tuple[pd.DataFrame, QualityReport]: (处理后的数据, 质量报告)
        """
        start_time = datetime.now()
        original_rows = len(data)
        
        # 数据验证
        issues = []
        
        # 基础结构验证
        issues.extend(self.validator.validate_data_structure(data))
        
        # 如果数据结构有严重问题，停止处理
        critical_issues = [issue for issue in issues if issue.severity == 'critical']
        if critical_issues and any('缺少必需列' in issue.description or '数据集为空' in issue.description for issue in critical_issues):
            processing_time = (datetime.now() - start_time).total_seconds()
            report = QualityReport(
                symbol=symbol,
                timeframe=timeframe,
                total_rows=original_rows,
                processed_rows=0,
                issues=issues,
                score=0.0,
                processing_time=processing_time,
                timestamp=datetime.now()
            )
            return data, report
        
        # 详细质量检查
        issues.extend(self.validator.check_missing_values(data))
        issues.extend(self.validator.check_ohlc_consistency(data))
        issues.extend(self.validator.check_negative_values(data))
        issues.extend(self.validator.check_duplicates(data))
        issues.extend(self.validator.check_outliers(data))
        issues.extend(self.validator.check_timestamp_gaps(data, expected_interval))
        
        # 数据清洗
        cleaned_data = data
        if auto_clean and issues:
            cleaned_data = self.cleaner.clean_data(data, issues)
        
        # 计算质量评分
        score = self._calculate_quality_score(issues, original_rows)
        
        # 生成报告
        processing_time = (datetime.now() - start_time).total_seconds()
        report = QualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_rows=original_rows,
            processed_rows=len(cleaned_data),
            issues=issues,
            score=score,
            processing_time=processing_time,
            timestamp=datetime.now()
        )
        
        self.reports.append(report)
        
        return cleaned_data, report
    
    def _calculate_quality_score(self, issues: List[QualityIssue], total_rows: int) -> float:
        """
        计算数据质量评分
        
        Args:
            issues: 质量问题列表
            total_rows: 总行数
            
        Returns:
            float: 质量评分 (0-100)
        """
        if total_rows == 0:
            return 0.0
        
        score = 100.0
        
        # 根据问题严重程度扣分
        for issue in issues:
            affected_ratio = len(issue.affected_rows) / total_rows if issue.affected_rows else 0.1
            
            if issue.severity == 'critical':
                score -= 20 * affected_ratio
            elif issue.severity == 'warning':
                score -= 10 * affected_ratio
            elif issue.severity == 'info':
                score -= 5 * affected_ratio
        
        return max(0.0, score)
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """获取质量汇总信息"""
        if not self.reports:
            return {'total_reports': 0}
        
        total_reports = len(self.reports)
        avg_score = sum(report.score for report in self.reports) / total_reports
        
        # 统计问题类型
        issue_counts = {}
        for report in self.reports:
            for issue in report.issues:
                issue_type = issue.issue_type.value
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # 最近报告
        latest_report = max(self.reports, key=lambda r: r.timestamp)
        
        return {
            'total_reports': total_reports,
            'average_score': avg_score,
            'issue_counts': issue_counts,
            'latest_report': {
                'symbol': latest_report.symbol,
                'score': latest_report.score,
                'timestamp': latest_report.timestamp.isoformat(),
                'issues_count': len(latest_report.issues)
            }
        }
    
    def export_report(self, report: QualityReport, format: str = 'dict') -> Any:
        """
        导出质量报告
        
        Args:
            report: 质量报告
            format: 导出格式 ('dict', 'json', 'text')
            
        Returns:
            导出的报告数据
        """
        if format == 'dict':
            return {
                'symbol': report.symbol,
                'timeframe': report.timeframe,
                'total_rows': report.total_rows,
                'processed_rows': report.processed_rows,
                'score': report.score,
                'processing_time': report.processing_time,
                'timestamp': report.timestamp.isoformat(),
                'issues': [
                    {
                        'type': issue.issue_type.value,
                        'severity': issue.severity,
                        'description': issue.description,
                        'affected_rows_count': len(issue.affected_rows),
                        'column': issue.column,
                        'details': issue.details
                    }
                    for issue in report.issues
                ]
            }
        
        elif format == 'text':
            text = f"""
数据质量报告
============
交易对: {report.symbol}
时间框架: {report.timeframe}
总行数: {report.total_rows}
处理后行数: {report.processed_rows}
质量评分: {report.score:.2f}/100
处理时间: {report.processing_time:.3f}秒
报告时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

问题详情:
--------
"""
            if not report.issues:
                text += "未发现质量问题\n"
            else:
                for issue in report.issues:
                    text += f"[{issue.severity.upper()}] {issue.description}\n"
                    if issue.column:
                        text += f"  影响列: {issue.column}\n"
                    text += f"  影响行数: {len(issue.affected_rows)}\n"
                    if issue.details:
                        text += f"  详细信息: {issue.details}\n"
                    text += "\n"
            
            return text
        
        elif format == 'json':
            import json
            
            def convert_numpy_types(obj):
                """转换numpy类型为Python原生类型"""
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar types
                    return obj.item()
                else:
                    return obj
            
            dict_data = self.export_report(report, 'dict')
            clean_data = convert_numpy_types(dict_data)
            return json.dumps(clean_data, indent=2, ensure_ascii=False)
        
        else:
            raise ValueError(f"不支持的导出格式: {format}")