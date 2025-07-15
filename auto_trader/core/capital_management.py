"""
资金管理策略模块

该模块实现了多种资金管理策略，包括：
- Kelly公式动态仓位管理
- 等额投资策略
- 固定比例投资策略
- 动态风险调整策略
- 均值方差优化策略

每种策略都考虑了风险控制和收益最大化的平衡。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

from .account import AccountManager
from .risk import RiskManager, RiskMetrics


class PositionSizeMethod(Enum):
    """仓位大小计算方法枚举"""
    KELLY = "kelly"                    # Kelly公式
    FIXED_PERCENT = "fixed_percent"    # 固定比例
    EQUAL_WEIGHT = "equal_weight"      # 等权重
    RISK_PARITY = "risk_parity"        # 风险平价
    MEAN_VARIANCE = "mean_variance"    # 均值方差优化
    DYNAMIC_RISK = "dynamic_risk"      # 动态风险调整
    VOLATILITY_TARGET = "volatility_target"  # 波动率目标


@dataclass
class PositionSizeResult:
    """仓位大小计算结果"""
    suggested_size: float              # 建议仓位大小（资金比例）
    max_allowed_size: float            # 最大允许仓位大小
    confidence_level: float            # 信心水平
    risk_adjusted_size: float          # 风险调整后的仓位大小
    explanation: str                   # 计算说明
    metadata: Dict[str, Any]           # 元数据


class CapitalManagementStrategy(ABC):
    """资金管理策略抽象基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化资金管理策略
        
        Args:
            name: 策略名称
            config: 策略配置
        """
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"CapitalManagement.{name}")
        
        # 策略参数
        self.max_position_percent = config.get('max_position_percent', 0.25)
        self.min_position_percent = config.get('min_position_percent', 0.01)
        self.rebalance_threshold = config.get('rebalance_threshold', 0.05)
        
        # 历史数据
        self.performance_history: List[Dict] = []
        self.position_history: List[Dict] = []
        
        self.logger.info(f"资金管理策略 {name} 初始化完成")
    
    @abstractmethod
    def calculate_position_size(self, 
                               symbol: str,
                               signal_strength: float,
                               account_manager: AccountManager,
                               risk_manager: RiskManager,
                               market_data: Dict[str, Any]) -> PositionSizeResult:
        """
        计算仓位大小
        
        Args:
            symbol: 交易对符号
            signal_strength: 信号强度 (-1 到 1)
            account_manager: 账户管理器
            risk_manager: 风险管理器
            market_data: 市场数据
            
        Returns:
            PositionSizeResult: 仓位大小计算结果
        """
        pass
    
    def update_performance(self, symbol: str, returns: float, position_size: float):
        """
        更新策略表现
        
        Args:
            symbol: 交易对符号
            returns: 收益率
            position_size: 仓位大小
        """
        performance_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'returns': returns,
            'position_size': position_size,
            'strategy': self.name
        }
        self.performance_history.append(performance_record)
        
        # 保留最近1000条记录
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]


class KellyCapitalManagement(CapitalManagementStrategy):
    """Kelly公式资金管理策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Kelly", config)
        
        # Kelly公式参数
        self.lookback_period = config.get('lookback_period', 252)  # 回看期（天）
        self.min_kelly_fraction = config.get('min_kelly_fraction', 0.01)
        self.max_kelly_fraction = config.get('max_kelly_fraction', 0.25)
        self.kelly_multiplier = config.get('kelly_multiplier', 0.25)  # Kelly分数乘数
        
    def calculate_position_size(self, 
                               symbol: str,
                               signal_strength: float,
                               account_manager: AccountManager,
                               risk_manager: RiskManager,
                               market_data: Dict[str, Any]) -> PositionSizeResult:
        """使用Kelly公式计算仓位大小"""
        
        # 获取历史表现数据
        win_rate, avg_win, avg_loss = self._calculate_win_loss_stats(symbol)
        
        if win_rate <= 0 or avg_win <= 0 or avg_loss <= 0:
            # 没有足够历史数据时使用保守策略
            return PositionSizeResult(
                suggested_size=self.min_position_percent,
                max_allowed_size=self.max_position_percent,
                confidence_level=0.3,
                risk_adjusted_size=self.min_position_percent,
                explanation="历史数据不足，使用最小仓位",
                metadata={
                    'method': 'kelly',
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'data_available': len(self.performance_history)
                }
            )
        
        # 计算Kelly分数
        kelly_fraction = self._calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        
        # 应用Kelly乘数（保守处理）
        adjusted_kelly = kelly_fraction * self.kelly_multiplier
        
        # 考虑信号强度
        signal_adjusted_size = adjusted_kelly * abs(signal_strength)
        
        # 应用风险限制
        risk_adjusted_size = self._apply_risk_constraints(
            signal_adjusted_size, risk_manager, account_manager
        )
        
        # 确保在允许范围内
        final_size = max(self.min_kelly_fraction, 
                        min(self.max_kelly_fraction, risk_adjusted_size))
        
        return PositionSizeResult(
            suggested_size=final_size,
            max_allowed_size=self.max_position_percent,
            confidence_level=min(1.0, len(self.performance_history) / 100),
            risk_adjusted_size=risk_adjusted_size,
            explanation=f"Kelly分数: {kelly_fraction:.3f}, 调整后: {final_size:.3f}",
            metadata={
                'method': 'kelly',
                'raw_kelly': kelly_fraction,
                'adjusted_kelly': adjusted_kelly,
                'signal_strength': signal_strength,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'samples': len(self.performance_history)
            }
        )
    
    def _calculate_kelly_fraction(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """计算Kelly分数"""
        if avg_loss == 0:
            return 0.0
        
        # Kelly公式: f = (p*b - q) / b
        # 其中 p = 胜率, q = 败率, b = 赔率
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss  # 赔率
        
        kelly_fraction = (p * b - q) / b
        
        # 确保Kelly分数为正
        return max(0, kelly_fraction)
    
    def _calculate_win_loss_stats(self, symbol: str) -> Tuple[float, float, float]:
        """计算胜率和平均盈亏"""
        symbol_trades = [trade for trade in self.performance_history 
                        if trade['symbol'] == symbol]
        
        if len(symbol_trades) < 10:
            return 0.0, 0.0, 0.0
        
        # 计算胜率
        wins = [trade for trade in symbol_trades if trade['returns'] > 0]
        losses = [trade for trade in symbol_trades if trade['returns'] < 0]
        
        win_rate = len(wins) / len(symbol_trades)
        avg_win = np.mean([trade['returns'] for trade in wins]) if wins else 0.0
        avg_loss = abs(np.mean([trade['returns'] for trade in losses])) if losses else 0.0
        
        return win_rate, avg_win, avg_loss
    
    def _apply_risk_constraints(self, 
                               suggested_size: float,
                               risk_manager: RiskManager,
                               account_manager: AccountManager) -> float:
        """应用风险约束"""
        # 获取当前风险指标
        risk_metrics = risk_manager.calculate_risk_metrics(account_manager)
        
        # 基于当前回撤调整仓位
        drawdown_multiplier = 1.0
        if risk_metrics.current_drawdown_percent > 0.05:  # 5%以上回撤
            drawdown_multiplier = 1.0 - risk_metrics.current_drawdown_percent
        
        # 基于波动率调整仓位
        volatility_multiplier = 1.0
        if risk_metrics.market_volatility > 0.02:  # 日波动率超过2%
            volatility_multiplier = 0.02 / risk_metrics.market_volatility
        
        # 综合调整
        adjusted_size = suggested_size * drawdown_multiplier * volatility_multiplier
        
        return max(0, adjusted_size)


class FixedPercentCapitalManagement(CapitalManagementStrategy):
    """固定比例资金管理策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("FixedPercent", config)
        
        self.base_position_percent = config.get('base_position_percent', 0.1)
        self.signal_multiplier = config.get('signal_multiplier', 2.0)
        
    def calculate_position_size(self, 
                               symbol: str,
                               signal_strength: float,
                               account_manager: AccountManager,
                               risk_manager: RiskManager,
                               market_data: Dict[str, Any]) -> PositionSizeResult:
        """使用固定比例计算仓位大小"""
        
        # 基础仓位乘以信号强度
        base_size = self.base_position_percent * abs(signal_strength)
        
        # 应用风险约束
        risk_adjusted_size = self._apply_risk_adjustments(
            base_size, risk_manager, account_manager
        )
        
        # 确保在允许范围内
        final_size = max(self.min_position_percent,
                        min(self.max_position_percent, risk_adjusted_size))
        
        return PositionSizeResult(
            suggested_size=final_size,
            max_allowed_size=self.max_position_percent,
            confidence_level=0.8,  # 固定比例策略的信心水平
            risk_adjusted_size=risk_adjusted_size,
            explanation=f"固定比例: {self.base_position_percent:.1%} × 信号强度: {signal_strength:.2f}",
            metadata={
                'method': 'fixed_percent',
                'base_percent': self.base_position_percent,
                'signal_strength': signal_strength,
                'signal_multiplier': self.signal_multiplier
            }
        )
    
    def _apply_risk_adjustments(self, 
                               base_size: float,
                               risk_manager: RiskManager,
                               account_manager: AccountManager) -> float:
        """应用风险调整"""
        risk_metrics = risk_manager.calculate_risk_metrics(account_manager)
        
        # 基于总仓位调整
        if risk_metrics.total_position_percent > 0.7:
            adjustment = 0.7 / risk_metrics.total_position_percent
            base_size *= adjustment
        
        # 基于日损失调整
        if risk_metrics.daily_pnl_percent < -0.02:  # 日损失超过2%
            base_size *= 0.5  # 减少50%仓位
        
        return base_size


class EqualWeightCapitalManagement(CapitalManagementStrategy):
    """等权重资金管理策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("EqualWeight", config)
        
        self.target_positions = config.get('target_positions', 5)
        self.rebalance_frequency = config.get('rebalance_frequency', 7)  # 天
        
    def calculate_position_size(self, 
                               symbol: str,
                               signal_strength: float,
                               account_manager: AccountManager,
                               risk_manager: RiskManager,
                               market_data: Dict[str, Any]) -> PositionSizeResult:
        """使用等权重计算仓位大小"""
        
        # 获取当前持仓数量
        current_positions = len([pos for pos in account_manager.positions.values() 
                               if pos.quantity != 0])
        
        # 计算每个仓位的目标权重
        target_weight = 1.0 / self.target_positions
        
        # 基于信号强度调整
        signal_adjusted_weight = target_weight * abs(signal_strength)
        
        # 考虑当前仓位分布
        if current_positions >= self.target_positions:
            # 如果已达到目标仓位数，减少新仓位
            signal_adjusted_weight *= 0.5
        
        # 应用风险约束
        risk_adjusted_size = min(signal_adjusted_weight, self.max_position_percent)
        
        return PositionSizeResult(
            suggested_size=risk_adjusted_size,
            max_allowed_size=self.max_position_percent,
            confidence_level=0.7,
            risk_adjusted_size=risk_adjusted_size,
            explanation=f"等权重策略: {target_weight:.1%} × 信号强度: {signal_strength:.2f}",
            metadata={
                'method': 'equal_weight',
                'target_positions': self.target_positions,
                'current_positions': current_positions,
                'target_weight': target_weight,
                'signal_strength': signal_strength
            }
        )


class VolatilityTargetCapitalManagement(CapitalManagementStrategy):
    """波动率目标资金管理策略"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("VolatilityTarget", config)
        
        self.target_volatility = config.get('target_volatility', 0.15)  # 15%年化波动率
        self.volatility_lookback = config.get('volatility_lookback', 30)  # 30天
        self.min_volatility = config.get('min_volatility', 0.05)  # 最小波动率
        
    def calculate_position_size(self, 
                               symbol: str,
                               signal_strength: float,
                               account_manager: AccountManager,
                               risk_manager: RiskManager,
                               market_data: Dict[str, Any]) -> PositionSizeResult:
        """基于波动率目标计算仓位大小"""
        
        # 估算资产波动率
        asset_volatility = self._estimate_asset_volatility(symbol, market_data)
        
        if asset_volatility <= 0:
            asset_volatility = self.min_volatility
        
        # 计算波动率调整倍数
        volatility_multiplier = self.target_volatility / asset_volatility
        
        # 基础仓位
        base_size = self.max_position_percent * volatility_multiplier
        
        # 应用信号强度
        signal_adjusted_size = base_size * abs(signal_strength)
        
        # 确保在合理范围内
        final_size = max(self.min_position_percent,
                        min(self.max_position_percent, signal_adjusted_size))
        
        return PositionSizeResult(
            suggested_size=final_size,
            max_allowed_size=self.max_position_percent,
            confidence_level=0.8,
            risk_adjusted_size=signal_adjusted_size,
            explanation=f"波动率目标: {self.target_volatility:.1%}, 资产波动率: {asset_volatility:.1%}",
            metadata={
                'method': 'volatility_target',
                'target_volatility': self.target_volatility,
                'asset_volatility': asset_volatility,
                'volatility_multiplier': volatility_multiplier,
                'signal_strength': signal_strength
            }
        )
    
    def _estimate_asset_volatility(self, symbol: str, market_data: Dict[str, Any]) -> float:
        """估算资产波动率"""
        # 简化版本：使用市场数据中的波动率
        if 'volatility' in market_data:
            return market_data['volatility']
        
        # 如果没有波动率数据，使用历史表现估算
        symbol_trades = [trade for trade in self.performance_history 
                        if trade['symbol'] == symbol]
        
        if len(symbol_trades) < 10:
            return self.min_volatility
        
        returns = [trade['returns'] for trade in symbol_trades[-self.volatility_lookback:]]
        return np.std(returns) * np.sqrt(252)  # 年化波动率


class CapitalManager:
    """资金管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化资金管理器
        
        Args:
            config: 资金管理配置
        """
        self.config = config
        self.logger = logging.getLogger("CapitalManager")
        
        # 资金管理策略
        self.strategies: Dict[str, CapitalManagementStrategy] = {}
        self.current_strategy = config.get('default_strategy', 'fixed_percent')
        
        # 初始化策略
        self._initialize_strategies()
        
        # 仓位大小历史
        self.position_size_history: List[Dict] = []
        
        self.logger.info("资金管理器初始化完成")
    
    def _initialize_strategies(self):
        """初始化资金管理策略"""
        strategy_configs = self.config.get('strategies', {})
        
        # Kelly策略
        if 'kelly' in strategy_configs:
            self.strategies['kelly'] = KellyCapitalManagement(strategy_configs['kelly'])
        
        # 固定比例策略
        if 'fixed_percent' in strategy_configs:
            self.strategies['fixed_percent'] = FixedPercentCapitalManagement(strategy_configs['fixed_percent'])
        
        # 等权重策略
        if 'equal_weight' in strategy_configs:
            self.strategies['equal_weight'] = EqualWeightCapitalManagement(strategy_configs['equal_weight'])
        
        # 波动率目标策略
        if 'volatility_target' in strategy_configs:
            self.strategies['volatility_target'] = VolatilityTargetCapitalManagement(strategy_configs['volatility_target'])
        
        # 如果没有配置策略，使用默认策略
        if not self.strategies:
            self.strategies['fixed_percent'] = FixedPercentCapitalManagement({
                'base_position_percent': 0.1,
                'max_position_percent': 0.25
            })
    
    def calculate_position_size(self, 
                               symbol: str,
                               signal_strength: float,
                               account_manager: AccountManager,
                               risk_manager: RiskManager,
                               market_data: Dict[str, Any],
                               strategy_name: Optional[str] = None) -> PositionSizeResult:
        """
        计算仓位大小
        
        Args:
            symbol: 交易对符号
            signal_strength: 信号强度
            account_manager: 账户管理器
            risk_manager: 风险管理器
            market_data: 市场数据
            strategy_name: 指定策略名称
            
        Returns:
            PositionSizeResult: 仓位大小计算结果
        """
        # 选择策略
        strategy_name = strategy_name or self.current_strategy
        
        if strategy_name not in self.strategies:
            self.logger.warning(f"策略 {strategy_name} 不存在，使用默认策略")
            strategy_name = list(self.strategies.keys())[0]
        
        strategy = self.strategies[strategy_name]
        
        # 计算仓位大小
        result = strategy.calculate_position_size(
            symbol, signal_strength, account_manager, risk_manager, market_data
        )
        
        # 记录历史
        self.position_size_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal_strength': signal_strength,
            'suggested_size': result.suggested_size,
            'strategy': strategy_name,
            'confidence': result.confidence_level
        })
        
        # 保留最近1000条记录
        if len(self.position_size_history) > 1000:
            self.position_size_history = self.position_size_history[-1000:]
        
        self.logger.info(f"计算仓位大小 - {symbol}: {result.suggested_size:.2%} (策略: {strategy_name})")
        
        return result
    
    def update_strategy_performance(self, symbol: str, returns: float, position_size: float):
        """
        更新策略表现
        
        Args:
            symbol: 交易对符号
            returns: 收益率
            position_size: 仓位大小
        """
        for strategy in self.strategies.values():
            strategy.update_performance(symbol, returns, position_size)
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        获取策略摘要
        
        Returns:
            Dict[str, Any]: 策略摘要信息
        """
        summary = {
            'current_strategy': self.current_strategy,
            'available_strategies': list(self.strategies.keys()),
            'position_size_history_count': len(self.position_size_history),
            'strategies_performance': {}
        }
        
        # 计算各策略表现
        for name, strategy in self.strategies.items():
            if strategy.performance_history:
                returns = [trade['returns'] for trade in strategy.performance_history]
                summary['strategies_performance'][name] = {
                    'total_trades': len(strategy.performance_history),
                    'avg_return': np.mean(returns),
                    'return_std': np.std(returns),
                    'win_rate': len([r for r in returns if r > 0]) / len(returns),
                    'last_30_days': len([trade for trade in strategy.performance_history 
                                       if trade['timestamp'] >= datetime.now() - timedelta(days=30)])
                }
        
        return summary
    
    def switch_strategy(self, strategy_name: str) -> bool:
        """
        切换资金管理策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            bool: 是否切换成功
        """
        if strategy_name not in self.strategies:
            self.logger.error(f"策略 {strategy_name} 不存在")
            return False
        
        old_strategy = self.current_strategy
        self.current_strategy = strategy_name
        
        self.logger.info(f"资金管理策略已从 {old_strategy} 切换到 {strategy_name}")
        return True
    
    def export_performance_report(self) -> Dict[str, Any]:
        """
        导出性能报告
        
        Returns:
            Dict[str, Any]: 性能报告
        """
        report = {
            'capital_management_summary': self.get_strategy_summary(),
            'position_size_history': self.position_size_history[-100:],  # 最近100条记录
            'strategies_detailed_performance': {}
        }
        
        # 详细策略表现
        for name, strategy in self.strategies.items():
            if strategy.performance_history:
                performance_data = strategy.performance_history[-100:]
                report['strategies_detailed_performance'][name] = {
                    'recent_performance': performance_data,
                    'configuration': strategy.config
                }
        
        report['report_time'] = datetime.now().isoformat()
        
        return report