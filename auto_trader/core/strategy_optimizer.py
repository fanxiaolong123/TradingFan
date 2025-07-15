#!/usr/bin/env python3
"""
策略优化器模块

提供策略参数优化、多周期回测、性能评估等功能
支持网格搜索、贝叶斯优化等多种优化方法
"""

import sys
import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auto_trader.strategies.base import StrategyConfig
from auto_trader.strategies.momentum import MomentumStrategy
from auto_trader.strategies.mean_reversion import MeanReversionStrategy
from auto_trader.strategies.trend_following import TrendFollowingStrategy
from auto_trader.strategies.breakout import BreakoutStrategy

# 尝试导入回测引擎
try:
    from real_historical_backtest import RealHistoricalBacktester
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False
    print("⚠️ 回测引擎不可用，请检查路径")

warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """优化配置类"""
    # 基础配置
    strategy_name: str                              # 策略名称
    symbol: str                                     # 交易对
    timeframe: str                                  # 时间周期
    
    # 时间范围
    start_date: datetime                            # 开始时间
    end_date: datetime                              # 结束时间
    
    # 优化参数
    param_ranges: Dict[str, List[Any]]              # 参数范围
    optimization_method: str = "grid_search"        # 优化方法
    max_iterations: int = 1000                      # 最大迭代次数
    
    # 评估指标
    target_metric: str = "sharpe_ratio"             # 目标指标
    performance_threshold: Dict[str, float] = field(default_factory=dict)  # 性能阈值
    
    # 并行配置
    n_jobs: int = 4                                 # 并行任务数
    
    # 结果保存
    save_results: bool = True                       # 是否保存结果
    results_dir: str = "optimization_results"       # 结果目录


@dataclass
class OptimizationResult:
    """优化结果类"""
    # 参数配置
    strategy_name: str                              # 策略名称
    symbol: str                                     # 交易对
    timeframe: str                                  # 时间周期
    params: Dict[str, Any]                          # 最优参数
    
    # 性能指标
    total_return: float                             # 总收益率
    annualized_return: float                        # 年化收益率
    volatility: float                               # 波动率
    sharpe_ratio: float                             # 夏普比率
    sortino_ratio: float                            # 索提诺比率
    max_drawdown: float                             # 最大回撤
    win_rate: float                                 # 胜率
    profit_factor: float                            # 盈亏比
    
    # 风险指标
    var_95: float                                   # 95% VaR
    cvar_95: float                                  # 95% CVaR
    calmar_ratio: float                             # 卡尔马比率
    
    # 交易统计
    total_trades: int                               # 总交易数
    avg_win: float                                  # 平均盈利
    avg_loss: float                                 # 平均亏损
    
    # 时间信息
    backtest_period: str                            # 回测周期
    optimization_time: float                        # 优化耗时
    
    # 通过性评估
    performance_score: float                        # 综合评分
    meets_threshold: bool                           # 是否达标
    
    # 额外信息
    notes: str = ""                                 # 备注


class StrategyOptimizer:
    """策略优化器"""
    
    def __init__(self, config: OptimizationConfig):
        """
        初始化策略优化器
        
        Args:
            config: 优化配置
        """
        self.config = config
        self.backtest_engine = None
        
        # 初始化回测引擎
        if BACKTEST_AVAILABLE:
            self.backtest_engine = RealHistoricalBacktester(
                initial_capital=100000.0,
                commission_rate=0.001
            )
        
        # 创建结果目录
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # 策略映射
        self.strategy_map = {
            'momentum': MomentumStrategy,
            'mean_reversion': MeanReversionStrategy,
            'trend_following': TrendFollowingStrategy,
            'breakout': BreakoutStrategy
        }
        
        # 性能阈值默认值
        self.default_thresholds = {
            'win_rate': 0.60,                          # 胜率 ≥ 60%
            'annualized_return': 0.30,                 # 年化收益率 ≥ 30%
            'max_drawdown': -0.20,                     # 最大回撤 ≤ 20%
            'sharpe_ratio': 1.0,                       # 夏普比率 ≥ 1.0
            'total_trades': 10                         # 最少交易数
        }
        
        # 合并用户阈值
        self.performance_threshold = {**self.default_thresholds, **config.performance_threshold}
        
        logger.info(f"✅ 策略优化器初始化完成")
        logger.info(f"📊 策略: {config.strategy_name}")
        logger.info(f"🪙 交易对: {config.symbol}")
        logger.info(f"⏱️ 时间周期: {config.timeframe}")
        logger.info(f"📅 优化期间: {config.start_date.strftime('%Y-%m-%d')} 到 {config.end_date.strftime('%Y-%m-%d')}")
    
    def generate_parameter_combinations(self) -> List[Dict[str, Any]]:
        """
        生成参数组合
        
        Returns:
            List[Dict[str, Any]]: 参数组合列表
        """
        if self.config.optimization_method == "grid_search":
            return self._generate_grid_combinations()
        elif self.config.optimization_method == "random_search":
            return self._generate_random_combinations()
        else:
            raise ValueError(f"不支持的优化方法: {self.config.optimization_method}")
    
    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """生成网格搜索参数组合"""
        param_names = list(self.config.param_ranges.keys())
        param_values = list(self.config.param_ranges.values())
        
        combinations = []
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, combination))
            combinations.append(param_dict)
        
        logger.info(f"📊 网格搜索生成 {len(combinations)} 个参数组合")
        return combinations
    
    def _generate_random_combinations(self) -> List[Dict[str, Any]]:
        """生成随机搜索参数组合"""
        combinations = []
        
        for _ in range(self.config.max_iterations):
            param_dict = {}
            for param_name, param_range in self.config.param_ranges.items():
                if isinstance(param_range[0], int):
                    # 整数参数
                    value = np.random.randint(param_range[0], param_range[1] + 1)
                else:
                    # 浮点数参数
                    value = np.random.uniform(param_range[0], param_range[1])
                param_dict[param_name] = value
            
            combinations.append(param_dict)
        
        logger.info(f"📊 随机搜索生成 {len(combinations)} 个参数组合")
        return combinations
    
    def evaluate_single_combination(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        评估单个参数组合
        
        Args:
            params: 参数组合
            
        Returns:
            Optional[Dict[str, Any]]: 评估结果
        """
        try:
            # 创建策略配置
            strategy_config = StrategyConfig(
                name=f"{self.config.strategy_name}_{self.config.symbol}",
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                parameters=params
            )
            
            # 运行回测
            if not self.backtest_engine:
                logger.error("❌ 回测引擎不可用")
                return None
            
            metrics = self.backtest_engine.run_backtest(
                strategy_config=strategy_config,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            
            # 提取关键指标
            result = {
                'params': params,
                'total_return': metrics.total_return,
                'annualized_return': metrics.annualized_return,
                'volatility': metrics.volatility,
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'calmar_ratio': metrics.calmar_ratio,
                'total_trades': metrics.total_trades,
                'avg_win': metrics.avg_win,
                'avg_loss': metrics.avg_loss
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ 参数组合评估失败: {params} - {str(e)}")
            return None
    
    def optimize(self) -> OptimizationResult:
        """
        执行策略优化
        
        Returns:
            OptimizationResult: 优化结果
        """
        logger.info(f"🚀 开始策略优化...")
        start_time = datetime.now()
        
        # 生成参数组合
        combinations = self.generate_parameter_combinations()
        
        if not combinations:
            raise ValueError("没有生成有效的参数组合")
        
        # 并行评估
        logger.info(f"🔄 开始并行评估 {len(combinations)} 个参数组合...")
        
        results = []
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.config.n_jobs) as executor:
            # 提交任务
            futures = {executor.submit(self.evaluate_single_combination, params): params 
                      for params in combinations}
            
            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
                
                completed += 1
                if completed % max(1, len(combinations) // 10) == 0:
                    progress = completed / len(combinations) * 100
                    logger.info(f"   进度: {progress:.1f}% ({completed}/{len(combinations)})")
        
        logger.info(f"✅ 评估完成，有效结果: {len(results)}/{len(combinations)}")
        
        if not results:
            raise ValueError("没有获得有效的优化结果")
        
        # 找到最优结果
        best_result = self._find_best_result(results)
        
        # 计算优化耗时
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # 构建最终结果
        final_result = OptimizationResult(
            strategy_name=self.config.strategy_name,
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            params=best_result['params'],
            total_return=best_result['total_return'],
            annualized_return=best_result['annualized_return'],
            volatility=best_result['volatility'],
            sharpe_ratio=best_result['sharpe_ratio'],
            sortino_ratio=best_result['sortino_ratio'],
            max_drawdown=best_result['max_drawdown'],
            win_rate=best_result['win_rate'],
            profit_factor=best_result['profit_factor'],
            var_95=best_result['var_95'],
            cvar_95=best_result['cvar_95'],
            calmar_ratio=best_result['calmar_ratio'],
            total_trades=best_result['total_trades'],
            avg_win=best_result['avg_win'],
            avg_loss=best_result['avg_loss'],
            backtest_period=f"{self.config.start_date.strftime('%Y-%m-%d')} 到 {self.config.end_date.strftime('%Y-%m-%d')}",
            optimization_time=optimization_time,
            performance_score=self._calculate_performance_score(best_result),
            meets_threshold=self._meets_performance_threshold(best_result)
        )
        
        # 保存结果
        if self.config.save_results:
            self._save_optimization_results(final_result, results)
        
        return final_result
    
    def _find_best_result(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        找到最佳结果
        
        Args:
            results: 所有结果
            
        Returns:
            Dict[str, Any]: 最佳结果
        """
        if self.config.target_metric == "sharpe_ratio":
            return max(results, key=lambda x: x['sharpe_ratio'])
        elif self.config.target_metric == "annualized_return":
            return max(results, key=lambda x: x['annualized_return'])
        elif self.config.target_metric == "calmar_ratio":
            return max(results, key=lambda x: x['calmar_ratio'])
        elif self.config.target_metric == "win_rate":
            return max(results, key=lambda x: x['win_rate'])
        else:
            # 默认使用夏普比率
            return max(results, key=lambda x: x['sharpe_ratio'])
    
    def _calculate_performance_score(self, result: Dict[str, Any]) -> float:
        """
        计算综合性能评分
        
        Args:
            result: 单个结果
            
        Returns:
            float: 综合评分 (0-100)
        """
        score = 0.0
        
        # 收益率评分 (30%)
        if result['annualized_return'] >= 0.30:
            score += 30
        elif result['annualized_return'] >= 0.20:
            score += 20
        elif result['annualized_return'] >= 0.10:
            score += 10
        
        # 夏普比率评分 (25%)
        if result['sharpe_ratio'] >= 2.0:
            score += 25
        elif result['sharpe_ratio'] >= 1.5:
            score += 20
        elif result['sharpe_ratio'] >= 1.0:
            score += 15
        elif result['sharpe_ratio'] >= 0.5:
            score += 10
        
        # 最大回撤评分 (20%)
        if result['max_drawdown'] >= -0.10:
            score += 20
        elif result['max_drawdown'] >= -0.15:
            score += 15
        elif result['max_drawdown'] >= -0.20:
            score += 10
        elif result['max_drawdown'] >= -0.30:
            score += 5
        
        # 胜率评分 (15%)
        if result['win_rate'] >= 0.70:
            score += 15
        elif result['win_rate'] >= 0.60:
            score += 12
        elif result['win_rate'] >= 0.50:
            score += 8
        elif result['win_rate'] >= 0.40:
            score += 5
        
        # 交易频率评分 (10%)
        if result['total_trades'] >= 100:
            score += 10
        elif result['total_trades'] >= 50:
            score += 8
        elif result['total_trades'] >= 20:
            score += 5
        elif result['total_trades'] >= 10:
            score += 3
        
        return min(score, 100.0)
    
    def _meets_performance_threshold(self, result: Dict[str, Any]) -> bool:
        """
        检查是否满足性能阈值
        
        Args:
            result: 单个结果
            
        Returns:
            bool: 是否达标
        """
        checks = [
            result['win_rate'] >= self.performance_threshold['win_rate'],
            result['annualized_return'] >= self.performance_threshold['annualized_return'],
            result['max_drawdown'] >= self.performance_threshold['max_drawdown'],
            result['sharpe_ratio'] >= self.performance_threshold['sharpe_ratio'],
            result['total_trades'] >= self.performance_threshold['total_trades']
        ]
        
        return all(checks)
    
    def _save_optimization_results(self, final_result: OptimizationResult, all_results: List[Dict[str, Any]]):
        """
        保存优化结果
        
        Args:
            final_result: 最终结果
            all_results: 所有结果
        """
        # 保存最优结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.strategy_name}_{self.config.symbol}_{self.config.timeframe}_{timestamp}"
        
        # 保存详细结果
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(self.results_dir / f"{filename}_detailed.csv", index=False)
        
        # 保存最优结果摘要
        summary = {
            'strategy_name': final_result.strategy_name,
            'symbol': final_result.symbol,
            'timeframe': final_result.timeframe,
            'backtest_period': final_result.backtest_period,
            'optimization_time': final_result.optimization_time,
            'best_params': final_result.params,
            'performance_metrics': {
                'total_return': final_result.total_return,
                'annualized_return': final_result.annualized_return,
                'sharpe_ratio': final_result.sharpe_ratio,
                'max_drawdown': final_result.max_drawdown,
                'win_rate': final_result.win_rate,
                'total_trades': final_result.total_trades
            },
            'performance_score': final_result.performance_score,
            'meets_threshold': final_result.meets_threshold
        }
        
        # 保存为JSON
        import json
        with open(self.results_dir / f"{filename}_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📁 优化结果已保存到: {self.results_dir / filename}")
    
    def generate_optimization_report(self, result: OptimizationResult) -> str:
        """
        生成优化报告
        
        Args:
            result: 优化结果
            
        Returns:
            str: 格式化报告
        """
        report = f"""
📊 策略参数优化报告 - {result.strategy_name}
{'='*80}

🎯 优化配置
--------------------------------------------------
策略名称: {result.strategy_name}
交易对: {result.symbol}
时间周期: {result.timeframe}
回测期间: {result.backtest_period}
优化耗时: {result.optimization_time:.2f}秒

🏆 最优参数
--------------------------------------------------
"""
        
        for param, value in result.params.items():
            report += f"{param}: {value}\n"
        
        report += f"""
📈 性能指标
--------------------------------------------------
总收益率: {result.total_return*100:.2f}%
年化收益率: {result.annualized_return*100:.2f}%
夏普比率: {result.sharpe_ratio:.3f}
索提诺比率: {result.sortino_ratio:.3f}
最大回撤: {result.max_drawdown*100:.2f}%
卡尔马比率: {result.calmar_ratio:.3f}

🔄 交易统计
--------------------------------------------------
总交易数: {result.total_trades}
胜率: {result.win_rate*100:.1f}%
盈亏比: {result.profit_factor:.2f}
平均盈利: {result.avg_win:.2f} USDT
平均亏损: {result.avg_loss:.2f} USDT

⚠️ 风险指标
--------------------------------------------------
95% VaR: {result.var_95*100:.2f}%
95% CVaR: {result.cvar_95*100:.2f}%
波动率: {result.volatility*100:.2f}%

🎖️ 综合评估
--------------------------------------------------
性能评分: {result.performance_score:.1f}/100
是否达标: {'✅ 通过' if result.meets_threshold else '❌ 未通过'}

💡 优化建议
--------------------------------------------------
"""
        
        # 添加建议
        if result.performance_score >= 80:
            report += "✅ 策略表现优秀，建议上线测试\n"
        elif result.performance_score >= 60:
            report += "✅ 策略表现良好，可考虑优化后上线\n"
        else:
            report += "⚠️ 策略表现一般，建议重新优化参数\n"
        
        if result.sharpe_ratio < 1.0:
            report += "⚠️ 夏普比率偏低，建议优化风险控制\n"
        
        if result.max_drawdown < -0.2:
            report += "⚠️ 最大回撤较大，建议加强止损机制\n"
        
        if result.win_rate < 0.5:
            report += "⚠️ 胜率偏低，建议优化入场条件\n"
        
        return report


class MultiStrategyOptimizer:
    """多策略优化器"""
    
    def __init__(self, strategies: List[str], symbols: List[str], timeframes: List[str]):
        """
        初始化多策略优化器
        
        Args:
            strategies: 策略列表
            symbols: 交易对列表
            timeframes: 时间周期列表
        """
        self.strategies = strategies
        self.symbols = symbols
        self.timeframes = timeframes
        
        # 创建结果目录
        self.results_dir = Path("multi_strategy_optimization")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"🚀 多策略优化器初始化完成")
        logger.info(f"📊 策略数: {len(strategies)}")
        logger.info(f"🪙 交易对数: {len(symbols)}")
        logger.info(f"⏱️ 时间周期数: {len(timeframes)}")
    
    def optimize_all(self, base_config: Dict[str, Any]) -> List[OptimizationResult]:
        """
        优化所有策略组合
        
        Args:
            base_config: 基础配置
            
        Returns:
            List[OptimizationResult]: 所有优化结果
        """
        all_results = []
        total_combinations = len(self.strategies) * len(self.symbols) * len(self.timeframes)
        current_combination = 0
        
        logger.info(f"🎯 开始优化 {total_combinations} 个策略组合...")
        
        for strategy in self.strategies:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    current_combination += 1
                    
                    logger.info(f"🔄 优化进度: {current_combination}/{total_combinations}")
                    logger.info(f"📊 当前组合: {strategy} - {symbol} - {timeframe}")
                    
                    try:
                        # 获取策略特定的参数范围
                        param_ranges = self._get_strategy_param_ranges(strategy)
                        
                        # 创建优化配置
                        config = OptimizationConfig(
                            strategy_name=strategy,
                            symbol=symbol,
                            timeframe=timeframe,
                            param_ranges=param_ranges,
                            **base_config
                        )
                        
                        # 执行优化
                        optimizer = StrategyOptimizer(config)
                        result = optimizer.optimize()
                        
                        all_results.append(result)
                        
                        # 显示结果摘要
                        logger.info(f"✅ 优化完成: {result.performance_score:.1f}分, "
                                   f"年化收益率: {result.annualized_return*100:.2f}%, "
                                   f"夏普比率: {result.sharpe_ratio:.3f}")
                        
                    except Exception as e:
                        logger.error(f"❌ 优化失败: {strategy}-{symbol}-{timeframe} - {str(e)}")
                        continue
        
        # 保存汇总结果
        self._save_summary_results(all_results)
        
        return all_results
    
    def _get_strategy_param_ranges(self, strategy: str) -> Dict[str, List[Any]]:
        """
        获取策略参数范围
        
        Args:
            strategy: 策略名称
            
        Returns:
            Dict[str, List[Any]]: 参数范围
        """
        if strategy == "momentum":
            return {
                'short_ma_period': [12, 20, 24, 30],
                'long_ma_period': [26, 50, 72, 100],
                'rsi_period': [14, 21, 28],
                'rsi_overbought': [70, 75, 80],
                'rsi_oversold': [20, 25, 30],
                'momentum_period': [20, 30, 48],
                'momentum_threshold': [0.01, 0.02, 0.03],
                'position_size': [0.2, 0.3, 0.4],
                'stop_loss_pct': [0.02, 0.03, 0.05],
                'take_profit_pct': [0.04, 0.06, 0.08]
            }
        elif strategy == "mean_reversion":
            return {
                'ma_period': [20, 30, 48],
                'deviation_threshold': [0.015, 0.02, 0.025, 0.03],
                'min_volume': [10, 50, 100],
                'position_size': [0.2, 0.25, 0.3],
                'stop_loss_pct': [0.02, 0.025, 0.03],
                'take_profit_pct': [0.04, 0.05, 0.06]
            }
        elif strategy == "trend_following":
            return {
                'fast_ma_period': [10, 15, 20],
                'slow_ma_period': [30, 50, 100],
                'atr_period': [14, 20, 28],
                'atr_multiplier': [1.5, 2.0, 2.5],
                'position_size': [0.2, 0.3, 0.4],
                'stop_loss_pct': [0.03, 0.04, 0.05],
                'take_profit_pct': [0.06, 0.08, 0.10]
            }
        elif strategy == "breakout":
            return {
                'lookback_period': [20, 30, 50],
                'breakout_threshold': [0.01, 0.015, 0.02],
                'volume_threshold': [1.2, 1.5, 2.0],
                'position_size': [0.2, 0.3, 0.4],
                'stop_loss_pct': [0.02, 0.03, 0.04],
                'take_profit_pct': [0.04, 0.06, 0.08]
            }
        else:
            raise ValueError(f"不支持的策略: {strategy}")
    
    def _save_summary_results(self, results: List[OptimizationResult]):
        """
        保存汇总结果
        
        Args:
            results: 所有优化结果
        """
        # 创建汇总数据
        summary_data = []
        
        for result in results:
            summary_data.append({
                'strategy': result.strategy_name,
                'symbol': result.symbol,
                'timeframe': result.timeframe,
                'performance_score': result.performance_score,
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
                'meets_threshold': result.meets_threshold,
                'params': str(result.params)
            })
        
        # 保存为CSV
        summary_df = pd.DataFrame(summary_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 按性能评分排序
        summary_df = summary_df.sort_values('performance_score', ascending=False)
        summary_df.to_csv(self.results_dir / f"optimization_summary_{timestamp}.csv", index=False)
        
        # 生成排行榜
        self._generate_leaderboard(summary_df)
        
        logger.info(f"📁 汇总结果已保存到: {self.results_dir}")
    
    def _generate_leaderboard(self, summary_df: pd.DataFrame):
        """
        生成策略排行榜
        
        Args:
            summary_df: 汇总数据
        """
        leaderboard = f"""
🏆 策略优化排行榜
{'='*100}

📊 总体统计
--------------------------------------------------
参与策略数: {len(summary_df)}
达标策略数: {len(summary_df[summary_df['meets_threshold']])}
平均性能评分: {summary_df['performance_score'].mean():.1f}
最高性能评分: {summary_df['performance_score'].max():.1f}

🥇 TOP 10 策略
--------------------------------------------------
排名  策略组合                     性能评分  年化收益率  夏普比率  最大回撤  胜率    是否达标
"""
        
        top_10 = summary_df.head(10)
        
        for i, (_, row) in enumerate(top_10.iterrows(), 1):
            leaderboard += f"{i:2d}   {row['strategy']:<12} {row['symbol']:<8} {row['timeframe']:<3} "
            leaderboard += f"{row['performance_score']:>7.1f}  {row['annualized_return']*100:>8.2f}%  "
            leaderboard += f"{row['sharpe_ratio']:>7.3f}  {row['max_drawdown']*100:>7.2f}%  "
            leaderboard += f"{row['win_rate']*100:>5.1f}%  {'✅' if row['meets_threshold'] else '❌'}\n"
        
        leaderboard += f"""
📈 按策略类型统计
--------------------------------------------------
"""
        
        # 按策略分组统计
        strategy_stats = summary_df.groupby('strategy').agg({
            'performance_score': ['mean', 'max', 'count'],
            'annualized_return': 'mean',
            'sharpe_ratio': 'mean',
            'meets_threshold': 'sum'
        }).round(3)
        
        for strategy in strategy_stats.index:
            stats = strategy_stats.loc[strategy]
            leaderboard += f"{strategy:<15} 平均评分: {stats[('performance_score', 'mean')]:>5.1f}  "
            leaderboard += f"最高评分: {stats[('performance_score', 'max')]:>5.1f}  "
            leaderboard += f"测试数: {stats[('performance_score', 'count')]:>2.0f}  "
            leaderboard += f"达标数: {stats[('meets_threshold', 'sum')]:>2.0f}\n"
        
        # 保存排行榜
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(self.results_dir / f"leaderboard_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(leaderboard)
        
        print(leaderboard)