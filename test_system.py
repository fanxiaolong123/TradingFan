#!/usr/bin/env python3
"""
系统测试脚本

用于测试AutoTrader系统的基本功能是否正常
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有模块的导入"""
    print("🔍 测试模块导入...")
    
    try:
        # 测试工具模块
        from auto_trader.utils import get_config, get_logger
        print("✅ utils模块导入成功")
        
        # 测试策略模块
        from auto_trader.strategies import Strategy, MeanReversionStrategy, StrategyConfig
        print("✅ strategies模块导入成功")
        
        # 测试核心模块
        from auto_trader.core import (
            DataManager, BinanceDataProvider,
            BinanceBroker, SimulatedBroker,
            AccountManager, RiskManager,
            BacktestEngine
        )
        print("✅ core模块导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_loading():
    """测试配置加载"""
    print("\n🔍 测试配置加载...")
    
    try:
        from auto_trader.utils import get_config
        
        config = get_config()
        
        # 测试基本配置项
        system_name = config.get('system.name')
        data_source = config.get('data_sources.default')
        
        print(f"✅ 系统名称: {system_name}")
        print(f"✅ 默认数据源: {data_source}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logger():
    """测试日志系统"""
    print("\n🔍 测试日志系统...")
    
    try:
        from auto_trader.utils import get_logger
        
        logger = get_logger("test")
        logger.info("这是一条测试日志")
        
        print("✅ 日志系统正常")
        return True
        
    except Exception as e:
        print(f"❌ 日志系统失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_creation():
    """测试策略创建"""
    print("\n🔍 测试策略创建...")
    
    try:
        from auto_trader.strategies import MeanReversionStrategy, StrategyConfig
        
        # 创建策略配置
        config = StrategyConfig(
            name="test_strategy",
            symbol="BTCUSDT", 
            timeframe="1h",
            parameters={
                'ma_period': 20,
                'deviation_threshold': 0.02
            }
        )
        
        # 创建策略实例
        strategy = MeanReversionStrategy(config)
        
        print(f"✅ 策略创建成功: {strategy.name}")
        return True
        
    except Exception as e:
        print(f"❌ 策略创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_manager():
    """测试数据管理器"""
    print("\n🔍 测试数据管理器...")
    
    try:
        from auto_trader.core import DataManager, BinanceDataProvider
        
        # 创建数据管理器
        data_manager = DataManager()
        
        # 添加数据提供者（不需要真实API密钥）
        provider = BinanceDataProvider()
        data_manager.add_provider('binance', provider, is_default=True)
        
        print("✅ 数据管理器创建成功")
        return True
        
    except Exception as e:
        print(f"❌ 数据管理器失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simulated_broker():
    """测试模拟经纪商"""
    print("\n🔍 测试模拟经纪商...")
    
    try:
        from auto_trader.core import SimulatedBroker
        
        # 创建模拟经纪商
        broker = SimulatedBroker(
            initial_balance={'USDT': 10000.0},
            commission_rate=0.001
        )
        
        # 获取账户余额
        balance = broker.get_account_balance()
        
        print(f"✅ 模拟经纪商创建成功，初始余额: {balance}")
        return True
        
    except Exception as e:
        print(f"❌ 模拟经纪商失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_account_manager():
    """测试账户管理器"""
    print("\n🔍 测试账户管理器...")
    
    try:
        from auto_trader.core import AccountManager
        
        # 创建账户管理器
        account_manager = AccountManager()
        account_manager.set_initial_balance({'USDT': 10000.0})
        
        # 获取账户摘要
        summary = account_manager.get_account_summary()
        
        print(f"✅ 账户管理器创建成功，账户类型: {summary['account_type']}")
        return True
        
    except Exception as e:
        print(f"❌ 账户管理器失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_risk_manager():
    """测试风险管理器"""
    print("\n🔍 测试风险管理器...")
    
    try:
        from auto_trader.core import RiskManager, RiskLimits
        
        # 创建风险限制
        risk_limits = RiskLimits()
        
        # 创建风险管理器
        risk_manager = RiskManager(risk_limits)
        
        # 获取风险摘要
        summary = risk_manager.get_risk_summary()
        
        print(f"✅ 风险管理器创建成功，当前风险级别: {summary['current_risk_level']}")
        return True
        
    except Exception as e:
        print(f"❌ 风险管理器失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_backtest_engine():
    """测试回测引擎"""
    print("\n🔍 测试回测引擎...")
    
    try:
        from auto_trader.core import BacktestEngine, DataManager
        
        # 创建数据管理器
        data_manager = DataManager()
        
        # 创建回测引擎
        backtest_engine = BacktestEngine(data_manager)
        
        print("✅ 回测引擎创建成功")
        return True
        
    except Exception as e:
        print(f"❌ 回测引擎失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("🚀 AutoTrader 系统测试")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_config_loading,
        test_logger,
        test_strategy_creation,
        test_data_manager,
        test_simulated_broker,
        test_account_manager,
        test_risk_manager,
        test_backtest_engine
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过！系统基本功能正常")
        return True
    else:
        print("⚠️ 部分测试失败，请检查相关模块")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)