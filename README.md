# AutoTrader 量化交易系统 - 快速开始指南

## 🚀 系统概述

AutoTrader 是一个完整的量化自动交易系统，支持策略回测、模拟交易和实盘交易。系统采用模块化设计，易于扩展和维护。

## 📋 系统特性

- ✨ **模块化架构**: 插件式策略设计，易于添加新策略
- 🔄 **多模式支持**: 回测、模拟、实盘交易无缝切换
- 🛡️ **风险管理**: 全面的风险控制和监控机制
- 📊 **数据分析**: 详细的绩效分析和可视化图表
- 🌐 **多交易所**: 支持币安等主流交易所
- 📱 **实时监控**: 实时行情订阅和交易执行

## 🔧 安装和配置

### 1. 环境要求

- Python 3.8+
- pip 或 poetry

### 2. 安装依赖

```bash
# 使用pip安装
pip install -r requirements.txt

# 或使用poetry安装（推荐）
poetry install
```

### 3. 配置敏感信息

```bash
# 复制敏感信息模板
cp secrets.yml.template secrets.yml

# 编辑secrets.yml，填入真实的API密钥
# 注意：请妥善保管API密钥，不要提交到代码仓库
```

### 4. 修改配置文件

编辑 `config.yml` 文件，根据需要调整配置参数：

- 交易对和时间周期
- 风险管理参数
- 策略参数
- 日志级别等

## 🎯 快速开始

### 1. 查看可用命令

```bash
python main.py --help
```

### 2. 列出可用策略

```bash
python main.py strategies
```

### 3. 运行回测

```bash
# 基本回测
python main.py backtest \
  --strategy mean_reversion \
  --symbol BTCUSDT \
  --start 2023-01-01 \
  --end 2023-12-31

# 带自定义参数的回测
python main.py backtest \
  --strategy mean_reversion \
  --symbol BTCUSDT \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --timeframe 4h \
  --balance 50000
```

### 4. 运行模拟交易

```bash
# 模拟交易（不会真实下单）
python main.py live \
  --strategy mean_reversion \
  --symbol BTCUSDT \
  --dry-run
```

### 5. 运行实盘交易

```bash
# ⚠️ 注意：这会进行真实交易，请确保配置正确
python main.py live \
  --strategy mean_reversion \
  --symbol BTCUSDT \
  --timeframe 1h
```

### 6. 查看系统状态

```bash
python main.py status
```

## 📊 回测结果分析

回测完成后，系统会在 `backtest_results/` 目录下生成以下文件：

- `backtest_summary.json`: 回测摘要信息
- `trades.csv`: 详细交易记录
- `equity_curve.csv`: 权益曲线数据
- `equity_curve.html`: 权益曲线图表
- `performance_analysis.html`: 绩效分析图表
- `account_report.json`: 账户详细报告
- `risk_report.json`: 风险分析报告

## 🛠️ 策略开发

### 1. 创建新策略

```python
# 在 auto_trader/strategies/ 目录下创建新的策略文件
from .base import Strategy, StrategyConfig, TradeSignal, SignalType

class MyCustomStrategy(Strategy):
    def initialize(self) -> None:
        # 策略初始化逻辑
        pass
    
    def on_data(self, data: pd.DataFrame) -> List[TradeSignal]:
        # 策略主逻辑
        signals = []
        
        # 分析数据并生成交易信号
        # ...
        
        return signals
```

### 2. 注册策略

在 `main.py` 的 `TradingSystem` 类中添加新策略的创建逻辑。

### 3. 配置策略参数

在 `config.yml` 中添加策略的配置段：

```yaml
strategies:
  my_custom_strategy:
    enabled: true
    symbol: "BTCUSDT"
    timeframe: "1h"
    parameters:
      # 策略特定参数
      param1: value1
      param2: value2
```

## 🔒 风险管理

系统内置了全面的风险管理机制：

### 1. 仓位控制

- 单个仓位最大比例限制
- 总仓位比例限制
- 单个交易对最大仓位数限制

### 2. 损失控制

- 每日最大损失比例
- 总最大损失比例
- 最大回撤比例

### 3. 交易频率控制

- 每小时最大交易次数
- 每日最大交易次数
- 最小交易间隔时间

### 4. 价格保护

- 最大价格偏离比例
- 最小/最大订单价值

可在 `config.yml` 的 `risk_management` 段中调整这些参数。

## 📝 日志和监控

### 1. 日志配置

系统支持多级别日志记录：

- 文件日志：保存到 `logs/trading.log`
- 控制台日志：实时显示
- 日志轮转：自动管理日志文件大小

### 2. 监控指标

系统实时监控以下指标：

- 账户价值变化
- 持仓情况
- 交易执行情况
- 风险指标
- 策略性能

## 🚨 注意事项

### 1. 安全提醒

- 妥善保管API密钥，不要泄露给他人
- 实盘交易前务必充分测试策略
- 建议先在测试网络上验证策略
- 设置合理的风险控制参数

### 2. 免责声明

- 本系统仅供学习和研究使用
- 量化交易存在风险，可能导致资金损失
- 使用前请充分了解相关风险
- 作者不对使用本系统造成的任何损失负责

## 🔧 故障排除

### 1. 常见问题

**问题**: 无法连接到交易所API
**解决**: 检查网络连接和API密钥配置

**问题**: 回测数据不足
**解决**: 调整回测时间范围或检查数据源配置

**问题**: 策略不生成信号
**解决**: 检查策略参数和市场数据质量

### 2. 日志检查

查看日志文件 `logs/trading.log` 获取详细错误信息。

### 3. 调试模式

使用 `--debug` 参数获取更详细的错误信息：

```bash
python main.py backtest --strategy mean_reversion --symbol BTCUSDT --start 2023-01-01 --end 2023-12-31 --debug
```

## 📚 进阶功能

### 1. 多策略组合

可以同时运行多个策略，系统会自动管理策略间的交互。

### 2. 自定义指标

在 `auto_trader/utils/data_utils.py` 中添加自定义技术指标。

### 3. 通知集成

配置邮件或Telegram通知，及时获取交易信息。

### 4. 数据库集成

配置数据库连接，持久化存储交易数据。

## 🤝 贡献指南

欢迎贡献代码和建议：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub Issues
- 邮箱：trading@fan.com

---

🎉 **祝您交易顺利，收益满满！**