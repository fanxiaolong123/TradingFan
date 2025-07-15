# 🚀 AutoTrader 量化交易系统

专业级量化交易平台，支持多策略并行、智能风控和实时监控。

## ✨ 核心特性

- 🎯 **多策略并行**: 支持6个币种同时运行不同策略
- 🛡️ **专业风控**: VaR、CVaR、夏普比率等高级风险指标
- 💰 **智能资金管理**: Kelly公式、波动率目标等多种策略
- 📊 **报告生成**: HTML、PDF、JSON格式的交易报告
- 📱 **实时监控**: 交互式Web界面，实时数据展示
- 🔔 **多渠道通知**: Telegram、邮件、Webhook集成

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

### 1. 启动实时监控界面

```bash
# 启动Web界面
python scripts/run_realtime_dashboard.py
```

访问 http://localhost:8501 查看实时监控界面

### 2. 运行基础功能

```bash
# 查看命令帮助
python main.py --help

# 运行基础回测
python main.py backtest \
  --strategy mean_reversion \
  --symbol BTCUSDT \
  --start 2023-01-01 \
  --end 2023-12-31
```

## 📁 项目结构

```
TradingFan/
├── auto_trader/           # 核心交易系统
│   ├── core/             # 核心模块
│   ├── strategies/       # 策略模块  
│   ├── ui/              # 用户界面
│   └── utils/           # 工具模块
├── scripts/             # 启动脚本
├── docs/               # 文档
├── logs/               # 日志文件
├── config.yml          # 主配置
└── secrets.yml         # API密钥配置
```

## 🛠️ 支持的策略

- **均值回归策略**: 基于价格偏离均值的反转交易
- **可扩展架构**: 易于添加新的交易策略

## 🚨 免责声明

- 本系统仅供学习和研究使用
- 量化交易存在风险，可能导致资金损失  
- 使用前请充分了解相关风险
- 妥善保管API密钥，建议先在测试网络验证

---

🎉 **AutoTrader - 专业量化交易平台**