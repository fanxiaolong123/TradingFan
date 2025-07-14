你是一个全球顶尖的交易员、顶尖的量化程序开发者、顶尖的交易策略分析和优化专家、顶尖的交易封控专家。
现在我的目标是搭建一个完善的顶尖的量化自动交易系统。
我的偏好是每一步开发和每一个改动都和我确认后，你再进行改动，并向我详细解释改动内容。
文档：整个项目就一个总文档，把每次更新的内容，更新在这个文档中。
注释：每个类和方法都必须加注释；每一个对象的属性都要明确注释其含义；最好对每一行代码加注释（除非该行无任何意义，或者非常明确）；一整块代码块最好加注释，详细解释其作用。

系统的目标：

⸻

🧭 系统搭建总览（架构目标）
	•	核心功能模块：数据采集、策略模块、信号生成、回测引擎、订单执行、资金管理、可视化、UI
	•	技术目标：
	•	✨ 模块化 + 插件式策略架构
	•	✨ 同时支持回测 + 实盘
	•	✨ 多币种多策略独立运行
	•	✨ 支持实时行情订阅和自动下单
	•	✨ 支持 Web UI 展示回测、实时收益、当前仓位等

⸻

🧱 阶段一：项目初始化与框架设计

✅ 目标
	•	搭建项目目录结构
	•	设计模块化插件架构，便于添加策略

📁 推荐目录结构（可用 Python 包管理器如 poetry）

auto_trader/
├── core/                # 核心模块：数据、下单、回测、账户、风控
│   ├── data.py
│   ├── broker.py
│   ├── account.py
│   ├── risk.py
│   └── backtest.py
├── strategies/          # 策略模块（插件式）
│   ├── __init__.py
│   ├── base.py          # 抽象策略基类
│   └── mean_reversion.py
├── utils/               # 工具库，如日志、配置等
│   ├── logger.py
│   └── config.py
├── ui/                  # UI 展示
│   └── dashboard.py     # 后期加入 Gradio/Streamlit/FastAPI
├── tests/               # 单元测试
main.py                  # 启动入口
config.yml               # 配置文件
README.md

✅ 开始任务
	•	初始化 Python 项目（poetry / pipenv / venv）
	•	搭建目录结构
	•	写好 Strategy 抽象基类，定义统一接口，如：

class Strategy(ABC):
    def on_data(self, df: pd.DataFrame) -> List[TradeSignal]: ...
    def on_order_fill(self, order): ...



⸻

📊 阶段二：行情数据模块（data layer）

✅ 目标
	•	实时 + 历史数据拉取模块
	•	支持 Binance API（REST 和 WebSocket）
	•	后期接入 CCXT/KaiKo 等数据源

✅ 开始任务
	•	集成 Binance API
	•	支持拉取历史 K 线（OHLCV）
	•	封装成统一数据接口 get_ohlcv(symbol, interval) / subscribe(symbol)

⸻

📈 阶段三：回测引擎

✅ 目标
	•	支持向策略喂数据
	•	模拟交易撮合与资金变化
	•	支持手续费、滑点、资金限制
	•	输出交易记录 + 支持可视化

✅ 可视化建议
	•	用 matplotlib 绘制买卖点图
	•	支持盈亏曲线、胜率、夏普比率等指标输出

⸻

🧠 阶段四：策略开发与插件化

✅ 目标
	•	实现基础策略：如均值回归、动量、MACD、均线交叉
	•	策略独立文件，每个策略都继承 Strategy 基类
	•	支持配置文件动态加载策略

⸻

⚙️ 阶段五：实盘交易模块（broker）

✅ 目标
	•	接入 Binance 实盘下单接口
	•	封装订单管理、止盈止损、仓位管理模块
	•	后期可扩展多账户、多平台（Multi-Broker）

⸻

🖥️ 阶段六：UI 可视化模块

✅ 目标
	•	提供简洁 Web UI 展示：
	•	当前策略运行状态、实盘资产、持仓、交易日志
	•	回测可视化图表、策略切换等
	•	推荐使用：
	•	✅ Streamlit（简单好用）
	•	✅ Plotly + FastAPI（更专业）
	•	✅ Gradio（模型交互友好）

⸻

🔁 阶段七：多币种与多策略管理

✅ 目标
	•	实现调度器 TraderEngine，统一管理多个币种和策略实例
	•	支持配置：

strategies:
  - name: btc_mean_reversion
    symbol: BTCUSDT
    interval: 1m
    class: MeanReversion
  - name: eth_momentum
    symbol: ETHUSDT
    interval: 5m
    class: MomentumStrategy



⸻

📌 后续进阶功能建议
	•	✅ 引入风控模块（最大回撤、仓位限制）
	•	✅ 引入资金管理策略（Kelly、等额、固定比例）
	•	✅ 生成交易报告（PDF 或 Web 报告）
	•	✅ 接入 Telegram/Lark 通知
	•	✅ 引入 AI 策略模块（如基于 LLM 的信号理解）

⸻

✅ 下一步任务建议

我们可以 从第一个模块开始搭建，我建议你先做以下几件事：

📦 步骤 1：初始化项目
	•	创建项目结构并初始化 git 仓库
	•	写好 Strategy 抽象类和项目配置文件

你准备好后告诉我，我会帮你一步步写出完整代码。你也可以说「现在开始做第一步」，我就会帮你生成代码框架，逐步推进。是否开始？