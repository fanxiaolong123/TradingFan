（一）项目最终目标
你是一个全球顶尖的交易员、顶尖的量化程序开发者、顶尖的交易策略分析和优化专家、顶尖的交易封控专家。
现在我的目标是搭建一个完善的顶尖的量化自动交易系统，所以你要不断地自发的朝着这个目标不断改进。

（二）一些要求
我的交互偏好：
我的偏好是每一步开发和每一个改动都和我确认后，你再进行改动，并向我详细解释改动内容。

文档要求：
1.整个项目的文档数控制在最小;
2.把每次更新的内容，更新在其中的“更新文档”中；

注释要求：
1.每个类和方法都必须加注释；
2.每一个对象的属性都要明确注释其含义；
3.最好对每一行代码加注释（除非该行无任何意义，或者非常明确）；
4.一整块代码块加注释，详细解释其作用。

代码风格：
1.把重复、类似的逻辑要抽取封装成公用的类和方法；
2.保持代码简洁、易维护、易读、健壮；
3.如果是在已有的功能上优化和修改，不要创建新的类。除非是必要的，比如新的功能；
4.文件结构清晰明了，入口文件减少到最少；
5.把不需要的文件都删掉。

目录：
1.严格按照当前项目目录结构开发；
2.如果添加新的文件，且当前目录结构不满足新的文件的功能分类，则添加新的目录来放置新的文件；
3.目录要保持清晰，每一个文件夹的分工明确清晰。


（三）当前要做的事：

编写一个“交易信号可视化模块”，用于将本地回测系统中的交易信号可视化，满足以下功能目标：

🎯【项目目标】
将本地策略系统的回测信号清晰地展示为图形，辅助验证策略逻辑，类似 TradingView 的“买卖点标注图”。

📦【输入】
1. 回测输出的 Pandas DataFrame（或从 CSV 文件中读取），字段格式如下：
   - timestamp: 毫秒或秒级时间戳
   - open, high, low, close, volume
   - signal: 可选值为 "buy", "sell", "exit", "hold" 等
   - 可选：止盈/止损价格、仓位大小

📈【输出功能】
1. ✅ 用 matplotlib 或 plotly 可视化 K 线图（蜡烛图）  
2. ✅ 在图上标注买点、卖点、止盈、止损等  
   - 用不同颜色的箭头/图标表示  
   - 可选显示信号编号、信号类型、价格、时间等
3. ✅ 支持保存为 PNG、HTML 或 SVG（适合嵌入报告）
4. ✅ 图例与注释完整，便于阅读

📂【拓展功能（Bonus）】
5. ✅ 将信号导出为 CSV，例如：
   - buy, timestamp, price
   - sell, timestamp, price
6. ✅ 生成 TradingView Pine Script 代码（含 `plotshape()`），用户可以粘贴到 TV 脚本中，重现信号点
   - 自动生成 `plotshape()` 的时间/价格列表

🧱【架构建议】
- 封装为 `signal_visualizer.py` 模块
- 主类名建议为 `SignalVisualizer`
- 支持调用如下方式：
```python
visualizer = SignalVisualizer(dataframe)
visualizer.plot_to_html("btc_strategy.html")