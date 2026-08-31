# optlab — ETF 期权日频模拟交易工作台

个人研究用的境内 ETF 期权模拟交易系统：日频数据采集 → 回测引擎 → 模拟盘 → 可视化工作台。纯模拟，不含实盘下单。

```
数据采集(akshare) → Parquet/SQLite → 撮合引擎(Broker) → 模拟盘(Paper) → Web 工作台
```

## 功能

- **五品种行情**：沪深300 / 上证50 / 中证500 / 科创50 / 创业板ETF（深市快照口径），期权链含 Delta/Gamma/Vega/Theta
- **模拟交易**：T 日挂单 → 人工确认 → T+1 撮合，交易所保证金公式、涨跌停、到期行权（现金结算简化）
- **回测**：卖出宽跨式等策略回测 + 希腊字母损益归因（逐腿分解）+ 参数平原（36 组网格）
- **工作台**：单文件 HTML，T 型报价（精简/完整两档）、四方向下单面板、持仓管理、压力矩阵、POP、IV 曲面
- **教学**：期权基础 / 希腊字母 / 策略图鉴 / 术语表 + 交互式盈亏实验台
- **81 项测试**：定价、撮合、组合、策略、模拟盘、API、集成全覆盖

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 采集历史数据（首次约 15 分钟，覆盖 5 个沪市品种 + 深市快照）
python -m optlab.scripts.collect_risk_history 240
python -m optlab.scripts.collect_contract_history all

# 3. 启动工作台（浏览器自动打开 http://127.0.0.1:8300）
python -m optlab.server
```

Windows 用户可双击 `启动工作台.bat`（服务器）和 `每日更新.bat`（收盘后 15:30 采集当日数据并刷新页面）。

## 目录结构

```
optlab/
├── core/        定价(BS/CRR/IV)、合约规则(保证金/涨跌停/到期日)、技术指标
├── data/        数据源适配、八道校验闸门、SQLite 持久化
├── engine/      撮合(Broker)、组合(Portfolio)、回测(BacktestRunner)、模拟盘(Paper)
├── strategy/    策略 DSL、12 个模板、推荐(Advisor)、信号(Signals)、盈亏结构
├── scripts/     数据采集、回测、参数平原、工作台生成
├── tests/       9 套测试（81 项）
└── server.py    本地服务器（标准库 HTTP，零第三方 Web 框架）
```

## 常用命令

| 命令 | 说明 |
|---|---|
| `python -m optlab.server` | 启动交易工作台（含页面 + API） |
| `python -m optlab.scripts.collect_daily` | 采集当日数据（交易日 15:30 后） |
| `python -m optlab.scripts.collect_contract_history all` | 补采五品种逐合约历史日线 |
| `python -m optlab.scripts.run_strangle_backtest` | 卖出宽跨式回测（约 1 年） |
| `python -m optlab.scripts.sensitivity_analysis` | 参数平原 + 成本压力 + 未来函数审计 |
| `python -m optlab.tests.test_core` ... | 运行测试（9 套） |

## 设计原则

1. **纯模拟**：不接券商、不下实盘单；撮合与回测共用同一 Broker 内核（模拟盘收益可被回测证伪）
2. **宁可低估收益**：价差模型用真实盘口校准（tick/价格主导），单笔 ≤ 当日成交量 2%
3. **T+1 纪律**：当日决策次日成交，杜绝"用收盘价给自己成交"的前视偏差
4. **数据诚实**：缺失就是缺失（NaN），不造 0 兜底；结算价以收盘价近似并在界面标注口径

## 已知局限

- 结算价以收盘价近似（交易所结算价无免费接口），保证金/涨跌停有约 1% 偏差
- 深市（159915）无逐合约历史日线，撮合走每日快照（前结算价），积累中
- 到期行权按现金结算简化（实物交割的份额交收/T+1 未建模）
- 模拟收益系统性偏高（无盘口深度、未建模保证金临时上调）；仅作策略间相对比较

## 免责声明

本项目为个人研究与教学工具，**不构成任何投资建议**。期权交易风险极高，卖方策略可能产生无上限亏损。实盘交易需满足交易所适当性要求（三级权限 + 50 万验资等）。数据来源的授权与合规由使用者自行负责。

## License

MIT — 见 [LICENSE](LICENSE)
