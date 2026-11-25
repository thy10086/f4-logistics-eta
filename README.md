# F4 物流时效预测与承诺校准

给定订单进入仓库时可见的字段，预测从付款到签收的总小时数。项目使用 LightGBM 分位数模型、按时间截点构造的历史特征及逐物流商校准，兼顾总体预测偏差、准时率和严重超时率。

## 赛题与提交要求

F4「包裹何时到？」是 SHEIN AI 创新挑战赛的跨境包裹时效预测赛题。官方题包从赛事页面获取，评测细则和结果格式以题包《赛题》和 README 为准。 [SHEIN 赛事官网与题包入口](https://careers.shein.cn/AI-Challenge)

| 材料 | 要求 |
| --- | --- |
| 方案报告 | PDF，F4 不超过 10 页，前 2 页填写官方《作品说明单》。正文交代业务目标、建模与校准方案、基线到最终方案的迭代及证据。 |
| 预测结果 | CSV，列名依次为 `pkg_no,pred_hours`，每个测试包裹一行，预测时效为有限正数，单位小时。 |
| 可验证成果 | 提供运行说明及可复跑代码。下文给出仓库中 v3 的完整复现入口。 |
| AI 协作记录 | 按官方模板填写使用的工具与模型、分工和校验方式。 |
| 演示视频 | F4 选交，如提交应控制在 5 分钟以内。 |

作品通过官网提交入口递交。《作品说明单》和《AI 协作记录》模板可从官网的作品提交清单下载。 [SHEIN 作品提交清单和模板](https://careers.shein.cn/AI-Challenge)

## 复现正式 v3

代码来自正式评委复现包。使用 64 位 CPython 3.12，准备从主办方取得的官方 `F4.zip`，首次冷启动需预留约 10 GB 磁盘。依赖由复现入口按 `requirements.txt` 安装。

```powershell
python -X utf8 review.py full --kit C:/path/to/F4.zip
```

默认写出 `f4-full-run/result/prediction.csv`，并检查 911,415 个测试 ID 的覆盖、正有限预测及冻结校验值。使用 `--work-dir` 指定新的运行目录可重新做冷启动。

| 本地验证窗 | A | 两项罚分 |
| --- | ---: | --- |
| 9 月 4–11 日完整窗 | 83.2302 | 0 / 0 |
| 9 月 12–20 日完整窗 | 84.0657 | 0 / 0 |
| 9 月 4–11 日哈希预留 | 83.0517 | 0 / 0 |
| 9 月 12–20 日哈希预留 | 84.0141 | 0 / 0 |

这些是基于有标签历史记录的本地分数。完整窗曾用于 v3 逐物流商设置选择，哈希预留子集独立于该选择，日期相同。汇总见 `evidence/verification.json` 和 `evidence/*_official_score.txt`。

- `review.py` 管理复现、校验和评分入口。
- `solution.py`、`history_model.py` 与 `adaptive.py` 实现特征和模型。
- `carrier_policy.py` 与 `release.py` 实现分物流商策略及正式结果导出。
- [model_notes.md](model_notes.md) 记录模型细节和来源。

`review.py score` 的快评模式使用原评委复现包中的 `evidence/hold_audit_prediction.csv.gz`，请从自己持有的原包解出该文件。官方题包、原始记录和逐单预测均在本地使用。
