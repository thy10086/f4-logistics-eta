# F4 逐物流商承诺优化

生成 `prediction.csv`，列为 `pkg_no,pred_hours`，每个测试包裹一行。单位为从付款到签收的小时数。

## 复现

在 Python 3.12 环境，把官方题包中的 `data` 目录路径传给第一条数据命令。模型使用固定随机种子 1729。建议预留至少 5 GB 工作空间，训练和复核均在本机进行。

```powershell
python -m pip install -r requirements.txt
python solution.py --data "路径/到/F4/data" --cache run --output result --phase prepare
python matched.py --db run/features.duckdb --output run/original --threads 6 --rounds 240
python history_model.py --db run/features.duckdb --output run/history --phase all --threads 6 --rounds 240
python carrier_policy.py --db run/features.duckdb --cache run/original --advanced run/history --diagnostics run/check --output result
```

回测产生四份核验文件对，`stress`、`hold` 分别对应 9 月 4 日至 11 日和 9 月 12 日至 20 日，在文件名后加 `_audit` 对应各自预留的核验样本。以下示例复核较晚窗口的核验样本。

```powershell
python "路径/到/F4/tools/score.py" --truth run/check/hold_audit_truth.csv --pred run/check/hold_audit_prediction.csv
```

测试包裹的预测在 `result/prediction.csv`。程序校验列名、有限正数、唯一包裹编号及其与官方测试集的完全匹配。更换源文件时请使用新的缓存目录。

## 估计与决策

1. **区分建模和校准。** 使用包裹编号哈希留出约八分之一的标签供校准。每个回测的训练只使用切点前已付款且已签收的记录。预测模型使用订单流入时可用的字段。
2. **补充时点特征。** 按物流商、渠道、物流商与目的州组合，构造上一日之前的签收时长、订单量及 8 天和 12 天的历史签收比例。8 天、12 天的签收比例使用已完整观察过相应期限的历史订单。回测时再将查询时点限制在切点以内。历史统计只汇总模型训练侧可见的包裹事件。
3. **分别拟合准时与尾部。** 两种特征组均拟合 LightGBM P90 和 P97。逐物流商比较两种模型及平均组合，再比较尾部权重和承诺分层算法。校准时可以用 Kaplan–Meier 估计已等待到统计时点仍未签收的残差分布。
4. **按赛事约束选择承诺。** 用两个时间窗各约 75% 的记录选择每家物流商的模型权重和余量。候选规则为每家设置准时与严重超时余量，然后在满足目标的候选中比较平均承诺时间。每个时间窗约 25% 的哈希核验样本留出，独立于该逐商选择。核验样本覆盖相同的日期范围。正式测试成绩还取决于未来日期的物流变化，由主办方使用测试标签评定。

## 开源实现与参考

- [LightGBM 4.6.0](https://github.com/lightgbm-org/LightGBM)（MIT）用于分位数预测。[Conformalized Quantile Regression](https://proceedings.neurips.cc/paper_files/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html) 提供按预测残差校准分位数的思路。
- [SciPy ECDF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ecdf.html)（BSD-3-Clause）估计右删失残差分布。承诺分层的候选方案用 [SciPy MILP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html) 调用 [HiGHS](https://github.com/ERGO-Code/HiGHS)（MIT），在两项预测风险预算内比较承诺时长。
- 调研了 [XGBoost AFT](https://xgboost.readthedocs.io/en/stable/python/survival-examples/) 的删失标签设计，以及 [CatBoost MultiQuantile](https://catboost.ai/docs/en/concepts/loss-functions-regression) 的联合分位数方案。复现命令所用模型为 LightGBM，风险校准与承诺策略为 F4 的独立实现。
