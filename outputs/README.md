# 2025 数模 B、C 题试解

## 文件结构

- `B题/B题解答.md`：B 题完整推导、算法、厚度结果和可靠性分析。
- `B题/b_results.json`：B 题机器可读数值结果。
- `B题/b_summary.csv`：B 题结果摘要。
- `B题/SiC_spectral_fit.png`、`B题/Si_spectral_fit.png`：频谱与模型图。
- `C题/C题解答.md`：C 题完整统计建模解答。
- `C题/c_results.json`：C 题机器可读数值结果。
- `C题/mixed_model_coefficients.csv`：混合效应模型系数。
- `C题/bmi_groups_and_timing.csv`：BMI 分组与检测时点。
- `C题/measurement_error_sensitivity.csv`：检测误差敏感性。
- `C题/multifactor_timing.csv`：多因素模型时点。
- `C题/male_longitudinal_and_timing.png`：重复测量和区间删失达标曲线。
- `C题/female_group_cv_roc.png`：女胎患者级交叉验证 ROC。

## 复现

在工作区根目录运行：

```powershell
E:\anaconda\python.exe analysis\b_solution.py
E:\anaconda\python.exe analysis\c_solution.py
```

环境使用 NumPy、pandas、SciPy、statsmodels、scikit-learn、matplotlib 和 openpyxl；没有使用大参数机器学习模型。

## 重要解释限制

- B 题相邻频谱点不独立，不能以 7469 个点直接构造普通独立样本置信区间。
- C 题独立单位是孕妇，所有验证必须按孕妇代码分组。
- 女胎 AE 列全部健康，AB 不是经出生结局确认的真实异常标签。
