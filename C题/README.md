# C题精简工程

本目录只保留原始数据、验证框架、按题号划分的核心算法、功能脚本，以及可用于论文写作的最终结果与证据图。探索性旧报告、烟雾测试和可再生中间产物已清理。

## 目录

```text
C题/
├─ data/raw/               # 原始附件与题面（只读输入）
├─ data/validation/        # 固定患者折号、验证协议和数据清单
├─ docs/                   # 总思路表、题目分析、术语与方法说明
├─ algorithms/q1..q4/      # 按题号管理的核心算法
├─ scripts/                # 验证、绘图、PDF构建和统一入口
└─ output/
   ├─ results/q1..q4/      # 最终结果及稳健性/敏感性证据
   ├─ figures/             # 论文证据图
   ├─ reports/             # 当前PDF与保留的正式DOCX
   ├─ latex/thoughts/      # 可再生LaTeX源码
   └─ manifests/           # 复现与文件审计清单
```

## 复现

轻量复现：

```powershell
E:\anaconda\python.exe scripts\run_all.py
```

完整复现（额外重跑问题3搜索、问题4嵌套验证和稳健性实验）：

```powershell
E:\anaconda\python.exe scripts\run_all.py --full
```

仅重新生成思路PDF：

```powershell
E:\anaconda\python.exe scripts\build_thoughts_pdf.py
```

当前推荐口径及实验数据以 `docs/C题总思路表.md` 和 `output/reports/C题总思路表.pdf` 为准。所有建议仅用于数学建模与研究性分流，不构成临床诊断或处置意见。
