# 问题1—3统一复现

运行：

```powershell
python scripts/run_all.py
```

脚本依次完成：问题1混合模型重建、问题2冻结政策校验、问题3嵌套外层审计、冻结结果哈希核验、9幅证据图重绘和复现清单更新。任一步失败均返回非零退出码。

- `plot_q123.py`：生成原始数据、建模过程和结果三类图。
- `build_manifest.py`：记录活动代码、结果和图件的SHA-256。
- `build_q123_report.py`：生成不覆盖旧报告的统合报告草稿。
- `equations_q123.json`：Word原生公式占位符映射。
- `render_q123_with_word.ps1`、`render_pdf_pages.py`：仅用于最终版式质检。

默认入口不重新执行历史大规模搜索；问题2/3采用已冻结且经哈希核验的候选结果，以避免清理工程后产生结论漂移。
