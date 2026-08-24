# 数学建模工程

工程采用相对路径定位数据、算法和输出，不依赖固定盘符、用户名或 Conda
环境名。建议在项目根目录创建独立虚拟环境：

```bash
python -m venv .venv
```

激活虚拟环境后安装依赖：

```bash
python -m pip install -r requirements.txt
```

主要入口：

```bash
python analysis/b_solution.py
python C题/scripts/run_all.py
python C题/scripts/run_all.py --full
```

C题入口默认复用当前 Python。需要为 XGBoost 实验使用另一环境时，设置
`C_PROJECT_XGB_PYTHON`；通用实验解释器可用 `C_PROJECT_PYTHON` 覆盖。
LaTeX 构建默认从 `PATH` 查找 `latexmk`，也支持 `LATEXMK` 环境变量。
