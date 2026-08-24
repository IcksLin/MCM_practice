# Question 1

问题1使用孕妇随机截距线性混合模型分析男胎Y染色体浓度。响应变量为Y浓度的logit；基础模型包含标准化孕周、孕周二次项、BMI及孕周×BMI交互，扩展模型加入年龄、身高、IVF、唯一比对读段数、比对比例和GC含量。

运行：

```powershell
& 'E:\anaconda\python.exe' 'question1\solve_q1.py'
```

结果写入`results/q1/`。同一孕妇的多次检测通过随机截距处理，显著性结论以系数检验和基础/扩展模型似然比检验共同给出。
