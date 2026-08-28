from __future__ import annotations

import copy
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
SKILL_SCRIPTS = Path(r"C:\Users\admin\.codex\skills\math-modeling\tools\docx\scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))
import paper_format as pf


def text_of(element) -> str:
    return "".join(element.itertext()).strip()


def append_report_body(target, source_path: Path, start_prefix: str, stop_prefix: str | None = None) -> None:
    source = Document(source_path)
    started = False
    for child in source.element.body.iterchildren():
        if child.tag == qn("w:sectPr"):
            continue
        text = text_of(child)
        if not started:
            if child.tag == qn("w:p") and text.startswith(start_prefix):
                started = True
            else:
                continue
        if stop_prefix and child.tag == qn("w:p") and text.startswith(stop_prefix):
            break
        # Source images have package-local relationship IDs.  Formal figures
        # are inserted again below from authoritative PNG files.
        if child.xpath(".//w:drawing") or child.xpath(".//w:pict"):
            continue
        target.element.body.insert(-1, copy.deepcopy(child))


def add_figure(doc, number: int, title: str, relative: str) -> None:
    pf.image(doc, ROOT / relative, width_cm=14.6)
    pf.figure_caption(doc, f"图{number}  {title}")


def main() -> None:
    doc = pf.new_document(contest="cumcm")
    pf.title(doc, "基于删失学习、稳健分组与嵌套验证的NIPT检测时点及异常筛查研究")
    pf.abstract_title(doc)
    for paragraph in [
        "本文针对NIPT检测中男胎Y染色体浓度达标时间、BMI分组推荐以及女胎染色体异常筛查问题，构建患者级、可复现且避免重复检测泄漏的建模流程。原始数据中男胎1082条记录对应267名孕妇，女胎605条记录对应147名孕妇；同一孕妇多次检测、达标时间存在左删失、区间删失和右删失，且女胎标签高度不平衡。本文首先统一孕周、患者标识与缺失口径，并在全部训练与验证环节按孕妇分组。",
        "对Y染色体浓度，采用带孕妇随机截距的logit混合效应模型刻画孕周、BMI及其交互作用。扩展模型AIC由836.330降至817.693，似然比检验p=2.97×10^-5；孕周效应显著为正，BMI效应显著为负，交互项未显示稳定证据。对首次达到4%的时间，以区间删失AFT思想生成患者级达标概率，在固定患者折中完成分组与时点决策。",
        "在题面三级风险口径下，BMI切点34.357对应18周和22周推荐，预测达标概率为0.9555。加入多因素后，严格嵌套流程审计得到平均测试达标概率0.9556、保守下限安全3/4折；最终冻结的可执行政策采用整数切点35，BMI<35的227人推荐18周，BMI≥35的40人推荐22周，固定政策预测达标概率0.9533，外层回放安全4/4，5000次固定候选Bootstrap的均值与保守下限安全率均为100%。两种审计口径分开报告。",
        "女胎部分采用患者级外层4折、内层3折的多标签XGBoost与概率校准。任一附件筛查异常标签的记录级PR-AUC为0.4622、灵敏度0.8337；进一步形成患者级三分流：40人直接阳性、71人建议复测、36人阴性，直接阳性PPV为0.550、特异度为0.825，直接阳性与复测合并覆盖灵敏度为0.909。鉴于附件标签不是临床确诊，模型仅用于研究性筛查与复测优先级。"
    ]:
        pf.body(doc, paragraph)
    pf.keywords(doc, "NIPT；区间删失；混合效应模型；Optimal Binning；嵌套交叉验证；多标签筛查")

    pf.heading1(doc, "1  问题背景、数据与总体路线")
    pf.body(doc, "NIPT检测时点既要尽可能提前，又必须保证胎儿游离DNA信号达到可靠检测水平。本文把问题拆为浓度影响机制、删失时间重建与分组决策、多因素稳健部署、女胎异常筛查四个相互衔接的层次。所有结果均以患者为统计单位；重复记录只用于纵向信息，不允许跨训练折与测试折。")
    pf.body(doc, "原始孕周字符串统一换算为十进制周，例如12w+3换算为12+3/7。对同一孕妇的多次记录保留时间顺序，并将首次观测已达标、相邻检测间跨越阈值、末次仍未达标分别编码为左删失、区间删失和右删失。缺失字段按既有冻结数据合同处理，模型输入、患者折号和结果文件均由清单记录。")
    pf.equation(doc, r"t=\mathrm{week}+\frac{\mathrm{day}}{7}")
    pf.equation(doc, r"P(Y\geq0.04;x,t)=1-\hat S(t;x)")

    pf.page_break(doc)
    append_report_body(doc, ROOT / "output/reports/legacy/C题问题1-3统合报告_v1.docx", "1 ", "参考文献")
    pf.page_break(doc)
    append_report_body(doc, ROOT / "output/reports/legacy/C题问题4最终报告_v5.docx", "1 ", "参考文献")

    pf.page_break(doc)
    pf.heading1(doc, "8  统一结果图与证据核验")
    pf.body(doc, "图1—图3分别给出前三个子问题的原始证据、策略稳定性和最终部署政策；图4—图9展示女胎标签分布、患者折平衡、模型选择、阈值权衡、PR曲线与校准表现。所有图均由当前冻结结果脚本生成，不采用归档探索图。")
    figures = [
        (1, "Y染色体浓度、孕周与BMI的原始关系", "output/figures/q123/raw_q1_y_week_bmi.png"),
        (2, "BMI分组政策在重复验证中的稳定性", "output/figures/q123/process_q2_policy_stability.png"),
        (3, "多因素扩展后的固定BMI政策与推荐时点", "output/figures/q123/result_q3_final_policy.png"),
        (4, "女胎T13、T18、T21标签分布", "output/figures/raw_q4_label_distribution.png"),
        (5, "患者级外层折的样本与阳性平衡", "output/figures/process_q4_fold_balance.png"),
        (6, "女胎候选模型的嵌套验证选择", "output/figures/process_q4_model_selection.png"),
        (7, "筛查阈值下灵敏度、精确率与复测代价权衡", "output/figures/process_q4_threshold_tradeoff.png"),
        (8, "各异常标签及任一异常的PR曲线", "output/figures/result_q4_pr_curves.png"),
        (9, "患者级任一异常概率的校准表现", "output/figures/result_q4_calibration.png"),
    ]
    for number, title, path in figures:
        add_figure(doc, number, title, path)
        pf.body(doc, f"如图{number}所示，该证据用于对应章节的结果判断；图中性能均来自患者隔离的外层测试或冻结策略回放，训练折内部结果不冒充独立测试结果。")

    pf.heading1(doc, "9  模型评价、适用边界与结论")
    pf.body(doc, "本文的主要优点是把重复检测、删失、分组优化和分类验证放在统一的患者级框架中；用严格外层折隔离政策选择与性能评价，并以Bootstrap、最低保守下限和测量误差模拟共同约束推荐。整数切点35不是医学诊断界值，而是开发后对安全平台、可解释性和部署稳定性的折中。")
    pf.body(doc, "主要局限包括：样本来自单一附件，缺少独立医院与时间外验证；第二问的早期AFT和Optimal Binning生成阶段在当前精简工程中作为冻结实验保留，现目录可复核策略及敏感性但不宣称一键重训该历史阶段；女胎AE列均为‘是’，只能验证附件AB筛查异常标签，不能预测真实胎儿疾病。")
    pf.body(doc, "最终建议为：题面三级风险口径使用34.357切点下的18/22周方案；若要求更便于部署且经多因素稳健审计的方案，使用BMI<35第18周、BMI≥35第22周。女胎模型用于复测优先级和研究性筛查，T21信号降级进入复测，不直接作稳定阳性诊断。")

    pf.heading1(doc, "参考文献")
    refs = [
        "[1] Chen E Z, et al. Noninvasive prenatal diagnosis of fetal trisomy 18 and trisomy 13 by maternal plasma DNA sequencing. PLoS ONE, 2011, 6(7): e21791.",
        "[2] Buckley J, James I. Linear regression with censored data. Biometrika, 1979, 66(3): 429-436.",
        "[3] Chen T, Guestrin C. XGBoost: A scalable tree boosting system. Proceedings of KDD, 2016: 785-794.",
        "[4] Steyerberg E W. Clinical Prediction Models. 2nd ed. Springer, 2019.",
        "[5] Efron B, Tibshirani R J. An Introduction to the Bootstrap. Chapman & Hall/CRC, 1993.",
    ]
    for ref in refs:
        pf.body(doc, ref)
    pf.body(doc, "说明：本文为基于现有代码与冻结实验结果生成的建模论文草稿，正式使用前须由参赛团队核对当届官方模板、引用元数据与全部临床表述。")

    out = ROOT / "docs" / "C题完整论文_v1.docx"
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(out)


if __name__ == "__main__":
    main()
