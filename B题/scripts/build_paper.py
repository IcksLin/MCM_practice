from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "docx_tools"))
import paper_format as pf


def paragraphs(doc, texts: list[str]) -> None:
    for text in texts:
        pf.body(doc, text)


def captioned_table(doc, number: int, title: str, rows: list[list[str]]) -> None:
    pf.figure_caption(doc, f"表{number}  {title}")
    pf.three_line_table(doc, rows)


def captioned_figure(doc, number: int, title: str, relative: str) -> None:
    pf.image(doc, ROOT / relative, width_cm=14.8)
    pf.figure_caption(doc, f"图{number}  {title}")


def main() -> None:
    robust = pd.read_csv(ROOT / "results" / "robustness_summary.csv")
    final = pd.read_csv(ROOT / "results" / "final_recommendations.csv")
    sens = pd.read_csv(ROOT / "results" / "sensitivity_details.csv")
    doc = pf.new_document(contest="cumcm")
    pf.title(doc, "基于色散修正干涉相位与连续留块验证的外延层厚度测量")
    pf.abstract_title(doc)
    paragraphs(doc, [
        "本文研究红外反射光谱条件下碳化硅外延层厚度的确定问题。原始附件包含碳化硅与硅材料在10°、15°入射角下的四条反射谱，每条7469个波数点。数据具有显著的低频背景、条纹包络、频谱点强自相关以及个别反射率超过100%的仪器相对标定现象；若直接把全部谱点作为独立样本，既会低估不确定性，也会让高自由度模型获得虚假的拟合优势。",
        "针对单次反射，本文由Snell定律推导色散与角度共同修正的光学坐标，采用Savitzky–Golay趋势分离、FFT周期初值、峰谷双通道定位和三点抛物线亚网格校正，将厚度估计转化为相邻同类极值在光学坐标中的间距估计。Si折射率使用Edwards–Ochoa色散式，4H-SiC使用Lorentz振子介电函数；主频段避开SiC强声子吸收区。",
        "计算得到SiC在10°和15°下的厚度分别为7.482232 μm和7.480400 μm，联合推荐值为7.481316 μm，两角度差仅0.001832 μm。对相邻阶局部厚度序列进行400次、块长2的移动区块Bootstrap，联合95%分位区间为[7.397793,7.584614] μm；角度、折射率、频段和越界反射率截断敏感性中的最大绝对变化为0.037599 μm。硅样品联合厚度为3.475480 μm，联合区间为[3.386298,3.628350] μm。",
        "针对多光束干涉，本文建立Airy型高阶反射模型，并以5折连续留块、验证块两侧各隔离96个谱点的方式比较单谐波与三谐波模型。三谐波模型在SiC和Si上的验证NRMSE分别恶化5.6755%和4.6810%，表明全样本BIC改善主要来自附加自由度和残差相关结构，不能证明厚度修正具有外推价值。因此最终保留谐波成分作为可能存在多光束影响的迹象，但不采用多谐波厚度修正。方法通过双角度一致性、相邻阶Bootstrap、参数敏感性与严格留块验证形成完整证据链。"
    ])
    pf.keywords(doc, "红外反射光谱；外延层厚度；色散修正；相邻阶Bootstrap；连续留块验证；多光束干涉")

    pf.heading1(doc, "1  问题重述与分析")
    pf.heading2(doc, "1.1  任务重述")
    paragraphs(doc, [
        "赛题要求根据外延层上下界面反射光的干涉条纹建立厚度模型，使用给定光谱计算碳化硅外延层厚度，并分析多光束干涉的发生条件及其对厚度计算的影响。附件1、2为同一类碳化硅样品在两个入射角下的反射谱，附件3、4提供硅样品用于多光束影响分析。核心输出不是一条拟合曲线，而是可解释的厚度值、跨角度一致性、不确定性范围以及是否需要高阶反射修正的判定。",
        "该问题的主要难点有三点。第一，折射率随波数变化，常折射率条纹间距公式会积累系统偏差；第二，相邻谱点并不独立，普通随机交叉验证和点级Bootstrap会产生信息泄漏；第三，多谐波模型在训练数据上必然比单谐波拟合更好，必须用保留连续频段的外推误差判断新增结构是否真正有用。"
    ])
    pf.heading2(doc, "1.2  数据检查与预处理原则")
    paragraphs(doc, [
        "四个工作簿均为7469行2列，波数从399.6747 cm⁻¹单调增加到4000.122 cm⁻¹，无波数重复与缺失。附件2有262个反射率观测超过100%，占3.5078%，最大值为102.7394%。这类小幅越界更符合仪器相对标定与噪声，而非物理反射率定义被改变，因此主分析保留原值，再把截断到[0,100%]作为敏感性方案，避免先验裁剪移动条纹极值。",
        "SiC主分析使用1500—4000 cm⁻¹，以避开约800—1000 cm⁻¹附近的强声子响应；Si使用1000—4000 cm⁻¹。背景由宽窗Savitzky–Golay平滑获得[3]，极值检测只作用于标准化残差，厚度公式仍使用原始波数位置。如图1所示，SiC原谱在选带内保留清晰条纹；如图2所示，四条谱去趋势后均呈周期结构；如图3所示，反射率越界集中在附件2。如表1所示，四个附件按材料与任务采用两个主频段。"
    ])
    captioned_figure(doc, 1, "SiC两入射角原始反射谱与有效频段", "figures/raw/raw_q1_sic_spectra.png")
    captioned_figure(doc, 2, "四条光谱去趋势后的标准化干涉条纹", "figures/raw/raw_q2_detrended_fringes.png")
    captioned_figure(doc, 3, "四个附件的反射率越界诊断", "figures/raw/raw_q3_measurement_diagnostics.png")
    captioned_table(doc, 1, "数据与主分析频段", [
        ["附件", "材料", "角度", "有效频段/cm⁻¹", "用途"],
        ["1", "4H-SiC", "10°", "1500—4000", "厚度估计"],
        ["2", "4H-SiC", "15°", "1500—4000", "厚度与角度验证"],
        ["3", "Si", "10°", "1000—4000", "多光束诊断"],
        ["4", "Si", "15°", "1000—4000", "多光束与角度验证"],
    ])

    pf.heading1(doc, "2  模型假设与符号说明")
    pf.heading2(doc, "2.1  基本假设")
    paragraphs(doc, [
        "假设测量区域内外延层厚度近似均匀，上下界面局部平行；空气折射率取1，入射角为题面给定的空气侧角度；主分析频段内材料折射率可由文献色散模型描述；背景与包络相对干涉相位缓慢变化；同一材料的10°与15°谱对应同类晶圆厚度，可用于外部一致性检验。多光束分析阶段允许高阶界面反射，但不在缺少掺杂浓度和消光系数实测值时同时自由拟合厚度、载流子浓度与全部界面参数。",
        "这些假设把可辨识的基本相位周期与难辨识的幅值包络分开。厚度由相邻干涉级次决定，而背景、多光束尖峰和仪器增益主要影响幅值。若样品存在明显楔角、强散射或频段内强吸收，条纹会被展宽甚至消失，此时本方法的极值序列不再稳定，模型适用性应重新评估。"
    ])
    pf.heading2(doc, "2.2  主要符号")
    captioned_table(doc, 2, "主要符号及单位", [
        ["符号", "含义", "单位"], ["σ", "真空波数", "cm⁻¹"], ["λ", "真空波长", "μm"],
        ["d", "外延层厚度", "μm"], ["θ₀", "空气侧入射角", "°"], ["θt", "层内折射角", "°"],
        ["n(σ)", "材料相折射率", "1"], ["q", "色散角度修正光学坐标", "cm⁻¹"], ["R", "反射率", "%"],
    ])
    paragraphs(doc, ["如表2所示，本文统一使用真空波数、空气侧入射角和微米厚度单位，所有公式中的角度与长度换算均按该符号合同执行。"])

    pf.heading1(doc, "3  单次反射干涉模型")
    pf.heading2(doc, "3.1  相位差与色散光学坐标")
    paragraphs(doc, [
        "空气中的入射光在外延层上表面发生一次反射，透射部分到达外延层—衬底界面后反射并再次出射。两束光的光程差由层内传播方向决定。由Snell定律，有"
    ])
    pf.equation(doc, r"\sin\theta_t=\frac{\sin\theta_0}{n(\sigma)}")
    paragraphs(doc, ["于是基本相位差可写为"])
    pf.equation(doc, r"\Phi(\sigma)=4\pi d\sigma\sqrt{n^2(\sigma)-\sin^2\theta_0}+\phi_r")
    paragraphs(doc, [
        "其中φr为界面反射相移。在窄局部范围内，背景B与包络A相对相位缓慢变化，单次反射近似写成下式。该表达式只把周期结构解释为厚度信息，并不强迫整个反射率包络满足常幅余弦。"
    ])
    pf.equation(doc, r"R(\sigma)=B(\sigma)+A(\sigma)\cos\Phi(\sigma)+\varepsilon(\sigma)")
    paragraphs(doc, ["定义色散与角度共同修正的光学坐标"])
    pf.equation(doc, r"q(\sigma,\theta_0)=2\sigma\sqrt{n^2(\sigma)-\sin^2\theta_0}")
    paragraphs(doc, ["相邻同类极值的相位相差2π，因而局部厚度估计为"])
    pf.equation(doc, r"\hat d_k=\frac{10^4}{q(\sigma_{k+1})-q(\sigma_k)}")
    paragraphs(doc, [
        "式中10⁴完成cm到μm的换算。若折射率近似常数，该式退化为熟知的d=[2Δσ√(n²−sin²θ₀)]⁻¹。本文保留n随波数的变化，图4展示四条谱对应的q坐标；同一材料不同角度曲线接近但不完全重合，这正是角度修正的来源。"
    ])
    captioned_figure(doc, 4, "色散与入射角修正后的光学坐标", "figures/process/process_q1_optical_coordinate.png")

    pf.heading2(doc, "3.2  折射率模型")
    paragraphs(doc, [
        "Si使用Edwards和Ochoa在红外波段给出的温度附近色散公式[1]，其中λ以μm计。该模型避免把厚度与任意折射率同时拟合造成尺度不可辨识。"
    ])
    pf.equation(doc, r"n=3.41983+0.159906L-0.123109L^2+1.26878\times10^{-6}\lambda^2-1.95104\times10^{-9}\lambda^4")
    pf.equation(doc, r"L=\frac{1}{\lambda^2-0.028}")
    paragraphs(doc, ["4H-SiC采用低掺杂Lorentz振子介电函数[2]，取介电函数实部的正平方根作为透明频段相折射率。"])
    pf.equation(doc, r"\varepsilon(\sigma)=\varepsilon_\infty\left(1+\frac{\sigma_L^2-\sigma_T^2}{\sigma_T^2-\sigma^2-i\Gamma\sigma}\right)")

    pf.heading1(doc, "4  厚度求解算法与碳化硅结果")
    pf.heading2(doc, "4.1  峰谷双通道与亚网格定位")
    paragraphs(doc, [
        "算法首先用FFT只估计条纹基本周期，并据此设置极值最小距离；随后分别在残差及其相反数上寻找峰和谷。为减小离散采样网格造成的极值量化误差，在每个极值的三个相邻点上拟合抛物线，其顶点作为亚网格波数位置。峰序列与谷序列分别计算局部厚度，只保留位于FFT初值60%—140%的间距，最后取中位数并平均峰谷两通道。",
        "这一设计比直接对全谱做高阶非线性波形拟合更稳健：厚度只依赖基本相位周期，背景与包络不会通过大量自由参数拖动周期；峰谷分开还可揭示峰形不对称和漏峰。附件4的峰谷差较大，正提示Si高波数端弱条纹的极值定位不确定性高于SiC。"
    ])
    pf.heading2(doc, "4.2  点估计与双角度验证")
    rows = [["数据", "材料", "角度", "厚度/μm", "Bootstrap 95%区间/μm", "H3相对H1 CV变化"]]
    for _, row in robust.iterrows():
        rows.append([row["record"], row["material"], f'{row["angle_deg"]:.0f}°', f'{row["thickness_um"]:.6f}', f'[{row["bootstrap_lo"]:.6f},{row["bootstrap_hi"]:.6f}]', f'{row["cv_gain_pct"]:.2f}%'])
    captioned_table(doc, 3, "四条光谱的厚度估计与验证指标", rows)
    paragraphs(doc, [
        "如表3所示，SiC两角度估计分别为7.482232 μm与7.480400 μm，绝对差仅0.001832 μm，相对差约0.0245%。同一晶圆在改变入射角后仍得到近乎相同的厚度，是对Snell角度修正、色散模型和极值定位共同有效的外部检验。Si两角度估计差为0.001585 μm，也显示基本相位周期具有良好一致性。",
        "相邻阶Bootstrap遵循移动区块重采样思想[4]，但不重新生成原始光谱，而是对已经由同类相邻极值构造的局部厚度序列按2个级次移动分块重采样。这样保留局部级次相关性，同时避免错误地把7469个高度相关谱点当作独立重复。如图5所示，400次重复围绕基准厚度分布；该区间反映有限条纹级次的抽样波动，不能替代具有标准样片、仪器重复测量和折射率标定的完整测量学不确定度。如图7所示，四个附件的区间宽度与条纹质量相符。"
    ])
    captioned_figure(doc, 5, "相邻阶局部厚度序列的移动区块Bootstrap", "figures/process/process_q2_block_bootstrap.png")
    captioned_figure(doc, 7, "四个附件厚度点估计及相邻阶序列Bootstrap区间", "figures/result/result_q1_thickness_intervals.png")

    pf.heading1(doc, "5  多光束干涉模型与模型选择")
    pf.heading2(doc, "5.1  多光束成立的物理条件")
    paragraphs(doc, [
        "外延层内的高阶反射只有在若干条件同时满足时才可观测：光源相干长度应大于相邻级次的光程差；上下界面有效反射率足够高且吸收不强；两界面局部平行、粗糙度小；各级出射光在探测器上空间重合；光谱仪分辨率足以解析变尖条纹。任一条件不满足，高阶光束都会因衰减或相位平均而难以辨认。",
        "令r01、r12为两个界面的复振幅反射系数，高阶反射构成几何级数，总反射振幅可写成Airy型表达式[5]："
    ])
    pf.equation(doc, r"r_{eff}=\frac{r_{01}+r_{12}e^{i\Phi}}{1+r_{01}r_{12}e^{i\Phi}}")
    pf.equation(doc, r"R=\left|r_{eff}\right|^2")
    paragraphs(doc, [
        "多光束主要使峰形变尖并产生二次、三次谐波，但理想平行层的基本相位周期不变。因而谐波存在不等于必须改变厚度；只有高阶模型在未参与拟合的连续频段上降低误差，才说明修正具备可推广价值。"
    ])
    pf.heading2(doc, "5.2  连续留块验证")
    paragraphs(doc, [
        "本文把每条谱按波数顺序分成5个连续验证块。每次拟合时，验证块及其两侧各96个谱点全部从训练集排除，以削弱相邻频谱点的相关泄漏。SiC厚度搜索边界固定为5至10 μm，Si固定为2至6 μm，边界由材料与条纹尺度的物理范围预先给定，而非由包含验证块的全谱点估计。归一化尺度只由当折训练集计算。"
    ])
    pf.equation(doc, r"NRMSE=\frac{\sqrt{\frac{1}{n_v}\sum_{i\in V}(R_i-\hat R_i)^2}}{s_{train}}")
    paragraphs(doc, [
        "如图6所示，四条谱的三谐波验证误差均高于单谐波：SiC平均恶化5.6755%，Si平均恶化4.6810%。虽然全样本三谐波拟合的残差平方和与形式BIC更低，但频谱残差强相关且高阶模型自由度更多，因此训练拟合改善不能直接作为物理模型成立的证据。最终判定为：光谱中存在谐波迹象，尤其Si的二次谐波相对较强；然而现有数据不支持用三谐波拟合修正厚度，报告厚度仍取色散修正基本周期估计。"
    ])
    captioned_figure(doc, 6, "单谐波与三谐波模型的5折连续留块验证误差", "figures/process/process_q3_blocked_cv.png")

    pf.heading1(doc, "6  稳健性、误差与最终建议")
    pf.heading2(doc, "6.1  参数敏感性")
    paragraphs(doc, [
        "对每条谱分别改变入射角±0.2°、折射率尺度±0.5%、拟合频段位置±8%，并比较是否把反射率截断至[0,100%]。定义单项相对变化为"
    ])
    pf.equation(doc, r"S_j=100\times\left(\frac{\hat d_j}{\hat d_0}-1\right)")
    paragraphs(doc, [
        "结果如图8所示。反射率截断几乎不改变厚度，说明附件2少量超过100%的相对读数主要影响幅值而非极值波数；折射率尺度±0.5%大致引起厚度∓0.5%的变化，体现厚度与折射率尺度的系统耦合；频段改变对Si的影响大于SiC，与Si弱条纹和附件4较大峰谷差一致。所有敏感性情景中的最大绝对偏移为SiC 0.037599 μm、Si 0.058301 μm。"
    ])
    captioned_figure(doc, 8, "角度、折射率、频段与反射率截断敏感性", "figures/result/result_q2_sensitivity_envelope.png")
    pf.page_break(doc)
    pf.heading2(doc, "6.2  最终厚度与解释边界")
    final_rows = [["材料", "推荐厚度/μm", "联合Bootstrap 95%区间/μm", "角度差/μm", "三谐波CV变化", "决策"]]
    for _, row in final.iterrows():
        final_rows.append([row["material"], f'{row["recommended_thickness_um"]:.6f}', f'[{row["bootstrap95_lo_um"]:.6f},{row["bootstrap95_hi_um"]:.6f}]', f'{row["angle_difference_um"]:.6f}', f'{row["three_harmonic_cv_gain_pct"]:.2f}%', "不作多谐波修正"])
    captioned_table(doc, 4, "最终厚度建议、区间和模型选择", final_rows)
    paragraphs(doc, [
        "如表4所示，综合点估计、双角度一致性、相邻阶序列Bootstrap与参数敏感性，SiC最终推荐厚度为7.481316 μm，有限级次抽样波动区间为[7.397793,7.584614] μm；若只需符合数据分辨率的工程报告，可写为7.48 μm，并把区间与最大敏感性偏移同时给出，而不把小数位数误当成测量精度。Si最终推荐厚度为3.475480 μm，对应联合区间[3.386298,3.628350] μm。",
        "如图9所示，最终数值与两类验证信息处于同一结论框架中：两个角度的点估计高度一致，说明基本周期可靠；三谐波留块验证为负增益，说明无需因训练拟合更好而改变厚度。该结论比旧的‘发现谐波便修正厚度’更保守，也更符合预测模型应以未见频段表现决定复杂度的原则。"
    ])
    captioned_figure(doc, 9, "最终厚度建议及跨角度、跨模型一致性", "figures/result/result_q3_final_recommendation.png")

    pf.heading1(doc, "7  模型评价与推广")
    pf.heading2(doc, "7.1  优点")
    paragraphs(doc, [
        "第一，模型直接从光程差推导，并在光学坐标中同时修正折射率色散与入射角，保持了厚度参数的物理可解释性。第二，FFT只用于初始化，最终厚度来自峰谷双通道和亚网格极值位置，降低背景、包络及离散网格的影响。第三，双角度独立估计构成外部一致性验证；相邻阶序列Bootstrap与参数敏感性分别描述有限条纹抽样波动和系统参数变化，避免把两者混为一个虚假的高精度置信区间。第四，多光束模型选择采用带隔离带的连续留块验证，主动抑制频谱自相关导致的信息泄漏。"
    ])
    pf.heading2(doc, "7.2  局限与改进方向")
    paragraphs(doc, [
        "折射率模型来自文献而非本批晶圆的同步椭偏测量。若4H-SiC掺杂浓度较高，自由载流子Drude项可能改变介电函数；附件未提供掺杂浓度、温度、消光系数和衬底参数，因此不能可靠地同时辨识厚度与载流子参数。建议后续加入标准厚度样片、重复测量和独立折射率标定，建立完整测量不确定度预算。",
        "相邻阶Bootstrap的有效样本是有限数量的局部间距，不是7469个谱点，因此区间会受漏峰、伪峰和包络衰减影响。对于附件4，峰谷差约0.239 μm，提示应增加低噪声重复测量或缩窄到条纹清晰区。若获得界面折射率、吸收与粗糙度参数，可进一步直接拟合带复折射率的Fresnel—Airy模型，并继续用连续留块验证确认其外推价值。",
        "本方法可推广到具有缓慢色散且条纹可解析的薄膜/外延层红外反射测厚。对于非平行层、强散射层或条纹数少于四个的情况，应改用空间分辨测量、椭偏法或具有外部校准的全物理反演，而不应机械套用相邻极值公式。"
    ])

    pf.heading1(doc, "参考文献")
    refs = [
        "[1] Edwards D F, Ochoa E. Infrared refractive index of silicon. Applied Optics, 1980, 19(24): 4130-4131. DOI: 10.1364/AO.19.004130.",
        "[2] Tang X, et al. Thickness determination of 4H-SiC epitaxial films by infrared reflectance. 2009 IEEE International Conference of Electron Devices and Solid-State Circuits, 2009: 295-297. DOI: 10.1109/EDSSC.2009.5394259.",
        "[3] Savitzky A, Golay M J E. Smoothing and differentiation of data by simplified least squares procedures. Analytical Chemistry, 1964, 36(8): 1627-1639. DOI: 10.1021/ac60214a047.",
        "[4] Efron B, Tibshirani R J. An Introduction to the Bootstrap. New York: Chapman & Hall/CRC, 1993.",
        "[5] Born M, Wolf E. Principles of Optics. 7th ed. Cambridge: Cambridge University Press, 1999.",
    ]
    for ref in refs:
        pf.body(doc, ref)
    paragraphs(doc, [
        "版式说明：本稿采用数学建模Skill内置CUMCM Word构建基线。因工程中未提供且当前未能核验2025届官方Word模板，本稿不声称已满足当届官方封面、编号页、页眉页脚和页面细则；正式提交前须以竞赛官网当届文件替换模板并复核。"
    ])
    output = ROOT / "docs" / "B题完整论文.docx"
    output.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output)


if __name__ == "__main__":
    main()
