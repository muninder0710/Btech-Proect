"""Build the B.Tech review deck in the DTU design of the November proposal.

Run from code/:  .venv/bin/python deck/build_review_pptx.py
Reads ../Arrhythmia_CPO-GWO_Proposal.pptx (left unchanged) for its theme, layouts and five reusable slides,
reads the chart values from results/, and writes ../Arrhythmia_Feature_Selection_Review.pptx.
"""
import copy
import csv
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

from pptx import Presentation
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE, XL_TICK_MARK
from pptx.enum.dml import MSO_THEME_COLOR
from pptx.oxml import parse_xml
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
SRC = ROOT / "Arrhythmia_CPO-GWO_Proposal.pptx"
OUT = ROOT / "Arrhythmia_Feature_Selection_Review.pptx"
RESULTS = CODE / "results"

TITLE = ("Mutual-Information-Guided Grey Wolf and Crested Porcupine Feature Selection "
         "for Arrhythmia Classification")
FOOTER = "Guided Grey Wolf and Crested Porcupine Feature Selection  ·  Delhi Technological University"
STALE = ("Lead-Aware", "Minority-Aware", "Research Proposal", "CPO–GWO", "CPO-GWO", "Leakage-Free")

NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"')
NAVY, MARK_GREY, DARK_GREY = "1F3864", "A6A6A6", "595959"

REFS = [
    ("H. A. Güvenir, B. Acar, G. Demiröz, and A. Çekin, “A supervised machine learning algorithm for arrhythmia "
     "analysis,” in ", "Proc. Computers in Cardiology", ", 1997, pp. 433–436, doi: 10.1109/CIC.1997.647926."),
    ("J. Y. Khaseeb, A. Keshk, and A. Youssef, “Improved binary grey wolf optimization approaches for feature "
     "selection optimization,” ", "Appl. Sci.", ", vol. 15, no. 2, Art. no. 489, 2025, doi: 10.3390/app15020489."),
    ("S. Dhanka and S. Maini, “A hybrid machine learning approach using particle swarm optimization for cardiac "
     "arrhythmia classification,” ", "Int. J. Cardiol.",
     ", vol. 432, Art. no. 133266, 2025, doi: 10.1016/j.ijcard.2025.133266."),
    ("H. Yu et al., “A systematic review of the crested porcupine optimizer: Variants and applications,” ",
     "Appl. Intell.", ", vol. 56, no. 14, Art. no. 422, 2026, doi: 10.1007/s10489-026-07453-1."),
    ("A.-D. Li, B. Xue, and M. Zhang, “Improved binary particle swarm optimization for feature selection with new "
     "initialization and search space reduction strategies,” ", "Appl. Soft Comput.",
     ", vol. 106, Art. no. 107302, 2021, doi: 10.1016/j.asoc.2021.107302."),
    ("S. Mirjalili, S. M. Mirjalili, and A. Lewis, “Grey wolf optimizer,” ", "Adv. Eng. Softw.",
     ", vol. 69, pp. 46–61, 2014, doi: 10.1016/j.advengsoft.2013.12.007."),
    ("M. Abdel-Basset, R. Mohamed, and M. Abouhawwash, “Crested porcupine optimizer: A new nature-inspired "
     "metaheuristic,” ", "Knowl.-Based Syst.", ", vol. 284, Art. no. 111257, 2024, doi: 10.1016/j.knosys.2023.111257."),
    ("S. Varma and R. Simon, “Bias in error estimation when using cross-validation for model selection,” ",
     "BMC Bioinformatics", ", vol. 7, Art. no. 91, 2006, doi: 10.1186/1471-2105-7-91."),
    ("A. Statnikov, C. F. Aliferis, I. Tsamardinos, D. Hardin, and S. Levy, “A comprehensive evaluation of "
     "multicategory classification methods for microarray gene expression cancer diagnosis,” ", "Bioinformatics",
     ", vol. 21, no. 5, pp. 631–643, 2005, doi: 10.1093/bioinformatics/bti033."),
    ("T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in ",
     "Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining",
     ", 2016, pp. 785–794, doi: 10.1145/2939672.2939785."),
    ("S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” ",
     "Patterns", ", vol. 4, no. 9, Art. no. 100804, 2023, doi: 10.1016/j.patter.2023.100804."),
    ("F. Li and J. Dong, “Crested porcupine optimizer-based feature selection for intrusion detection system,” in ",
     "Proc. 7th Int. Conf. Mechanical, Control and Computer Engineering (ICMCCE)",
     ", 2024, pp. 279–286, doi: 10.1109/ICMCCE63640.2024.11163434."),
    ("X. He, D. Zhang, K. Li, W. Li, and X. Shu, “Feature selection of crested porcupine optimization algorithm "
     "based on multi-strategy fusion,” in ", "Proc. 6th Int. Conf. Machine Learning and Computer Application (ICMLCA)",
     ", 2025, pp. 657–661, doi: 10.1109/ICMLCA66850.2025.11336492."),
    ("Z. Liu and M. Tao, “ECPO: An enhanced crested porcupine optimizer,” in ",
     "Proc. 4th Int. Conf. Communication Technology and Information Technology (ICCTIT)",
     ", 2024, pp. 91–97, doi: 10.1109/ICCTIT64404.2024.10928405."),
    ("S. Nogueira, K. Sechidis, and G. Brown, “On the stability of feature selection algorithms,” ",
     "J. Mach. Learn. Res.", ", vol. 18, no. 174, pp. 1–54, 2018."),
    ("P. Niu, S. Niu, N. Liu, and L. Chang, “The defect of the grey wolf optimization algorithm and its verification "
     "method,” ", "Knowl.-Based Syst.", ", vol. 171, pp. 37–43, 2019, doi: 10.1016/j.knosys.2019.01.018."),
]

NOTES = {
    "cover": "Opening line. We select a small set of ECG measurements that classifies arrhythmia, we compare against "
             "two recent base papers, and every number in this deck comes from a saved result file that one command "
             "reproduces. The title names the three parts of the method in full words: the "
             "mutual-information-guided start, the grey wolf step and the crested porcupine step.",
    "outline": "Eight parts. Parts 1 to 5 set up the problem, the two base papers, what we contribute, the method and "
               "how we test it. Part 6 gives the results against both base papers. Part 7 explains why we do not "
               "report the 95% figures. Part 8 lists what held, what failed, the limits and the conclusion. Two "
               "backup slides follow the thank-you slide.",
    "problem": "With more than half as many measurements as patients, a classifier trained on everything fits noise. "
               "Feature selection keeps a small subset. If asked why not accuracy: the majority class is 245 of 452, "
               "which is 54.2 percent, so accuracy rewards ignoring the rare classes. Macro-F1 of the always-normal "
               "model is 0.060 on twelve classes. The chart shows the classes as the dataset lists them. Our "
               "twelve-class task merges classes 7 and 8, which have 3 and 2 patients.",
    "base": "The method base paper gives us the search framework: a grey wolf step, then a second step, then keep the "
            "best. Its best variant is called IBGWO4. It tests on random 80/20 hold-out splits with no "
            "cross-validation, which it names as future work. The application base paper gives us the dataset, the "
            "two-class task and the XGBoost classifier to compare against. Each of our changes answers a "
            "future-work item that one of these papers names.",
    "contrib": "To our knowledge this is the first binary hybrid of the grey wolf optimiser and the crested porcupine "
               "optimiser for feature selection. The 2026 review of the crested porcupine optimiser [4] says binary "
               "versions barely exist. We do not claim the first hybrid of the two in general: two exist, for path "
               "planning and for tuning a neural network. The guided start itself comes from Li, Xue and Zhang 2021 "
               "[5], a particle swarm paper. What is ours is showing, under fair testing, that it carries the gain "
               "for grey wolf hybrids, across 17 optimisers and on the base paper’s own data. Be ready to say plainly "
               "that the hybrid alone is level with the base hybrid.",
    "method": "A subset is a string of 279 bits, one per attribute. Mutual information measures how much knowing an "
              "attribute reduces uncertainty about the class. In the guided start, an attribute enters a starting "
              "subset with a chance that grows with its mutual information with the class. The start comes from "
              "Li, Xue and Zhang, Applied Soft Computing 2021, with their settings lambda 0.5 and sigma 10. A "
              "starting subset then holds about 19 attributes, against about 128 for a random start. The crested "
              "porcupine step has two moves that explore and two that exploit, with the original paper's settings "
              "alpha 0.1 and Tf 0.5. Nothing was tuned on our data. The fitness is 0.7 times one minus macro-F1, "
              "plus 0.3 times one minus balanced accuracy, plus 0.01 times the share of attributes kept.",
    "protocol": "This is nested cross-validation. The outer loop measures, the inner loop chooses. Leakage means "
                "information from the test patients reaching any step that chooses something, which makes the "
                "reported score too high. The equal budget matters because the porcupine optimiser shrinks its "
                "population, so equal iterations would give it fewer evaluations. Folds overlap in training "
                "patients, so we call the p-values indicative and read the mean difference and its interval first.",
    "result1": "Bars start at zero and both tasks use one scale. The base method is IBGWO4 with its accuracy fitness "
               "and random start, exactly as the base paper describes it, run under our protocol. Paired "
               "differences: plus 0.063 on twelve classes with a 95 percent interval of 0.037 to 0.089, and plus "
               "0.036 on two classes with an interval of 0.015 to 0.059. If asked why the scores look low: twelve "
               "unbalanced classes, 452 patients, and testing that the search cannot see. A model that always "
               "answers normal scores a macro-F1 of 0.060. File: results/informed_init/informed_init_claim.csv",
    "ablation": "This is the honest core of the project. Our first claim, that swapping the particle swarm step for "
                "the porcupine step would beat the base paper, failed. We then tested one idea from the literature, "
                "once, under a stop rule: the guided start. It works, and it works for every optimiser. On twelve "
                "classes it raises macro-F1 by 0.02 to 0.13 across the five optimisers and cuts every subset to "
                "between 8 and 19 features. We also reran the comparison under the base paper's own protocol, a "
                "random 80/20 split with a population of 10 and 30 runs, and the hybrid did not beat the base "
                "method in any of four settings, so the result is not caused by our stricter testing. Files: "
                "results/informed_init/informed_init_attribution.csv and results/full/summary/KNN/",
    "optimisers": "The twelve extra optimisers were added at our request, by a rule fixed in advance: an arrhythmia "
                  "paper in our notes used the optimiser, its update rule fits the shared bit-flip rule, and it was "
                  "checked against its authors’ code or paper. They test no claim. Two classes show the same "
                  "pattern: from a random start 0.620 to 0.733, from the guided start 15 of 17 within 0.704 to "
                  "0.740. Ours is second on twelve classes, 0.417 against marine predators at 0.421, and first on "
                  "two classes at 0.740, inside the noise. From a random start the crested porcupine optimiser alone "
                  "has the highest twelve-class score, 0.393. So the porcupine is a strong optimiser. What does not "
                  "help is putting its step inside the grey wolf frame. Dragonfly and salp swarm stay lower because "
                  "their update rules overwrite the start. The 15 span 0.380 to 0.421, so say about 0.04. File: "
                  "results/more_optimizers/table_imbalance_fitness.csv",
    "genedata": "Same protocol as the base paper: random 80/20 splits, 30 runs, a population of 10, 2,000 evaluations "
                "and 5-nearest neighbours. The rule was fixed before the run: the claim replicates only if our full "
                "model is higher on macro-F1 across the nine datasets, by a Wilcoxon test at 5 percent, with the "
                "search kept away from the reported rows. It is ahead on 7 of 9, but p is 0.13, so it does not "
                "replicate. With nine datasets the smallest possible p is 0.004. As secondary results, accuracy rises "
                "by 0.022 with p 0.039, and with paper-style testing our model is ahead with p 0.018, but the rule "
                "uses the clean test. The guided start alone is the clear winner here. The data are the GEMS files of "
                "Statnikov et al. 2005, from the authors’ site through the Internet Archive. They carry no licence, "
                "so we cite them and do not share them. File: results/base_datasets/base_datasets_across.csv",
    "result2": "Our earlier results use 5-nearest neighbours and the base paper uses XGBoost, so comparing those "
               "numbers would mix the selection with the classifier. Here both selections feed the same XGBoost. "
               "The two-class task uses the base paper's 420 patients. Both rules we fixed in advance failed: not "
               "better, and the interval reaches below the minus 0.02 margin we set for matching with fewer "
               "features. XGBoost on all attributes scores 3.3 points above our subset on two classes, because "
               "boosted trees already ignore weak attributes. The twelve-class filter is our extension of the "
               "paper's filter, with the ANOVA F statistic in place of Pearson correlation. File: "
               "results/same_classifier/same_classifier_claims.csv",
    "why95": "The paper's text leaves open whether its tuning saw the test patients, so we measured both cases. The "
             "gap between 88.8 and 79.6 is 9.2 points, with a 95 percent interval of 6.9 to 11.6. Fair tuning never "
             "scored more than 38 of 42 correct. Separately, a 42-patient test set is small: over 200 fair splits, "
             "the same pipeline scored anywhere from 71 to 90 percent between the 5th and 95th percentiles. Limits "
             "to state if asked: the paper gives no tuning ranges, so the ranges are ours, and our rebuilt filters "
             "share 86 of the 100 attributes the paper lists. File: results/tuning_on_test/tuning_on_test_summary.csv",
    "khaseeb95": "The paper does not say whether its search scored subsets on the training rows or on the 20% it "
                 "reports, so we ran both readings. Scoring on the reported rows gives 0.938, with 4 datasets at 1.00 "
                 "against the paper’s 5. Keeping the search away from them gives 0.775, with none at 1.00. Everything "
                 "else is the paper’s own protocol: 30 random 80/20 splits per dataset, a population of 10, 2,000 "
                 "evaluations and 5-nearest neighbours. Say that the printed figure is reproduced only by the "
                 "paper-style reading. Do not say the paper is wrong on purpose. File: "
                 "results/base_datasets/base_datasets_table.csv",
    "shortcuts": "Each shortcut was rebuilt next to a fair version that differs only in the shortcut. Row one is "
                 "twelve classes. Row two is two classes; on twelve classes the same shortcut moves macro-F1 from "
                 "0.36 to 0.45. A control with shuffled labels confirms the inflation is an artefact: with labels "
                 "that carry no information, the first shortcut still reports 0.83 macro-F1, and the second reports "
                 "0.69 to 0.74 balanced accuracy where chance is 0.50. The fair versions stay at chance. Files: "
                 "results/leakage/, results/test_reuse/, results/tuning_on_test/, results/small_split/",
    "claims": "We show this slide because it is what makes the positive results credible. Each claim and its pass "
              "rule were written in a dated log before the experiment ran. After the first claim failed we allowed "
              "ourselves one literature-based change, the guided start, tested once, with a stop rule against trying "
              "more variants on the same folds. The application-paper comparison used fresh folds for the same "
              "reason. The last two rows come from the method base paper’s own nine gene datasets, run under its "
              "protocol with the rule fixed beforehand.",
    "limits": "State these before the examiner does. The logistic regression reversal happens because our subsets "
              "are chosen for 5-nearest neighbours and hold about 8 attributes, while logistic regression does "
              "better with the larger subsets of the base method. Folds of repeated cross-validation overlap, so "
              "p-values are indicative. Mutual information uses scikit-learn’s default estimator, not the "
              "discretisation of the source paper. On improved porcupine optimisers: PLGCPO (Li and Dong 2024) and "
              "ICPO (He et al. 2025) do feature selection, and ECPO (Liu and Tao 2024) is continuous only. Their own "
              "tests have problems. PLGCPO’s lens step pulls points to the centre of the search range, where five of "
              "its six test functions have their answer, and ICPO computes its fitness on the test set. We have not "
              "run them.",
    "conclusion": "Close on the four findings, then future work. If asked what is new: the first binary grey wolf and "
                  "crested porcupine hybrid for feature selection; evidence across an ablation, 17 optimisers and "
                  "the base paper’s own data that the guided start carries the gain; and a fair re-test that "
                  "measures how far both base papers’ 95% results depend on the testing. If asked what failed: the "
                  "hybrid alone is level with the base hybrid, the class-balanced fitness had no effect, and the "
                  "full model’s gain did not replicate on the gene data. If asked what you would do differently: "
                  "start with a larger dataset, where a gain of one or two points can be resolved.",
    "references": "The numbers in square brackets on the slides point here, in order of first use. Every entry with "
                  "a DOI was checked against Crossref on 24 and 25 November 2026. Nogueira et al. has no DOI.",
    "thanks": "Close here and take questions. Two backup slides follow: terms in plain words, and the "
              "benchmark-function check.",
    "terms": "Keep this slide for questions. F1 is the harmonic mean of precision and recall: two times precision "
             "times recall, divided by their sum.",
    "benchmarks": "We asked for this check, and it stays in the appendix: no arrhythmia claim rests on it. As "
                  "published, our hybrid beats grey wolf plus swarm on 12 of 13 functions. With the answer moved, it "
                  "is better on 5 and worse on 1 with a loose swarm speed limit, and better on 1 and worse on 12 "
                  "with the usual limit. Plain crested porcupine beats it on 9 of 13 once the answer is moved. So "
                  "the functions give no robust evidence that our hybrid beats the base hybrid. Never show the "
                  "as-published column without the moved ones. Files: results/benchmark_functions/",
}


def emu(inches):
    return int(round(inches * 914400))


def r(text, size=14, color="tx1", bold=False, italic=False, spc=None, sup=False):
    attrs = f'lang="en-US" sz="{round(size * 100)}"'
    if bold:
        attrs += ' b="1"'
    if italic:
        attrs += ' i="1"'
    if spc:
        attrs += f' spc="{spc}" kern="0"'
    if sup:
        attrs += ' baseline="30000"'
    text = text.replace(" = ", " = ")
    keep = ' xml:space="preserve"' if text != text.strip() else ""
    return (f'<a:r><a:rPr {attrs} dirty="0"><a:solidFill><a:schemeClr val="{color}"/></a:solidFill></a:rPr>'
            f'<a:t{keep}>{escape(text)}</a:t></a:r>')


def p(*runs, size=14, align="l", bullet=False, before=0, after=0):
    spacing = ""
    if before:
        spacing += f'<a:spcBef><a:spcPts val="{before * 100}"/></a:spcBef>'
    if after:
        spacing += f'<a:spcAft><a:spcPts val="{after * 100}"/></a:spcAft>'
    if bullet:
        ppr = (f'<a:pPr marL="228600" indent="-228600" algn="{align}">{spacing}'
               f'<a:buSzPct val="100000"/><a:buChar char="&#x2022;"/></a:pPr>')
    else:
        ppr = f'<a:pPr marL="0" indent="0" algn="{align}">{spacing}<a:buNone/></a:pPr>'
    return f'<a:p>{ppr}{"".join(runs)}<a:endParaRPr lang="en-US" sz="{round(size * 100)}" dirty="0"/></a:p>'


def frag(xml):
    """Parse a fragment whose root tag needs the DrawingML namespace declarations."""
    head, rest = xml.split(">", 1)
    return parse_xml(f"{head} {NS}>{rest}")


BORDER = "".join(
    f'<a:{side} w="9525" cap="flat" cmpd="sng" algn="ctr"><a:solidFill><a:srgbClr val="BFBFBF"/></a:solidFill>'
    f'<a:prstDash val="solid"/><a:round/><a:headEnd type="none" w="med" len="med"/>'
    f'<a:tailEnd type="none" w="med" len="med"/></a:{side}>'
    for side in ("lnL", "lnR", "lnT", "lnB"))


class Canvas:
    def __init__(self, slide):
        self.tree = slide.shapes._spTree

    def _next_id(self):
        return max(int(el.get("id")) for el in self.tree.iter() if el.tag.endswith("}cNvPr")) + 1

    def _add(self, xml):
        el = frag(xml)
        self.tree.insert_element_before(el, "p:extLst")
        return el

    def text(self, x, y, w, h, paras, anchor="t"):
        sid = self._next_id()
        return self._add(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Text {sid}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln><a:noFill/></a:ln></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="0" tIns="0" rIns="0" bIns="0" rtlCol="0" anchor="{anchor}">'
            f'<a:noAutofit/></a:bodyPr><a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')

    def box(self, x, y, w, h, fill="bg2", line=None, line_w=0.75, geom="rect", paras=None, anchor="ctr",
            inset=(0.22, 0.08, 0.22, 0.08)):
        sid = self._next_id()
        fill_xml = f'<a:solidFill><a:schemeClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
        line_xml = (f'<a:ln w="{round(line_w * 12700)}"><a:solidFill><a:schemeClr val="{line}"/></a:solidFill></a:ln>'
                    if line else "<a:ln><a:noFill/></a:ln>")
        adj = '<a:gd name="adj" fmla="val 50000"/>' if geom == "roundRect" else ""
        body = ""
        if paras is not None:
            left, top, right, bottom = (emu(v) for v in inset)
            body = (f'<p:txBody><a:bodyPr wrap="square" lIns="{left}" tIns="{top}" rIns="{right}" bIns="{bottom}" '
                    f'rtlCol="0" anchor="{anchor}"><a:noAutofit/></a:bodyPr><a:lstStyle/>{"".join(paras)}</p:txBody>')
        return self._add(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{sid}" name="Box {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm>'
            f'<a:prstGeom prst="{geom}"><a:avLst>{adj}</a:avLst></a:prstGeom>{fill_xml}{line_xml}</p:spPr>'
            f'{body}</p:sp>')

    def circle(self, x, y, d, label, size=18):
        return self.box(x, y, d, d, fill="accent1", geom="ellipse", inset=(0, 0, 0, 0),
                        paras=[p(r(label, size, "bg1", bold=True), size=size, align="ctr")])

    def line(self, x1, y1, x2, y2, color="accent1", width=1.5, arrow=False):
        sid = self._next_id()
        flip = (' flipH="1"' if x2 < x1 else "") + (' flipV="1"' if y2 < y1 else "")
        tail = '<a:tailEnd type="triangle" w="med" len="med"/>' if arrow else ""
        return self._add(
            f'<p:cxnSp><p:nvCxnSpPr><p:cNvPr id="{sid}" name="Connector {sid}"/><p:cNvCxnSpPr/><p:nvPr/>'
            f'</p:nvCxnSpPr><p:spPr><a:xfrm{flip}><a:off x="{emu(min(x1, x2))}" y="{emu(min(y1, y2))}"/>'
            f'<a:ext cx="{emu(abs(x2 - x1))}" cy="{emu(abs(y2 - y1))}"/></a:xfrm>'
            f'<a:prstGeom prst="straightConnector1"><a:avLst/></a:prstGeom><a:ln w="{round(width * 12700)}">'
            f'<a:solidFill><a:schemeClr val="{color}"/></a:solidFill>{tail}</a:ln></p:spPr></p:cxnSp>')

    def table(self, x, y, widths, head, rows, row_h=0.55, head_h=0.48, size=14, ours=()):
        """A cell is a str, or (text, scheme colour, bold)."""
        sid = self._next_id()

        def cell(content, fill, color="tx1", bold=False):
            if isinstance(content, tuple):
                content, color, bold = content
            return (f'<a:tc><a:txBody><a:bodyPr/><a:lstStyle/>{p(r(content, size, color, bold), size=size)}'
                    f'</a:txBody><a:tcPr marL="101600" marR="101600" marT="50800" marB="50800" anchor="ctr">'
                    f'{BORDER}<a:solidFill><a:schemeClr val="{fill}"/></a:solidFill></a:tcPr></a:tc>')

        trs = [f'<a:tr h="{emu(head_h)}">' + "".join(cell(h, "accent1", "bg1", True) for h in head) + "</a:tr>"]
        for i, row in enumerate(rows):
            fill = "accent5" if i in ours else ("bg1" if i % 2 == 0 else "bg2")
            trs.append(f'<a:tr h="{emu(row_h)}">' + "".join(cell(c, fill) for c in row) + "</a:tr>")
        grid = "".join(f'<a:gridCol w="{emu(w)}"/>' for w in widths)
        return self._add(
            f'<p:graphicFrame><p:nvGraphicFramePr><p:cNvPr id="{sid}" name="Table {sid}"/>'
            f'<p:cNvGraphicFramePr><a:graphicFrameLocks noGrp="1"/></p:cNvGraphicFramePr><p:nvPr/>'
            f'</p:nvGraphicFramePr><p:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/>'
            f'<a:ext cx="{sum(emu(w) for w in widths)}" cy="{emu(head_h + row_h * len(rows))}"/></p:xfrm>'
            f'<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/table"><a:tbl>'
            f'<a:tblPr/><a:tblGrid>{grid}</a:tblGrid>{"".join(trs)}</a:tbl></a:graphicData></a:graphic>'
            f'</p:graphicFrame>')

    def callout(self, y, h, lead, rest, size=15, fill="accent5", lead_color="accent1", color="tx1"):
        return self.box(0.60, y, 12.13, h, fill=fill, paras=[p(
            r(lead, size, lead_color, bold=True), r(rest, size, color), size=size)])


def set_text(sp, text):
    """Write text into the first run of the first paragraph and drop the other runs and paragraphs."""
    paras = sp.findall(".//" + qn("a:p"))
    runs = paras[0].findall(qn("a:r"))
    runs[0].find(qn("a:t")).text = text
    for extra in runs[1:]:
        paras[0].remove(extra)
    for extra in paras[1:]:
        extra.getparent().remove(extra)


def shape_with_text(slide, prefix):
    hits = [sh for sh in slide.shapes if sh.has_text_frame and sh.text_frame.text.startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def chrome_protos(slide):
    found = {}
    for sp in slide.shapes._spTree.iterchildren(qn("p:sp")):
        ph = sp.find(".//" + qn("p:ph"))
        if ph is not None:
            found[{"body": "label", "title": "title", "sldNum": "num"}[ph.get("type")]] = copy.deepcopy(sp)
    assert set(found) == {"label", "title", "num"}, found.keys()
    return found


def content_slide(prs, layout, protos, label, title):
    slide = prs.slides.add_slide(layout)
    for ph in list(slide.placeholders):
        ph._element.getparent().remove(ph._element)
    tree = slide.shapes._spTree
    for key, text in (("label", label), ("title", title)):
        sp = copy.deepcopy(protos[key])
        set_text(sp, text)
        tree.insert_element_before(sp, "p:extLst")
    tree.insert_element_before(copy.deepcopy(protos["num"]), "p:extLst")
    return slide, Canvas(slide)


def relabel(slide, label, title):
    for sp in slide.shapes._spTree.iterchildren(qn("p:sp")):
        ph = sp.find(".//" + qn("p:ph"))
        if ph is not None and ph.get("type") in ("body", "title"):
            set_text(sp, label if ph.get("type") == "body" else title)


def read_csv(rel):
    with open(RESULTS / rel, newline="") as fh:
        return list(csv.DictReader(fh))


def guided_claim_numbers():
    rows = {(row["task"], row["metric"]): row for row in read_csv("informed_init/informed_init_claim.csv")
            if row["classifier"] == "KNN" and row["a"] == "GWO-CPO / imbalance / informed"
            and row["b"] == "IBGWO4 / accuracy / random"}
    got = {}
    for task, key in (("arrhythmia-multiclass", "12"), ("arrhythmia-binary", "2")):
        f1, acc, feats = (rows[(task, m)] for m in ("macro_f1", "acc", "n_features"))
        got[key] = dict(base=float(f1["mean_b"]), ours=float(f1["mean_a"]), wins=int(f1["wins"]),
                        p=float(f1["p_exploratory"]), acc_base=float(acc["mean_b"]), acc_ours=float(acc["mean_a"]),
                        feat_base=float(feats["mean_b"]), feat_ours=float(feats["mean_a"]),
                        fewer_in=int(feats["losses"]))
    expected = {"12": (0.353, 0.417, 21, 0.0001, 67.3, 69.4, 21, 8), "2": (0.703, 0.740, 19, 0.005, 72.0, 75.0, 31, 9)}
    for key, (b, o, wins, pv, ab, ao, fb, fo) in expected.items():
        g = got[key]
        assert (round(g["base"], 3), round(g["ours"], 3), g["wins"]) == (b, o, wins), (key, g)
        assert round(g["p"], 4 if key == "12" else 3) == pv, (key, g["p"])
        assert (round(g["acc_base"] * 100, 1), round(g["acc_ours"] * 100, 1)) == (ab, ao), (key, g)
        assert (round(g["feat_base"]), round(g["feat_ours"]), g["fewer_in"]) == (fb, fo, 25), (key, g)
    return got


def optimiser_scores():
    rows = [row for row in read_csv("more_optimizers/all_optimizers_summary.csv")
            if row["task"] == "arrhythmia-multiclass" and row["fitness"] == "imbalance" and row["classifier"] == "KNN"]
    scores = {start: {row["method"]: float(row["macro_f1_mean"]) for row in rows if row["start"] == start}
              for start in ("random", "informed")}
    rand, guided = scores["random"], scores["informed"]
    assert len(rand) == len(guided) == 17
    top15 = sorted(guided.values())[2:]
    assert round(max(rand.values()) - min(rand.values()), 2) == 0.13
    assert round(top15[-1] - top15[0], 3) == 0.041
    assert all(guided[m] > rand[m] for m in rand)
    assert sorted(guided, key=guided.get, reverse=True).index("GWO-CPO") == 1
    return scores


def swarm(items, in_per_x, in_per_y, diameter):
    """Offset each point vertically, nearest row first, so markers do not overlap."""
    step = diameter / in_per_y
    placed, out = [], []
    for method, value in items:
        for k in range(40):
            dy = ((k + 1) // 2) * step * (1 if k % 2 else -1)
            if all(((value - pv) * in_per_x) ** 2 + ((dy - pdy) * in_per_y) ** 2 >= diameter ** 2
                   for pv, pdy in placed):
                break
        placed.append((value, dy))
        out.append((method, value, dy))
    return out


def style_chart_text(chart):
    chart.font.size = Pt(12)
    chart.font.color.theme_color = MSO_THEME_COLOR.TEXT_1


def bar_chart(slide, x, y, w, h, categories, series, colors, title, vmax):
    data = CategoryChartData(number_format="0.000")
    data.categories = categories
    for name, values in series:
        data.add_series(name, values)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h),
                                   data).chart
    style_chart_text(chart)
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    title_font = chart.chart_title.text_frame.paragraphs[0].runs[0].font
    title_font.size, title_font.bold = Pt(14), False
    title_font.color.theme_color = MSO_THEME_COLOR.ACCENT_1
    plot = chart.plots[0]
    plot.gap_width, plot.overlap = 60, -10
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.number_format, labels.number_format_is_linked = "0.000", False
    labels.position = XL_LABEL_POSITION.OUTSIDE_END
    labels.font.size = Pt(13)
    for ser, color in zip(plot.series, colors):
        ser.format.fill.solid()
        ser.format.fill.fore_color.rgb = RGBColor.from_string(color)
        ser.invert_if_negative = False
    value_axis = chart.value_axis
    value_axis.minimum_scale, value_axis.maximum_scale = 0, vmax
    value_axis.has_major_gridlines = False
    value_axis.visible = False
    cat_axis = chart.category_axis
    cat_axis.reverse_order = True
    cat_axis.major_tick_mark = XL_TICK_MARK.NONE
    cat_axis.format.line.fill.background()
    cat_axis.tick_labels.font.size = Pt(13)
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.BOTTOM
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(12)
    return chart


def strip_chart(slide, c, scores):
    x_lo, x_hi, y_lo, y_hi = 0.25, 0.45, 0.35, 2.65
    cx, cy, cw, ch = 2.35, 1.45, 6.70, 3.95
    lx, ly, lw, lh = 0.02, 0.13, 0.95, 0.68
    in_per_x, in_per_y = cw * lw / (x_hi - x_lo), ch * lh / (y_hi - y_lo)
    diameter = 11 / 72 + 0.02
    rows = {"random": 2.0, "informed": 1.0}
    groups = {"other": [], "base": [], "ours": []}
    for start, row_y in rows.items():
        items = sorted(scores[start].items(), key=lambda kv: kv[1])
        for method, value, dy in swarm(items, in_per_x, in_per_y, diameter):
            key = {"GWO-CPO": "ours", "IBGWO4": "base"}.get(method, "other")
            groups[key].append((value, row_y + dy))
    data = XyChartData()
    names = {"other": "15 other optimisers", "base": "Base paper’s hybrid", "ours": "Our hybrid"}
    for key in ("other", "base", "ours"):
        ser = data.add_series(names[key], number_format="0.000")
        for px, py in groups[key]:
            ser.add_data_point(px, py)
    chart = slide.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(cx), Inches(cy), Inches(cw), Inches(ch),
                                   data).chart
    style_chart_text(chart)
    chart.has_title = False
    looks = [(XL_MARKER_STYLE.CIRCLE, 10, MARK_GREY), (XL_MARKER_STYLE.DIAMOND, 12, DARK_GREY),
             (XL_MARKER_STYLE.CIRCLE, 12, NAVY)]
    for ser, (style, size, color) in zip(chart.plots[0].series, looks):
        ser.smooth = False
        ser.format.line.fill.background()
        ser.marker.style, ser.marker.size = style, size
        ser.marker.format.fill.solid()
        ser.marker.format.fill.fore_color.rgb = RGBColor.from_string(color)
        ser.marker.format.line.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        ser.marker.format.line.width = Pt(0.75)
    x_axis = chart.category_axis
    x_axis.minimum_scale, x_axis.maximum_scale, x_axis.major_unit = x_lo, x_hi, 0.05
    x_axis.has_major_gridlines = True
    x_axis.major_gridlines.format.line.color.rgb = RGBColor.from_string("D9D9D9")
    x_axis.major_gridlines.format.line.width = Pt(0.75)
    x_axis.tick_labels.number_format, x_axis.tick_labels.number_format_is_linked = "0.00", False
    x_axis.tick_labels.font.size = Pt(12)
    x_axis.format.line.color.rgb = RGBColor.from_string("BFBFBF")
    x_axis.has_title = True
    x_axis.axis_title.text_frame.text = "Twelve-class macro-F1, mean over 25 test folds"
    axis_font = x_axis.axis_title.text_frame.paragraphs[0].runs[0].font
    axis_font.size, axis_font.bold = Pt(12), False
    axis_font.color.theme_color = MSO_THEME_COLOR.ACCENT_3
    y_axis = chart.value_axis
    y_axis.minimum_scale, y_axis.maximum_scale = y_lo, y_hi
    y_axis.has_major_gridlines = False
    y_axis.visible = False
    chart.has_legend = True
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.include_in_layout = False
    chart.legend.font.size = Pt(12)
    plot_area = chart._chartSpace.chart.plotArea
    old = plot_area.find(qn("c:layout"))
    if old is not None:
        plot_area.remove(old)
    plot_area.insert(0, parse_xml(
        '<c:layout xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart"><c:manualLayout>'
        '<c:layoutTarget val="inner"/><c:xMode val="edge"/><c:yMode val="edge"/>'
        f'<c:x val="{lx}"/><c:y val="{ly}"/><c:w val="{lw}"/><c:h val="{lh}"/></c:manualLayout></c:layout>'))
    for start, label in (("random", "Random start"), ("informed", "Guided start")):
        centre = cy + ch * ly + (y_hi - rows[start]) * in_per_y
        c.text(0.60, centre - 0.25, 1.70, 0.50, [p(r(label, 15, "accent1", bold=True), size=15)], anchor="ctr")


def build_base(c):
    cards = [
        (0.60, "METHOD BASE PAPER", "Khaseeb, Keshk and Youssef", "Applied Sciences, vol. 15, art. 489 [2]", [
            "Improves the binary grey wolf optimiser by adding a particle swarm step",
            "Prints 0.95 mean accuracy on nine gene datasets, from 80/20 hold-out splits and accuracy only",
            "Future work it names: cross-validation, other classifiers, other hybrids"]),
        (6.81, "APPLICATION BASE PAPER", "Dhanka and Maini", "Int. J. Cardiology, vol. 432, art. 133266 [3]", [
            "Two filters keep 100 attributes, and particle swarm tunes XGBoost",
            "Reports 95.24% for normal against arrhythmia, on 42 test patients",
            "Future work it names: more than two classes"]),
    ]
    for x, eyebrow, name, cite, bullets in cards:
        c.box(x, 1.55, 5.92, 3.70, fill="bg2")
        c.text(x + 0.30, 1.80, 5.32, 0.28, [p(r(eyebrow, 11, "accent2", bold=True, spc=150), size=11)])
        c.text(x + 0.30, 2.15, 5.32, 0.42, [p(r(name, 20, "accent1", bold=True), size=20)])
        c.text(x + 0.30, 2.62, 5.32, 0.30, [p(r(cite, 12, "accent3", italic=True), size=12)])
        c.text(x + 0.30, 3.12, 5.32, 2.00, [p(r(b, 15), size=15, bullet=True, after=10) for b in bullets])
    c.callout(5.55, 0.75, "We build on the first paper’s method ", "and test on the second paper’s dataset and classifier.")


def build_contrib(c):
    items = [
        ("A guided hybrid selector", "The first binary grey wolf and crested porcupine hybrid for feature selection "
                                     "[4], started from informative subsets."),
        ("Where the gain comes from", "An ablation, 17 optimisers and the base paper’s own nine datasets show that "
                                      "the guided start carries the gain."),
        ("A fair re-test of both base papers", "Both 95% results return only when the search or tuning sees the "
                                               "test patients. Four shortcuts measured."),
    ]
    for i, (head, note) in enumerate(items):
        x = 0.60 + i * 4.145
        c.box(x, 1.55, 3.84, 3.45, fill="bg2")
        c.circle(x + 0.30, 1.85, 0.62, str(i + 1))
        c.text(x + 0.30, 2.70, 3.24, 0.70, [p(r(head, 18, "accent1", bold=True), size=18)])
        c.text(x + 0.30, 3.50, 3.24, 1.35, [p(r(note, 14, "accent3"), size=14)])
    c.callout(5.25, 1.30, "Every claim was written down with its pass rule before the experiment ran. ",
              "We report the claims that failed as well as those that held.", size=16)


def build_method(c):
    steps = [
        ("Guided start", "Informative attributes are more likely to start in a subset [5].", True),
        ("Grey wolf step", "Every subset moves towards the three best found so far [6].", False),
        ("Porcupine step", "One of four defence moves [7]. Replaces the particle swarm step.", True),
        ("Keep the best", "The best 20 of the old and new subsets go to the next round.", False),
    ]
    w, gap = 2.75, 0.377
    for i, (head, note, changed) in enumerate(steps):
        x = 0.60 + i * (w + gap)
        c.box(x, 1.55, w, 3.05, fill="accent5" if changed else "bg2")
        c.circle(x + 0.25, 1.80, 0.55, str(i + 1), size=16)
        c.text(x + 0.25, 2.55, w - 0.5, 0.40, [p(r(head, 17, "accent1", bold=True), size=17)])
        c.text(x + 0.25, 3.00, w - 0.5, 1.00, [p(r(note, 13), size=13)])
        pill = ("CHANGED BY US", "accent1", None, "bg1") if changed else ("FROM BASE PAPER", "bg1", "accent6", "accent3")
        c.box(x + 0.25, 4.08, 1.75, 0.32, fill=pill[1], line=pill[2], geom="roundRect", inset=(0.04, 0, 0.04, 0),
              paras=[p(r(pill[0], 10, pill[3], bold=True, spc=50), size=10, align="ctr")])
        if i < 3:
            c.line(x + w + 0.06, 3.075, x + w + gap - 0.06, 3.075, width=1.75, arrow=True)
    left, right = 0.60 + (w + gap), 0.60 + 3 * (w + gap) + w
    c.line(left, 4.72, left, 4.86, width=1.25)
    c.line(left, 4.86, right, 4.86, width=1.25)
    c.line(right, 4.72, right, 4.86, width=1.25)
    c.text(left, 4.95, right - left, 0.32, [p(r("Steps 2 to 4 repeat until 2,000 evaluations are used", 13,
                                                "accent1", bold=True), size=13, align="ctr")])
    c.callout(5.55, 1.05, "Fitness. ", "Each evaluation trains a 5-nearest-neighbour classifier on the kept attributes "
              "and scores it with every class counted equally, plus a small penalty for each attribute kept.",
              size=14, fill="bg2")


def build_protocol(c):
    c.box(0.60, 1.55, 5.45, 3.80, fill="accent5")
    c.text(0.85, 1.75, 5.00, 0.35, [p(r("Training patients, 80% of each fold", 15, "accent1", bold=True), size=15)])
    chips = ["Scaling and gap filling are fitted here", "The guided start is computed here",
             "The search scores subsets on an inner 3-fold split"]
    for i, chip in enumerate(chips):
        c.box(0.85, 2.25 + i * 0.68, 4.95, 0.52, fill="bg1", line="accent1", inset=(0.15, 0.03, 0.15, 0.03),
              paras=[p(r(chip, 13), size=13)])
    c.text(0.85, 4.45, 5.00, 0.65, [p(r("Everything that chooses something stays inside this box.", 13, "accent1",
                                        bold=True), size=13)])
    c.box(6.30, 1.55, 1.80, 3.80, fill="accent1", inset=(0.18, 0.1, 0.18, 0.1), paras=[
        p(r("Test patients, 20%", 15, "bg1", bold=True), size=15),
        p(r("Used once, to score the chosen subset.", 12, "bg1"), size=12, before=8)])
    c.box(8.35, 1.55, 4.38, 3.80, fill="bg2")
    c.text(8.60, 1.75, 3.90, 0.35, [p(r("Same conditions for every method", 15, "accent1", bold=True), size=15)])
    facts = ["25 folds per task: 5-fold cross-validation, repeated 5 times",
             "Every optimiser gets the same 20 starting subsets and the same 2,000 evaluations",
             "Compared fold by fold, with a bootstrap interval and a Wilcoxon test",
             "Nested cross-validation: the inner loop chooses, the outer loop measures [8]"]
    c.text(8.60, 2.25, 3.90, 3.00, [p(r(f, 13), size=13, bullet=True, after=8) for f in facts])
    c.callout(5.62, 0.95, "Check. ", "Replacing the test patients with random noise leaves every selected subset "
              "unchanged.")


def build_result1(slide, c, nums):
    bar_chart(slide, 0.60, 1.45, 7.40, 3.70, ["Twelve classes", "Two classes"],
              [("Base method, as published", (nums["12"]["base"], nums["2"]["base"])),
               ("Proposed", (nums["12"]["ours"], nums["2"]["ours"]))],
              [MARK_GREY, NAVY], "Mean macro-F1 over 25 test folds", 0.85)
    cards = [(1.55, "8 features, not 21", "Twelve classes. Higher in 21 of 25 folds, p = 0.0001."),
             (3.45, "9 features, not 31", "Two classes. Higher in 19 of 25 folds, p = 0.005.")]
    for y, big, note in cards:
        c.box(8.35, y, 4.38, 1.70, fill="bg2")
        c.text(8.60, y + 0.22, 3.90, 0.50, [p(r(big, 26, "accent1", bold=True), size=26)])
        c.text(8.60, y + 0.85, 3.90, 0.70, [p(r(note, 13, "accent3"), size=13)])
    c.text(0.60, 5.22, 7.40, 0.28, [p(r("5-nearest neighbours. Base method: IBGWO4 as published, run under our "
                                        "testing.", 11, "accent3"), size=11)])
    c.callout(5.62, 0.98, "Accuracy also rises, ", "from 67.3% to 69.4% on twelve classes and from 72.0% to 75.0% on "
              "two. The proposed method picks fewer features than the base method in all 25 folds.", size=14)


def build_ablation(c):
    rows = [
        [("Guided start, added to the unchanged base method", "accent1", True), ("+0.059 (p = 0.001)", "accent1", True),
         ("+0.028 (p = 0.13)", "accent1", True)],
        ["Porcupine step in place of the particle swarm step", "+0.010 (p = 0.67)", "+0.024 (p = 0.11)"],
        ["Class-balanced fitness in place of the accuracy fitness", "−0.004 (p = 0.96)", "+0.006 (p = 0.73)"],
        ["Proposed against the base method, both with the guided start", "+0.005 (p = 0.47)", "+0.009 (p = 0.18)"],
    ]
    c.table(0.60, 1.55, [6.73, 2.70, 2.70], ["One change switched on", "Twelve classes", "Two classes"], rows,
            row_h=0.62, head_h=0.50, ours={0})
    c.text(0.60, 4.62, 12.13, 0.28, [p(r("Change in macro-F1, paired over 25 test folds, 5-nearest neighbours.", 11,
                                         "accent3"), size=11)])
    c.callout(5.15, 1.30, "The guided start lifts all five optimisers we tested, including the base paper’s own "
              "method. ", "Changing the optimiser or the fitness moves the score by less than the fold-to-fold noise.")


def build_optimisers(slide, c, scores):
    strip_chart(slide, c, scores)
    c.box(9.30, 1.55, 3.43, 3.80, fill="bg2")
    stats = [(1.75, "0.13", "between the best and worst optimiser from a random start"),
             (2.95, "about 0.04", "spread of 15 of 17 optimisers from the guided start"),
             (4.15, "17 of 17", "score higher from the guided start than from a random start")]
    for y, big, note in stats:
        c.text(9.55, y, 2.95, 0.45, [p(r(big, 24, "accent1", bold=True), size=24)])
        c.text(9.55, y + 0.45, 2.95, 0.65, [p(r(note, 12, "accent3"), size=12)])
    c.callout(5.62, 0.98, "The start, not the optimiser, sets the score. ", "Our hybrid ranks second of 17 on twelve "
              "classes and first on two, with gaps inside the noise.", size=14)


def build_genedata(c):
    c.text(0.60, 1.50, 12.13, 0.60, [p(r("Nine cancer gene datasets under the base paper’s protocol: 30 random 80/20 "
                                         "splits each, with the search kept away from the reported patients.", 15),
                                       size=15)])
    rows = [
        [("Guided start added", "accent1", True), ("+0.034 accuracy", "accent1", True), ("8 of 9", "accent1", True),
         ("Higher, p = 0.012", "accent1", True)],
        ["Porcupine step, not the swarm step", "−0.009 accuracy", "4 of 9", "No clear difference, p = 0.30"],
        ["Our full model, the claim under test", "+0.028 macro-F1", "7 of 9", ("Not replicated, p = 0.13", "accent4", True)],
    ]
    c.table(0.60, 2.25, [4.60, 2.35, 1.55, 3.63],
            ["Against the base method as published", "Change", "Higher on", "Verdict across datasets"], rows,
            row_h=0.55, ours={0})
    c.text(0.60, 4.50, 12.13, 0.28, [p(r("Wilcoxon test on the nine dataset means. Data: the GEMS collection of "
                                         "Statnikov et al. [9].", 11, "accent3"), size=11)])
    c.callout(5.05, 1.05, "The guided start carries over to the base paper’s own data ", "and keeps about 7 genes in "
              "place of 112 on the median dataset. Our full model’s gain does not replicate there.")


def build_result2(c):
    rows = [
        ["All attributes", "80.3%", "0.466", "about 255"],
        ["The base paper’s two filters", "78.4%", "0.470", "100"],
        [("Proposed selection", "accent1", True), ("77.0%", "accent1", True), ("0.466", "accent1", True),
         ("13 and 24", "accent1", True)],
    ]
    c.table(0.60, 1.55, [4.73, 2.40, 2.60, 2.40],
            ["Attributes given to XGBoost", "Two-class accuracy", "Twelve-class macro-F1", "Attributes used"], rows,
            row_h=0.58, ours={2})
    c.text(0.60, 3.87, 12.13, 0.28, [p(r("Default XGBoost [10] for every row, 25 fresh folds. The search wraps a light "
                                         "XGBoost.", 11, "accent3"), size=11)])
    c.callout(4.40, 1.40, "No clear difference in score. ", "The gaps of 1.4 points and 0.004 are inside the noise "
              "(p = 0.22 and 0.77). The proposed selection does not beat the filter. It reaches a similar score with "
              "13 to 24 attributes in place of 100.")
    c.text(0.60, 6.05, 12.13, 0.40, [p(r("For a strong classifier, feature selection buys a smaller model, not a higher "
                                         "score.", 15, "accent1", bold=True), size=15)])


def build_three_numbers(c, cards, note, caption):
    for i, (num, color, head, text) in enumerate(cards):
        x = 0.60 + i * 4.145
        c.box(x, 1.55, 3.84, 2.95, fill="bg2")
        c.text(x + 0.30, 1.78, 3.24, 0.62, [p(r(num, 34, color, bold=True), size=34)])
        c.text(x + 0.30, 2.50, 3.24, 0.55, [p(r(head, 15, "accent1", bold=True), size=15)])
        c.text(x + 0.30, 3.10, 3.24, 1.30, [p(r(text, 13, "accent3"), size=13)])
    c.callout(4.75, 1.05, "Note. ", note, size=13, fill="bg2")
    c.text(0.60, 5.98, 12.13, 0.30, [p(r(caption, 11, "accent3"), size=11)])


def build_why95(c):
    build_three_numbers(c, [
        ("88.8%", "accent4", "Tuned on the test patients", "What the base paper’s pipeline prints when its tuning can "
                                                           "see the 42 test patients. It reaches 95.24% in 5 of 40 "
                                                           "splits."),
        ("79.6%", "accent1", "Same models, new patients", "The score of those same tuned models on 42 patients that no "
                                                          "step had seen."),
        ("80.3%", "accent1", "Tuned fairly", "The pipeline’s score when tuning uses the training patients only. This is "
                                             "the honest bar to compare with."),
    ], "We make no claim about how the published result was produced. We report only that its figure could not be "
       "reproduced under fair testing.",
        "The base paper’s filters and XGBoost, rebuilt, over 40 random splits.")


def build_khaseeb95(c):
    build_three_numbers(c, [
        ("0.95", "tx1", "Printed in the paper", "Mean accuracy over the nine gene datasets, five of them at 1.00."),
        ("0.938", "accent4", "Search scored on the reported rows", "Our rerun when the search scores subsets on the "
                                                                   "20% it then reports."),
        ("0.775", "accent1", "Search never sees those rows", "Our rerun when the search scores subsets inside the 80% "
                                                             "only."),
    ], "We make no claim about how the published result was produced. The paper does not say which rows its search "
       "scored, and only the reading that scores on the reported rows reproduces its figure.",
        "IBGWO4 as published, under the paper’s protocol [2]: 30 random 80/20 splits per dataset.")


def build_shortcuts(c):
    bad = "accent4"
    rows = [
        ["Balancing classes and filtering before cross-validation", "0.37 macro-F1", ("0.91 macro-F1", bad, True)],
        ["Letting the search score subsets on the test split", "0.71 macro-F1", ("0.84 macro-F1", bad, True)],
        ["Tuning the classifier on the 42 test patients", "79.6% accuracy", ("88.8% accuracy", bad, True)],
        ["Reporting one split of 42 test patients", "81% median", ("71% to 90% by luck of the split", bad, True)],
    ]
    c.table(0.60, 1.55, [5.93, 2.55, 3.65], ["Shortcut", "Tested fairly", "With the shortcut"], rows, row_h=0.58)
    c.text(0.60, 4.45, 12.13, 0.28, [p(r("Each shortcut rebuilt next to a fair version that differs only in the "
                                         "shortcut. Leakage of this kind is common in published work [11].", 11,
                                         "accent3"), size=11)])
    c.callout(5.00, 1.45, "On this dataset, how a method is tested matters more than which method it is. ",
              "Our best method change gained 0.04 to 0.06 macro-F1. These shortcuts move a score by 0.09 to 0.54.")


def build_claims(c):
    good, bad = "accent1", "accent4"
    rows = [
        ["The hybrid with a random start beats the base method", ("Failed: comparable", bad, True)],
        ["The hybrid beats grey wolf alone and porcupine alone", ("Failed", bad, True)],
        ["The class-balanced fitness raises macro-F1", ("Failed: no effect", bad, True)],
        ["Feature selection beats all attributes (5-nearest neighbours)", ("Held: +0.13 to +0.15", good, True)],
        ["The guided hybrid beats the base method as published", ("Held on both tasks", good, True)],
        ["The guided hybrid beats the application paper’s filter", ("Not held: no clear difference", bad, True)],
        ["The evaluation shortcuts inflate reported scores", ("Held", good, True)],
        ["The guided start lifts the base method on its nine datasets", ("Held: 8 of 9, p = 0.012", good, True)],
        ["The guided hybrid beats the base method on those datasets", ("Not held: 7 of 9, p = 0.13", bad, True)],
    ]
    c.table(0.60, 1.50, [8.73, 3.40], ["Claim fixed before the experiment", "Outcome"], rows, row_h=0.42,
            head_h=0.45, size=13)
    c.text(0.60, 5.88, 12.13, 0.30, [p(r("Each claim and its pass rule were logged with a date before the run: "
                                         "code/design/decisions.tsv", 11, "accent3"), size=11)])


def build_limits(c):
    bullets = [
        "UCI Arrhythmia has 452 patients, and our full model’s gain did not replicate on the base paper’s gene data "
        "(p = 0.13).",
        "The gain over the method base paper holds for 5-nearest neighbours. It reverses for logistic regression.",
        "With XGBoost, selection gives a smaller model, not a higher score.",
        "We compared with the original crested porcupine optimiser only, not improved versions such as PLGCPO [12], "
        "ICPO [13] or ECPO [14].",
        "The selected attributes change from fold to fold, so the subsets are not yet stable.",
        "Classes with 4 or 5 patients are almost never recognised by any method.",
        "No cardiologist has reviewed the selected attributes.",
    ]
    c.text(0.60, 1.55, 7.70, 4.95, [p(r(b, 15), size=15, bullet=True, after=9) for b in bullets])
    c.box(8.70, 1.55, 4.03, 4.20, fill="bg2")
    c.text(8.95, 1.75, 3.55, 0.35, [p(r("Subsets are not yet stable", 15, "accent1", bold=True), size=15)])
    c.text(8.95, 2.25, 3.55, 0.55, [p(r("0.11 to 0.22", 28, "accent1", bold=True), size=28)])
    c.text(8.95, 2.85, 3.55, 0.65, [p(r("stability index of the search methods with a random start", 13, "accent3"),
                                      size=13)])
    c.text(8.95, 3.70, 3.55, 0.55, [p(r("0.60", 28, "accent3", bold=True), size=28)])
    c.text(8.95, 4.30, 3.55, 1.20, [p(r("for a simple filter. An index of 1 means the same subset in every fold [15].",
                                        13, "accent3"), size=13)])


def build_conclusion(c):
    statements = [
        "A guided start lifts twelve-class macro-F1 from 0.353 to 0.417 over the method base paper, with 8 features "
        "in place of 21.",
        "The start produces the gain. It lifts all 17 optimisers we ran, and the base paper’s own method on 8 of its 9 "
        "datasets.",
        "Our full model’s gain did not carry over to those gene datasets. It was higher on 7 of 9, with p = 0.13.",
        "Both base papers’ 95% results return only when the search or tuning sees the test patients. Tested fairly, "
        "they score 80.3% and 77.5%.",
    ]
    for i, statement in enumerate(statements):
        y = 1.60 + i * 0.93
        c.circle(0.60, y + 0.05, 0.52, str(i + 1), size=16)
        c.text(1.40, y, 11.33, 0.62, [p(r(statement, 16), size=16)], anchor="ctr")
    c.box(0.60, 5.40, 12.13, 1.15, fill="accent1", inset=(0.30, 0.1, 0.30, 0.1), paras=[p(
        r("Next. ", 15, "bg1", bold=True),
        r("A study on fresh folds of which choice matters most (missing values, selection or classifier), then larger "
          "ECG datasets with patient-wise splits.", 15, "bg1"), size=15)])


def build_terms(c):
    rows = [
        [("Wrapper feature selection", "accent1", True), "Try a subset of attributes, train a classifier on it, and "
                                                         "use the classifier’s score to judge the subset."],
        [("Mutual information", "accent1", True), "How much knowing an attribute reduces uncertainty about the class. "
                                                  "Zero means it tells us nothing."],
        [("Macro-F1", "accent1", True), "The F1 score of each class, averaged so that a class of 4 patients counts as "
                                        "much as a class of 245."],
        [("Balanced accuracy", "accent1", True), "The share of each class that is recognised, averaged over the "
                                                 "classes."],
        [("Nested cross-validation", "accent1", True), "An inner loop chooses the subset. An outer loop, which the "
                                                       "inner loop never sees, measures it."],
        [("Leakage", "accent1", True), "Information from the test patients reaching a step that chooses something. It "
                                       "makes a score too high."],
    ]
    c.table(0.60, 1.50, [3.40, 8.73], ["Term", "Meaning"], rows, row_h=0.62)
    c.callout(5.85, 0.80, "Abbreviations. ", "GWO, grey wolf optimiser. PSO, particle swarm optimisation. CPO, crested "
              "porcupine optimiser. IBGWO4, the method base paper’s best variant.", size=13, fill="bg2")


def build_benchmarks(c):
    rows = [
        ["Particle swarm", "4.74", "4.53", "2.15"],
        ["Grey wolf", "3.13", "3.17", "4.20"],
        ["Crested porcupine", "1.68", "1.60", "2.36"],
        ["Grey wolf + swarm, base paper", "3.63", "2.97", "2.47"],
        [("Grey wolf + porcupine, ours", "accent1", True), ("1.83", "accent1", True), ("2.73", "accent1", True),
         ("3.83", "accent4", True)],
    ]
    c.table(0.60, 1.50, [4.13, 2.50, 2.75, 2.75],
            ["Optimiser, mean rank of 5", "Answer at zero, as published", "Answer moved, loose swarm limit",
             "Answer moved, usual swarm limit"], rows, row_h=0.48, head_h=0.70, ours={4})
    c.text(0.60, 4.70, 12.13, 0.28, [p(r("Functions F1 to F13, 30 dimensions, 30 runs, 15,000 evaluations. Lower rank "
                                         "is better. Appendix only.", 11, "accent3"), size=11)])
    c.box(0.60, 5.15, 12.13, 1.35, fill="accent5", paras=[p(
        r("The textbook form flatters grey-wolf-type steps. ", 15, "accent1", bold=True),
        r("With the answer at zero, our hybrid reaches 6 × 10", 15), r("−113", 15, sup=True),
        r(" on the sphere function. With the answer moved, it reaches about 1,700, because such steps drift toward "
          "zero [16].", 15), size=15)])


def edit_cover(slide):
    sub = shape_with_text(slide, "Final-Year B.Tech Project")
    set_text(sub._element, "Final-Year B.Tech Project  ·  Project Review")
    set_text(shape_with_text(slide, "Minority-Aware")._element, TITLE)


def edit_outline(slide):
    items = [
        ("The Problem", "Many ECG measurements, few patients and rare classes"),
        ("Two Base Papers", "The method we extend and the result we re-test"),
        ("What We Contribute", "A guided hybrid selector, where its gain comes from, and a fair re-test"),
        ("The Proposed Method", "Guided start, grey wolf step, porcupine step, keep the best"),
        ("How We Test", "Nested cross-validation: the search never sees the test patients"),
        ("Results", "Both base papers, 17 optimisers and the base paper’s nine gene datasets"),
        ("The 95% Results", "Why we do not report 95%, and four shortcuts that inflate scores"),
        ("Outcomes", "What held and what failed, limitations and conclusion"),
    ]
    boxes = [sh for sh in slide.shapes if sh.has_text_frame and not sh.is_placeholder
             and sh.text_frame.paragraphs[0].runs and sh.width > Inches(3)]
    heads = sorted((sh for sh in boxes if sh.text_frame.paragraphs[0].runs[0].font.size == Pt(18)),
                   key=lambda sh: (sh.left, sh.top))
    assert len(heads) == 8, len(heads)
    for head, (title, desc) in zip(heads, items):
        below = [sh for sh in boxes if sh.left == head.left and sh.top > head.top
                 and sh.text_frame.paragraphs[0].runs[0].font.size == Pt(14)]
        set_text(head._element, title)
        set_text(min(below, key=lambda sh: sh.top)._element, desc)


def edit_problem(slide):
    relabel(slide, "01 · THE PROBLEM", "Many Measurements, Few Patients")
    swaps = {"patient records": "patients", "attributes per record": "ECG measurements per patient",
             "13 of 16": "12", "classes have records": "classes; the smallest has 4 patients", "83%": "54%",
             "of records miss the J angle": "accuracy from always answering “normal”",
             "Red: classes with five or fewer": "Red: five or fewer patients. Classes 7 and 8 are merged in our "
                                                "twelve-class task. Data: [1]."}
    for old, new in swaps.items():
        set_text(shape_with_text(slide, old)._element, new)
    drop = ("Attribute structure", "15 general", "144 width", "120 amplitude")
    for sh in list(slide.shapes):
        if sh.name.startswith(("Attribute segment", "Legend key")) or (
                sh.has_text_frame and sh.text_frame.text.startswith(drop)):
            sh._element.getparent().remove(sh._element)
    Canvas(slide).box(0.60, 4.62, 4.60, 1.85, fill="accent5", paras=[p(
        r("So we judge every method by macro-F1: ", 15, "accent1", bold=True),
        r("the F1 score of each class, averaged with every class counted equally. Accuracy rewards ignoring the rare "
          "classes.", 15), size=15)])


def edit_references(slide):
    frames = [sh for sh in slide.shapes if sh.has_table]
    assert len(frames) == 2
    numbered = list(enumerate(REFS, 1))
    for frame, chunk in zip(sorted(frames, key=lambda sh: sh.left), (numbered[:8], numbered[8:])):
        tbl = frame._element.find(".//" + qn("a:tbl"))
        rows = tbl.findall(qn("a:tr"))
        template = rows[0]
        for row in rows:
            tbl.remove(row)
        for n, (before, journal, after) in chunk:
            tr = copy.deepcopy(template)
            num_cell, text_cell = tr.findall(qn("a:tc"))
            for tc, new_p in ((num_cell, p(r(f"[{n}]", 10, "accent1", bold=True), size=10)),
                              (text_cell, p(r(before, 10), r(journal, 10, italic=True), r(after, 10), size=10))):
                old_p = tc.find(".//" + qn("a:p"))
                old_p.getparent().replace(old_p, frag(new_p))
            tbl.append(tr)


def edit_closing(slide):
    runs = shape_with_text(slide, "Key message").text_frame.paragraphs[0].runs
    runs[1].text = ("Test fairly first. Under fair testing, the guided start carries the gain, and both published "
                    "95% results need the test patients to be seen.")
    for extra in runs[2:]:
        extra._r.getparent().remove(extra._r)


def reorder_and_prune(prs, ordered):
    sld_id_lst = prs.slides._sldIdLst
    by_part = {prs.part.related_part(sld_id.rId): sld_id for sld_id in sld_id_lst}
    wanted = [by_part[slide.part] for slide in ordered]
    for sld_id in list(sld_id_lst):
        if not any(sld_id is w for w in wanted):
            rid = sld_id.rId
            sld_id_lst.remove(sld_id)
            prs.part.drop_rel(rid)
    for sld_id in wanted:
        sld_id_lst.remove(sld_id)
        sld_id_lst.append(sld_id)


def main():
    nums = guided_claim_numbers()
    scores = optimiser_scores()
    prs = Presentation(SRC)
    old = list(prs.slides)
    cover, outline, problem, refs, closing = old[0], old[1], old[3], old[14], old[16]
    protos = chrome_protos(old[2])
    layout = old[2].slide_layout
    for sp in layout.shapes._spTree.iter(qn("a:t")):
        if sp.text and sp.text.startswith("Lead-Aware"):
            sp.text = FOOTER

    edit_cover(cover)
    edit_outline(outline)
    edit_problem(problem)
    edit_references(refs)
    edit_closing(closing)

    plan = [
        ("base", "02 · BASE PAPERS", "Two Base Papers, Both from 2025", build_base),
        ("contrib", "03 · CONTRIBUTIONS", "What We Contribute", build_contrib),
        ("method", "04 · METHOD", "The Proposed Method, Step by Step", build_method),
        ("protocol", "05 · TESTING", "The Search Never Sees the Test Patients", build_protocol),
        ("result1", "06 · RESULTS", "Result 1: Higher Score, Fewer Features", None),
        ("ablation", "06 · RESULTS", "The Gain Comes from the Start", build_ablation),
        ("optimisers", "06 · RESULTS", "The Start Matters More than the Optimiser", None),
        ("genedata", "06 · RESULTS", "Replication on the Base Paper’s Own Data", build_genedata),
        ("result2", "06 · RESULTS", "Result 2: Against the Application Paper", build_result2),
        ("why95", "07 · THE 95% RESULTS", "Why We Do Not Report 95%", build_why95),
        ("khaseeb95", "07 · THE 95% RESULTS", "Khaseeb et al.’s 0.95, Tested Two Ways", build_khaseeb95),
        ("shortcuts", "07 · THE 95% RESULTS", "Four Shortcuts That Inflate a Score", build_shortcuts),
        ("claims", "08 · OUTCOMES", "What Held and What Failed", build_claims),
        ("limits", "08 · OUTCOMES", "Limitations", build_limits),
        ("conclusion", "08 · OUTCOMES", "Conclusion", build_conclusion),
        ("terms", "BACKUP", "Terms in Plain Words", build_terms),
        ("benchmarks", "BACKUP", "Standard Benchmark Functions", build_benchmarks),
    ]
    made = {}
    for key, label, title, build in plan:
        slide, canvas = content_slide(prs, layout, protos, label, title)
        if key == "result1":
            build_result1(slide, canvas, nums)
        elif key == "optimisers":
            build_optimisers(slide, canvas, scores)
        else:
            build(canvas)
        made[key] = slide

    order = [("cover", cover), ("outline", outline), ("problem", problem)]
    order += [(k, made[k]) for k in ("base", "contrib", "method", "protocol", "result1", "ablation", "optimisers",
                                     "genedata", "result2", "why95", "khaseeb95", "shortcuts", "claims", "limits",
                                     "conclusion")]
    order += [("references", refs), ("thanks", closing), ("terms", made["terms"]), ("benchmarks", made["benchmarks"])]
    reorder_and_prune(prs, [slide for _, slide in order])
    for key, slide in order:
        slide.notes_slide.notes_text_frame.text = NOTES[key]
    for i, slide in enumerate(prs.slides, 1):
        for fld in slide._element.iter(qn("a:fld")):
            if fld.get("type") == "slidenum":
                fld.find(qn("a:t")).text = str(i)

    prs.core_properties.title = TITLE
    prs.core_properties.subject = "B.Tech project review"
    prs.save(OUT)

    with zipfile.ZipFile(OUT) as z:
        leftovers = [(name, word) for name in z.namelist() if name.endswith(".xml")
                     for word in STALE if word in z.read(name).decode("utf-8")]
    assert not leftovers, leftovers
    print(f"wrote {OUT} with {len(prs.slides)} slides")


if __name__ == "__main__":
    main()
