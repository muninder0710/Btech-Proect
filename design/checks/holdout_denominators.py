"""Which test-set size explains the 'Best' fitness values in Geng & Li 2024, Table 4?

Fitness there is 1 - accuracy (their Eq. 31), so one run's fitness is k/m for a
test set of m records. Values are printed to 4 decimals, so a candidate m fits a
value v when some integer k gives |v - k/m| <= 0.00005. For each dataset we find
the smallest m that fits every Best value, and compare it with ceil(0.3 n).

Usage: python holdout_denominators.py <Europe PMC text of PMC11352133>
"""
import math
import re
import sys

N = {"Wine": 178, "Sonar": 208, "Breast Cancer": 569, "Heart": 270, "Dermatology": 366,
     "Arrhythmia": 452, "Lymphography": 148, "SPECT": 267, "Zoo": 101, "Semeion": 1953,
     "Isolet": 1560, "Ionosphere": 351, "Musk": 476, "Hill-Valley": 606, "Amazon": 1500,
     "Arcene": 200, "DLBCL": 77, "Micro Mass": 571, "Leukemia": 72, "Prostate": 102, "Colon": 62}

lines = open(sys.argv[1], encoding="utf-8").read().splitlines()
headers = [i for i, l in enumerate(lines) if re.search(r"\|\s*BABC\s*\|\s*BBA\s*\|", l)]
table4 = lines[headers[0]:headers[1]]

best, current = {}, None
for l in table4:
    cells = [c.strip() for c in l.split("|") if c.strip()]
    if len(cells) >= 2 and cells[1] == "Mean" and cells[0] in N:
        current = cells[0]
    elif cells and cells[0] == "Best" and current:
        best[current] = [float(c) for c in cells[1:13]]


def fits(m, values):
    return all(abs(v - round(v * m) / m) <= 0.00005 + 1e-12 for v in values)


print(f"{'dataset':14s} {'n':>5s} {'ceil(0.3n)':>10s} {'smallest m':>10s}  agrees")
agree = 0
for name, values in best.items():
    expected = math.ceil(0.3 * N[name])
    smallest = next(m for m in range(1, N[name] + 1) if fits(m, values))
    ok = smallest == expected or (expected % smallest == 0 and fits(expected, values))
    agree += ok
    print(f"{name:14s} {N[name]:5d} {expected:10d} {smallest:10d}  {'yes' if ok else 'NO'}")
print(f"{agree} of {len(best)} datasets fit a 30% hold-out")
