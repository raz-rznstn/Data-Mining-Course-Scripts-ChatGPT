from itertools import combinations
from collections import defaultdict
from math import ceil

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

# ────────────────────────── Apriori ──────────────────────────
def apriori(transactions, min_support=0.5):
    n_tx   = len(transactions)
    mincnt = min_support if isinstance(min_support, int) else ceil(min_support * n_tx)

    # L1
    freq = {(item,): 0 for tx in transactions for item in tx}
    for tx in transactions:
        for it in tx:
            freq[(it,)] += 1
    print(f"{YELLOW}\nC1 (Candidates size‑1):{RESET}")
    for it, c in sorted(freq.items()):
        print(f"{_set_str(it):<10} {c:>2} (/{n_tx})")
    freq = {k: v for k, v in freq.items() if v >= mincnt}

    print(f"{GREEN}\nL1 (Frequent size‑1):{RESET}")
    for it, c in sorted(freq.items()):
        print(f"{_set_str(it):<10} {c:>2} (/{n_tx})  {c/n_tx*100:6.2f}%")

    all_freq = freq.copy()
    k = 2
    while freq:
        prev = sorted(freq)
        candidates = {
            tuple(sorted(set(a) | set(b)))
            for i, a in enumerate(prev) for b in prev[i + 1:]
            if a[:-1] == b[:-1]
        }

        counts = defaultdict(int)
        for tx in map(set, transactions):
            for c in candidates:
                if set(c) <= tx:
                    counts[c] += 1

        print(f"{YELLOW}\nC{k} (Candidates size‑{k}):{RESET}")
        for it in sorted(candidates):
            print(f"{_set_str(it):<10} {counts[it]:>2} (/{n_tx})")

        freq = {k_: v for k_, v in counts.items() if v >= mincnt}

        if freq:
            print(f"{GREEN}\nL{k} (Frequent size‑{k}):{RESET}")
            for it, c in sorted(freq.items()):
                print(f"{_set_str(it):<10} {c:>2} (/{n_tx})  {c/n_tx*100:6.2f}%")

        all_freq.update(freq)
        k += 1

    return all_freq, n_tx

# ─────────────── All‑rule generation (no pruning) ───────────────
def generate_all_rules(freq, n_tx, transactions):
    tx_sets = [set(t) for t in transactions]
    by_level = defaultdict(list)

    for items, cnt in freq.items():
        if len(items) < 2:
            continue
        k = len(items)
        for r in range(1, k):
            for ante in combinations(items, r):
                ante  = tuple(sorted(ante))
                cons  = tuple(sorted(set(items) - set(ante)))
                conf  = cnt / freq[ante]
                suppA = freq[ante] / n_tx
                cntC  = freq.get(cons) or sum(1 for tx in tx_sets if set(cons) <= tx)
                suppC = cntC / n_tx
                lift  = conf / suppC #  => P(A,B) / P(A)*P(B)
                by_level[k].append((ante, cons, conf, lift))
    return by_level

# ─────────────── Strong‑rule generation (with pruning) ───────────────
def generate_strong_rules(all_rules_by_level, min_conf=0.7):
    strong = []
    for rules in all_rules_by_level.values():
        strong.extend([r for r in rules if r[2] >= min_conf])
    return strong

# ────────────────────────── Print ──────────────────────────
def _set_str(items):
    return '{' + ', '.join(items) + '}'

def print_all_rules(all_rules, min_conf):
    for k in sorted(all_rules):
        print(f"{CYAN}\nRules from L{k} (itemset size {k}):{RESET}")
        for ante, cons, conf, lift in sorted(all_rules[k]):
            mark = f"{GREEN}✔{RESET}" if conf >= min_conf else f"{RED}✖{RESET}"
            line = f"{_set_str(ante)} -> {_set_str(cons)}"
            print(f"{mark} {line:<25} conf={conf*100:6.2f}%  lift={lift:6.3f}")

def print_strong_rules(rules):
    print(f"{GREEN}\nStrong rules (meet confidence threshold){RESET}")
    for ante, cons, conf, lift in sorted(rules):
        line = f"{_set_str(ante)} -> {_set_str(cons)}"
        print(f"{line:<25} conf={conf*100:6.2f}%  lift={lift:6.3f}")
    print(f"{GREEN}\nTotal strong rules: {len(rules)}{RESET}")

# ────────────────────────── Example dataset ──────────────────────────
#EDIT DATASET HERE
T = [
    ['A', 'B','C'],
    ['D','E','A'],
    ['A', 'C', 'F'],
    ['D', 'E'],
    ['B', 'F', 'D'],
    ['A', 'C', 'D','E'],
    ['D', 'F', 'A'],
    ['B', 'C'],
    ['A', 'C', 'D','E'],
    ['B', 'F']
]

# ───────────────────────── Run with thresholds ─────────────────────────
#EDIT CONF\SUPP HERE
min_support    = 0.5
min_confidence = 0.7

freq_sets, n_tx     = apriori(T, min_support=min_support)
all_rules_by_level  = generate_all_rules(freq_sets, n_tx, T)
strong_rules        = generate_strong_rules(all_rules_by_level, min_confidence)

print_all_rules(all_rules_by_level, min_confidence)

print_strong_rules(strong_rules)
