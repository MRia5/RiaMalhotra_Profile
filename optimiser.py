"""
============================================================
  Welfare Budget Optimisation Model
  Dynamic Programming vs Greedy vs Random Allocation
============================================================
  Algorithm : 0/1 Knapsack (DP), O(N*W) time
  Baseline  : Greedy (value/cost ratio sort)
  Validation: Paired t-test + Cohen's d effect size
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import random

np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────
#  SAMPLE DATASET — Social Welfare Programs
# ─────────────────────────────────────────────────────────

PROGRAMS = [
    {"name": "Midday Meal Scheme",       "cost": 12, "impact": 85, "category": "Nutrition"},
    {"name": "Rural Health Clinics",     "cost": 18, "impact": 92, "category": "Healthcare"},
    {"name": "Digital Literacy Drive",   "cost":  7, "impact": 64, "category": "Education"},
    {"name": "Women Skill Development",  "cost":  9, "impact": 78, "category": "Employment"},
    {"name": "Sanitation Drive",         "cost": 15, "impact": 88, "category": "Hygiene"},
    {"name": "Affordable Housing Units", "cost": 25, "impact": 95, "category": "Housing"},
    {"name": "Solar Microgrid Villages", "cost": 20, "impact": 82, "category": "Energy"},
    {"name": "Maternal Health Program",  "cost": 11, "impact": 89, "category": "Healthcare"},
    {"name": "Rural Road Connectivity",  "cost": 22, "impact": 74, "category": "Infrastructure"},
    {"name": "Startup Incubation Hubs",  "cost": 14, "impact": 67, "category": "Employment"},
    {"name": "Clean Drinking Water",     "cost": 16, "impact": 91, "category": "Hygiene"},
    {"name": "Old Age Pension Scheme",   "cost": 10, "impact": 80, "category": "Social Security"},
    {"name": "Tribal Education Centers", "cost":  8, "impact": 71, "category": "Education"},
    {"name": "Crop Insurance Scheme",    "cost": 13, "impact": 76, "category": "Agriculture"},
    {"name": "Mental Health Clinics",    "cost":  6, "impact": 60, "category": "Healthcare"},
]

BUDGET = 100  # crore (INR)

df_programs = pd.DataFrame(PROGRAMS)

print("=" * 55)
print("  Welfare Budget Optimisation Model")
print("=" * 55)
print(f"\n  Budget : ₹{BUDGET} Cr")
print(f"  Programs: {len(PROGRAMS)}\n")
print(df_programs[['name', 'cost', 'impact', 'category']].to_string(index=False))


# ─────────────────────────────────────────────────────────
#  ALGORITHM 1 — Dynamic Programming (0/1 Knapsack)
# ─────────────────────────────────────────────────────────

def knapsack_dp(costs, values, capacity):
    """
    Standard 0/1 Knapsack with DP.
    Returns (max_value, selected_indices).
    Space-optimised to 1D DP array.
    """
    n  = len(costs)
    dp = [0] * (capacity + 1)

    # Track selected items via backtracking table
    keep = [[False] * (capacity + 1) for _ in range(n)]

    for i in range(n):
        # Traverse backwards to avoid using item i twice
        for w in range(capacity, costs[i] - 1, -1):
            if dp[w - costs[i]] + values[i] > dp[w]:
                dp[w] = dp[w - costs[i]] + values[i]
                keep[i][w] = True

    # Backtrack to find selected items
    selected = []
    w = capacity
    for i in range(n - 1, -1, -1):
        if keep[i][w]:
            selected.append(i)
            w -= costs[i]

    return dp[capacity], selected


# ─────────────────────────────────────────────────────────
#  ALGORITHM 2 — Greedy (value/cost ratio)
# ─────────────────────────────────────────────────────────

def knapsack_greedy(costs, values, capacity):
    """
    Greedy: sort by value/cost descending, pick while budget allows.
    Provably suboptimal for 0/1 Knapsack.
    """
    ratios   = [(v / c, i) for i, (c, v) in enumerate(zip(costs, values))]
    ratios.sort(reverse=True)

    selected = []
    remaining = capacity
    total_val = 0

    for _, i in ratios:
        if costs[i] <= remaining:
            selected.append(i)
            remaining -= costs[i]
            total_val += values[i]

    return total_val, selected


# ─────────────────────────────────────────────────────────
#  RUN ON SAMPLE DATA
# ─────────────────────────────────────────────────────────

costs  = df_programs['cost'].tolist()
values = df_programs['impact'].tolist()

dp_val, dp_sel     = knapsack_dp(costs, values, BUDGET)
greedy_val, g_sel  = knapsack_greedy(costs, values, BUDGET)

dp_cost     = sum(costs[i] for i in dp_sel)
greedy_cost = sum(costs[i] for i in g_sel)

print("\n\n" + "=" * 55)
print("  RESULTS — Single Budget Allocation")
print("=" * 55)

print(f"\n  DP Optimal:")
print(f"    Programs  : {[df_programs.iloc[i]['name'] for i in dp_sel]}")
print(f"    Cost Used : ₹{dp_cost} Cr  ({dp_cost/BUDGET*100:.1f}% of budget)")
print(f"    Welfare   : {dp_val}")

print(f"\n  Greedy:")
print(f"    Programs  : {[df_programs.iloc[i]['name'] for i in g_sel]}")
print(f"    Cost Used : ₹{greedy_cost} Cr  ({greedy_cost/BUDGET*100:.1f}% of budget)")
print(f"    Welfare   : {greedy_val}")

print(f"\n  DP outperforms greedy by : {dp_val - greedy_val} points "
      f"({(dp_val - greedy_val)/greedy_val*100:.1f}%)")


# ─────────────────────────────────────────────────────────
#  STATISTICAL VALIDATION — 1000 random instances
# ─────────────────────────────────────────────────────────

N_TRIALS = 1000
dp_scores     = []
greedy_scores = []
random_scores = []

for _ in range(N_TRIALS):
    n = random.randint(8, 20)
    c = [random.randint(3, 30) for _ in range(n)]
    v = [random.randint(20, 100) for _ in range(n)]
    b = random.randint(40, 120)

    dp_v, _  = knapsack_dp(c, v, b)
    gr_v, _  = knapsack_greedy(c, v, b)

    # Random baseline
    rand_sel, rand_budget, rand_val = [], b, 0
    idxs = list(range(n))
    random.shuffle(idxs)
    for i in idxs:
        if c[i] <= rand_budget:
            rand_sel.append(i)
            rand_budget -= c[i]
            rand_val    += v[i]

    dp_scores.append(dp_v)
    greedy_scores.append(gr_v)
    random_scores.append(rand_val)

dp_arr     = np.array(dp_scores)
greedy_arr = np.array(greedy_scores)
rand_arr   = np.array(random_scores)

t_stat, p_val = stats.ttest_rel(dp_arr, greedy_arr)
effect_d = (dp_arr.mean() - greedy_arr.mean()) / np.std(dp_arr - greedy_arr)

print("\n\n" + "=" * 55)
print("  STATISTICAL VALIDATION (N=1000 random instances)")
print("=" * 55)
print(f"  DP mean welfare     : {dp_arr.mean():.1f} ± {dp_arr.std():.1f}")
print(f"  Greedy mean welfare : {greedy_arr.mean():.1f} ± {greedy_arr.std():.1f}")
print(f"  Random mean welfare : {rand_arr.mean():.1f} ± {rand_arr.std():.1f}")
print(f"  Paired t-test       : t={t_stat:.2f}, p={p_val:.4e}")
print(f"  Cohen's d           : {effect_d:.2f}")
print(f"  DP never worse?     : {(dp_arr >= greedy_arr).all()}")


# ─────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Welfare Budget Optimisation — DP vs Greedy vs Random", fontweight='bold')

# Chart 1: Welfare score comparison (single instance)
methods = ['DP (Optimal)', 'Greedy', 'Random']
scores  = [dp_val, greedy_val, int(np.mean(random_scores))]
colours = ['#1a7a3c', '#ffc107', '#dc3545']
axes[0].bar(methods, scores, color=colours, edgecolor='white')
axes[0].set_title("Welfare Score (Sample Instance)")
axes[0].set_ylabel("Total Impact Score")
for bar, score in zip(axes[0].patches, scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 str(score), ha='center', fontweight='bold')

# Chart 2: Distribution across 1000 trials
axes[1].hist(dp_arr,     bins=40, alpha=0.7, color='#1a7a3c', label='DP')
axes[1].hist(greedy_arr, bins=40, alpha=0.7, color='#ffc107', label='Greedy')
axes[1].hist(rand_arr,   bins=40, alpha=0.5, color='#dc3545', label='Random')
axes[1].set_title("Welfare Score Distribution (N=1000)")
axes[1].set_xlabel("Welfare Score")
axes[1].set_ylabel("Frequency")
axes[1].legend()

# Chart 3: DP gain over greedy per trial
diff = dp_arr - greedy_arr
axes[2].hist(diff, bins=30, color='#4c72b0', edgecolor='white')
axes[2].axvline(diff.mean(), color='red', linestyle='--',
                label=f'Mean gain = {diff.mean():.1f}')
axes[2].set_title("DP Gain Over Greedy (per trial)")
axes[2].set_xlabel("DP score − Greedy score")
axes[2].set_ylabel("Frequency")
axes[2].legend()

plt.tight_layout()
plt.savefig('welfare_optimisation_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📊 Chart saved as welfare_optimisation_results.png")

# ─────────────────────────────────────────────────────────
#  SELECTED PROGRAM TABLE (DP solution)
# ─────────────────────────────────────────────────────────

print("\n\n" + "=" * 55)
print("  DP OPTIMAL ALLOCATION — Program Breakdown")
print("=" * 55)
selected_df = df_programs.iloc[dp_sel].copy()
selected_df['Efficiency (impact/cost)'] = (selected_df['impact'] / selected_df['cost']).round(2)
print(selected_df[['name', 'cost', 'impact', 'category', 'Efficiency (impact/cost)']].to_string(index=False))
print(f"\n  Total Cost   : ₹{selected_df['cost'].sum()} Cr")
print(f"  Total Impact : {selected_df['impact'].sum()}")
print("\n  ⚠️  This is a demonstration model. Real welfare allocation")
print("     requires qualitative, equity, and political considerations.")
