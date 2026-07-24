"""
run_remaining_experiments.py
============================================================
Runs only Exp 8, 9, 10, and 12 from run_comprehensive_experiments.py.
Exp 1-7 completed successfully in the previous run.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Import everything from the main file
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Re-export main experiment functions
from run_comprehensive_experiments import (
    run_exp8_tail_risk_cvar,
    run_exp9_rademacher_gap,
    run_exp10_dow30_real_backtest,
    run_exp12_ko_param_noise,
)
import time

if __name__ == "__main__":
    t_start = time.perf_counter()
    print("=" * 70)
    print(" REMAINING EXPERIMENTS: Exp 8, 9, 10, 12")
    print("=" * 70)

    run_exp8_tail_risk_cvar()
    run_exp9_rademacher_gap()
    run_exp10_dow30_real_backtest()
    run_exp12_ko_param_noise()

    elapsed = (time.perf_counter() - t_start) / 60.0
    print(f"\n{'='*70}")
    print(f" REMAINING EXPERIMENTS COMPLETED in {elapsed:.1f} minutes!")
    print(f"{'='*70}")
