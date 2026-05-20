import sys, os
root = os.path.abspath(os.path.dirname(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

import pandas as pd
import numpy as np
import random
import math
import crawler
from backtest import backtest_squeeze_strategy, has_any_ad_signal


# ==========================================
# 基因演算法：自動搜尋 A~C 的最佳參數
# ==========================================
def _evaluate_params_on_universe(target_list, params, sample_limit=None):
    """以目標清單評估一組參數的績效回報（回傳 fitness 與統計）。

    params: dict 含 keys: continuous_weeks, min_growth, last_week_threshold, pop_decline_threshold
    sample_limit: 若指定，只取前 N 檔股票來加速評估。
    """
    all_trades = []
    total_signals = 0
    returns = []

    target_iter = target_list if sample_limit is None else target_list[:sample_limit]

    for sid in target_iter:
        df = crawler.get_individual_stock_data(sid)
        if df is None or df.empty:
            continue

        # 先用預篩加速
        if not has_any_ad_signal(df,
                                 continuous_weeks=params['continuous_weeks'],
                                 min_growth=params['min_growth'],
                                 last_week_threshold=params['last_week_threshold'],
                                 pop_decline_threshold=params['pop_decline_threshold']):
            continue

        trades = backtest_squeeze_strategy(df,
                                          continuous_weeks=params['continuous_weeks'],
                                          min_growth=params['min_growth'],
                                          last_week_threshold=params['last_week_threshold'],
                                          pop_decline_threshold=params['pop_decline_threshold'])

        if trades:
            all_trades.extend(trades)

    # 若沒有任何 trade，我們回傳 fitness=0（代表該參數組沒有產生可評分訊號），
    # 但不使用 -1 作為特殊值，避免 GA 排序被極端值主導。
    if not all_trades:
        return {'fitness': 0.0, 'n_signals': 0, 'avg_return': 0.0, 'win_rate': 0.0}

    trades_df = pd.DataFrame(all_trades).dropna(subset=['週報酬%'])
    if trades_df.empty:
        return {'fitness': 0.0, 'n_signals': 0, 'avg_return': 0.0, 'win_rate': 0.0}

    n = len(trades_df)
    avg_ret = trades_df['週報酬%'].mean()
    win_rate = (trades_df['週報酬%'] > 0).mean() * 100

    # fitness 定義：以勝率、平均報酬、訊號數綜合評分（可依需求調整）
    fitness = (win_rate / 100.0) * avg_ret * math.sqrt(n)

    return {'fitness': fitness, 'n_signals': n, 'avg_return': avg_ret, 'win_rate': win_rate}


def run_genetic_algorithm(target_list, generations=10, population_size=100, sample_limit=None,
                          retain_top=0.2, mutate_chance=0.001, random_seed=None):
    """執行簡單基因演算法來搜尋 `continuous_weeks`, `min_growth`, `last_week_threshold`, `pop_decline_threshold`。

    - target_list: 股票代號清單
    - sample_limit: 每次評估只使用前 N 檔股票以加速（可設為 None 使用全部）
    """
    if random_seed is not None:
        random.seed(random_seed)

    # 參數空間定義
    param_bounds = {
        'continuous_weeks': (3, 8),            # 整數周數
        'min_growth': (0.0, 1.0),              # % for average per person growth (0% - 1%)
        'last_week_threshold': (0.0, 10.0),    # % 最後一週門檻
        'pop_decline_threshold': (0.1, 5.0)    # % 總股東人數下降
    }

    def random_individual():
        return {
            'continuous_weeks': random.randint(param_bounds['continuous_weeks'][0], param_bounds['continuous_weeks'][1]),
            'min_growth': round(random.uniform(*param_bounds['min_growth']), 4),
            'last_week_threshold': round(random.uniform(*param_bounds['last_week_threshold']), 3),
            'pop_decline_threshold': round(random.uniform(*param_bounds['pop_decline_threshold']), 3)
        }

    # 初始化族群
    population = [random_individual() for _ in range(population_size)]

    scored_pop = []
    for gen in range(generations):
        print(f"GA generation {gen+1}/{generations} - evaluating {len(population)} individuals...")

        scored_pop = []
        for individual in population:
            stats = _evaluate_params_on_universe(target_list, individual, sample_limit=sample_limit)
            scored_pop.append((individual, stats['fitness'], stats))

        # 依 fitness 排序
        scored_pop.sort(key=lambda x: x[1], reverse=True)

        best_individual, best_fitness, best_stats = scored_pop[0]
        print(f"  Best so far: fitness={best_fitness:.4f} signals={best_stats['n_signals']} avg_ret={best_stats['avg_return']:.3f} win_rate={best_stats['win_rate']:.1f}% params={best_individual}")

        # 選擇保留
        retain_length = max(1, int(retain_top * len(scored_pop)))
        next_gen = [ind for ind, _, _ in scored_pop[:retain_length]]

        # 以 tournament selection 產生新個體
        while len(next_gen) < population_size:
            # 選父母
            def tournament():
                competitors = random.sample(scored_pop, min(3, len(scored_pop)))
                competitors.sort(key=lambda x: x[1], reverse=True)
                return competitors[0][0]

            parent1 = tournament()
            parent2 = tournament()

            # crossover (uniform)
            child = {}
            for k in parent1.keys():
                child[k] = parent1[k] if random.random() < 0.5 else parent2[k]

            # mutation
            if random.random() < mutate_chance:
                key_to_mutate = random.choice(list(param_bounds.keys()))
                if key_to_mutate == 'continuous_weeks':
                    child['continuous_weeks'] = random.randint(param_bounds['continuous_weeks'][0], param_bounds['continuous_weeks'][1])
                else:
                    lo, hi = param_bounds[key_to_mutate]
                    child[key_to_mutate] = round(random.uniform(lo, hi), 4 if key_to_mutate=='min_growth' else 3)

            # 保證型別
            child['continuous_weeks'] = int(child['continuous_weeks'])
            next_gen.append(child)

        population = next_gen

    # 最終排序並回傳最好的幾個
    final_scores = []
    for individual in population:
        stats = _evaluate_params_on_universe(target_list, individual, sample_limit=sample_limit)
        final_scores.append((individual, stats['fitness'], stats))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    return final_scores


def main():
    stock_list = crawler.get_stock_ids(crawler.list_url)
    print('Total stocks available:', len(stock_list))
    target_list = stock_list[:50]

    results = run_genetic_algorithm(target_list,
                                   generations=30,  # 演化世代數
                                   population_size=100, # 每代個體數
                                   sample_limit=None,   # 每次評估使用前 N 檔股票（None 表示全部）
                                   retain_top=0.2,  # 每代保留前 20% 的個體
                                   mutate_chance=0.01,  # 1% 的突變機率
                                   random_seed=None)    # 固定隨機種子以重現結果（可設為 None 以獲得不同結果）

    print('\nTop results:')
    for idx, (ind, fit, stats) in enumerate(results[:5]):
        print(idx+1, 'fitness=', fit, 'signals=', stats['n_signals'], 'avg_ret=', stats['avg_return'], 'win_rate=', stats['win_rate'], 'params=', ind)


if __name__ == '__main__':
    main()
