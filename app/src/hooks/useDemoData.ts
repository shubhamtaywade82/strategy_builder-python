import type { ResearchResult, RRRatio } from '../types';

export function generateDemoResults(symbol: string, rrs: RRRatio[]): ResearchResult {
  const rrResults: any = {};

  for (const rr of rrs) {
    const winRate = rr === '3:1' ? 0.38 : rr === '2:1' ? 0.42 : rr === '1:1' ? 0.51 : rr === '1:2' ? 0.58 : 0.62;
    const tradeCount = Math.floor(200 + Math.random() * 300);
    const pf = 1.1 + Math.random() * 0.8;

    rrResults[rr] = {
      strategies: [
        {
          id: `${rr}_long_0.55`,
          name: `${rr} Long Trend-Follow`,
          description: `4H_trend>=1 AND ltf_rvol<0.002 AND vol_ratio>1.5`,
          side: 'long',
          conditions: [
            { feature: '4h_trend', operator: '>=', threshold: 1, importance: 0.15 },
            { feature: 'ltf_rvol_20', operator: '<', threshold: 0.002, importance: 0.12 },
            { feature: 'ltf_vol_ratio', operator: '>', threshold: 1.5, importance: 0.10 },
            { feature: '1h_ema9_ratio', operator: '>', threshold: 0.001, importance: 0.08 },
          ],
          metrics: {
            tradeCount, winCount: Math.floor(tradeCount * winRate),
            lossCount: tradeCount - Math.floor(tradeCount * winRate),
            winRate, profitFactor: pf,
            expectancy: (winRate * 0.01 - (1 - winRate) * 0.005) * 10 - 0.009,
            netPnl: tradeCount * ((winRate * 0.01 - (1 - winRate) * 0.005) * 10 - 0.009),
            avgWin: 0.0095, avgLoss: -0.0059,
            maxDrawdown: -0.15, sharpe: 1.2 + Math.random(),
            avgBarsHeld: 45, targetHitRate: winRate, stopHitRate: 1 - winRate,
          },
          isViable: pf > 1.1 && tradeCount > 30,
        },
        {
          id: `${rr}_short_0.60`,
          name: `${rr} Short Mean-Revert`,
          description: `1H_overbought AND ltf_bar_pos>0.8 AND vol_spike`,
          side: 'short',
          conditions: [
            { feature: '1h_bar_pos', operator: '>', threshold: 0.8, importance: 0.14 },
            { feature: 'ltf_vol_ratio', operator: '>', threshold: 2.0, importance: 0.11 },
            { feature: 'ltf_hh', operator: '==', threshold: 1, importance: 0.09 },
            { feature: '15m_rvol', operator: '>', threshold: 0.003, importance: 0.07 },
          ],
          metrics: {
            tradeCount: Math.floor(tradeCount * 0.7),
            winCount: Math.floor(tradeCount * 0.7 * (winRate - 0.05)),
            lossCount: Math.floor(tradeCount * 0.7 * (1 - winRate + 0.05)),
            winRate: winRate - 0.05,
            profitFactor: pf * 0.9,
            expectancy: ((winRate - 0.05) * 0.01 - (1.05 - winRate) * 0.005) * 10 - 0.009,
            netPnl: Math.floor(tradeCount * 0.7) * (((winRate - 0.05) * 0.01 - (1.05 - winRate) * 0.005) * 10 - 0.009),
            avgWin: 0.0092, avgLoss: -0.0061,
            maxDrawdown: -0.18, sharpe: 0.9 + Math.random(),
            avgBarsHeld: 38, targetHitRate: winRate - 0.05, stopHitRate: 1.05 - winRate,
          },
          isViable: pf * 0.9 > 1.0 && Math.floor(tradeCount * 0.7) > 20,
        },
      ],
      walk_forward: {
        folds: 5,
        avg_test_auc: 0.52 + Math.random() * 0.08,
        min_test_auc: 0.48 + Math.random() * 0.06,
        stability: 0.6 + Math.random() * 0.3,
        degradation: -0.05 + Math.random() * 0.04,
        isValid: Math.random() > 0.3,
        fold_results: Array.from({ length: 5 }, (_, i) => ({
          fold: i + 1,
          train_auc: 0.55 + Math.random() * 0.15,
          test_auc: 0.50 + Math.random() * 0.12,
          test_precision: 0.40 + Math.random() * 0.20,
          test_recall: 0.35 + Math.random() * 0.25,
          n_train: 2000 + i * 500,
          n_test: 500,
        })),
      },
      shuffle_test: {
        real_auc: 0.54 + Math.random() * 0.10,
        shuffled_auc_mean: 0.50,
        shuffled_auc_std: 0.02,
        p_value: 0.01 + Math.random() * 0.15,
        is_significant: Math.random() > 0.4,
      },
      top_features: [
        { feature: '4h_trend', direction: 'high', threshold: 0.5, importance: 0.15, winRateAbove: 0.58, winRateBelow: 0.42, shapValue: 0.08 },
        { feature: 'ltf_rvol_20', direction: 'low', threshold: 0.002, importance: 0.12, winRateAbove: 0.38, winRateBelow: 0.55, shapValue: -0.06 },
        { feature: 'ltf_vol_ratio', direction: 'high', threshold: 1.5, importance: 0.10, winRateAbove: 0.56, winRateBelow: 0.41, shapValue: 0.05 },
        { feature: '1h_ema9_ratio', direction: 'high', threshold: 0.001, importance: 0.08, winRateAbove: 0.54, winRateBelow: 0.44, shapValue: 0.04 },
        { feature: 'ltf_bar_pos', direction: 'low', threshold: 0.3, importance: 0.07, winRateAbove: 0.40, winRateBelow: 0.53, shapValue: -0.04 },
        { feature: '15m_trend', direction: 'high', threshold: 0.5, importance: 0.06, winRateAbove: 0.52, winRateBelow: 0.43, shapValue: 0.03 },
        { feature: 'ltf_mom_10', direction: 'high', threshold: 0.005, importance: 0.05, winRateAbove: 0.51, winRateBelow: 0.44, shapValue: 0.03 },
        { feature: '1h_rvol', direction: 'low', threshold: 0.005, importance: 0.05, winRateAbove: 0.39, winRateBelow: 0.52, shapValue: -0.02 },
      ],
      label_stats: {
        total: 5000,
        longWins: Math.floor(5000 * (0.35 + Math.random() * 0.1)),
        shortWins: Math.floor(5000 * (0.35 + Math.random() * 0.1)),
        noTrade: Math.floor(5000 * 0.15),
      },
    };
  }

  return {
    symbol,
    config: { symbol, rr: rrs[0], upPct: 0.01, dnPct: 0.005, leverage: 10, horizon: 120, side: 'both' },
    strategies: Object.values(rrResults).flatMap((r: any) => r.strategies),
    walkForward: rrResults[rrs[0]]?.walk_forward || { folds: 0, isValid: false, foldResults: [] },
    topFeatures: rrResults[rrs[0]]?.top_features || [],
    labels: rrResults[rrs[0]]?.label_stats || { total: 0, longWins: 0, shortWins: 0, noTrade: 0 },
  };
}
