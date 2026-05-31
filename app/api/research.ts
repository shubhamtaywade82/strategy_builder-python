import { z } from "zod";
import { createRouter, publicQuery } from "./middleware";
import { spawn } from "child_process";
import { readFile, unlink } from "fs/promises";
import { existsSync } from "fs";
import path from "path";
import os from "os";
import crypto from "crypto";

const ENGINE_SCRIPT = path.resolve(
  import.meta.dirname ?? path.dirname(new URL(import.meta.url).pathname),
  "../public/engine/main.py"
);

const PYTHON_BIN = process.env.PYTHON_BIN ?? "python3";

function runPython(
  symbol: string,
  days: number,
  leverage: number,
  horizon: number,
  outputPath: string,
  timeoutMs = 300_000
): Promise<string> {
  return new Promise((resolve, reject) => {
    const args = [
      ENGINE_SCRIPT,
      "--symbol", symbol,
      "--days", String(days),
      "--leverage", String(leverage),
      "--horizon", String(horizon),
      "--output", outputPath,
    ];

    const proc = spawn(PYTHON_BIN, args, {
      stdio: ["ignore", "pipe", "pipe"],
      env: { ...process.env },
    });

    let stdout = "";
    let stderr = "";
    proc.stdout.on("data", (d: Buffer) => { stdout += d.toString(); });
    proc.stderr.on("data", (d: Buffer) => { stderr += d.toString(); });

    const timer = setTimeout(() => {
      proc.kill("SIGKILL");
      reject(new Error(`Research timed out after ${timeoutMs / 1000}s`));
    }, timeoutMs);

    proc.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) {
        resolve(stdout);
      } else {
        reject(new Error(`Python engine exited with code ${code}.\n${stderr.slice(-2000)}`));
      }
    });

    proc.on("error", (err) => {
      clearTimeout(timer);
      reject(new Error(`Failed to start Python: ${err.message}`));
    });
  });
}

// Map Python engine JSON output → TypeScript ResearchResult shape
function mapResult(raw: any, symbol: string, rrs: string[], leverage: number) {
  const allStrategies: any[] = [];
  let walkForward: any = { folds: 0, avgTestAuc: 0.5, minTestAuc: 0.5, stability: 0, degradation: 0, isValid: false, foldResults: [] };
  let topFeatures: any[] = [];
  let labels = { longWins: 0, shortWins: 0, noTrade: 0, total: 0 };

  for (const rr of rrs) {
    const rrData = raw?.results?.[rr];
    if (!rrData) continue;

    for (const side of ["long", "short"] as const) {
      const sideData = rrData[side];
      if (!sideData || sideData.error) continue;

      for (const strat of sideData.strategies ?? []) {
        const m = strat.metrics ?? {};
        allStrategies.push({
          id: `${rr}_${strat.name ?? side}`,
          name: strat.name ?? `${rr} ${side}`,
          description: strat.description ?? "",
          side: strat.side ?? side,
          conditions: (strat.conditions ?? []).map((c: any) => ({
            feature: c.feature,
            operator: c.operator,
            threshold: c.threshold,
            importance: c.importance ?? 0,
          })),
          metrics: {
            tradeCount: m.trade_count ?? 0,
            winCount: m.win_count ?? 0,
            lossCount: m.loss_count ?? 0,
            winRate: m.win_rate ?? 0,
            profitFactor: m.profit_factor ?? 0,
            expectancy: m.expectancy ?? 0,
            netPnl: m.net_pnl ?? 0,
            avgWin: m.avg_win ?? 0,
            avgLoss: m.avg_loss ?? 0,
            maxDrawdown: m.max_drawdown ?? 0,
            sharpe: m.sharpe ?? 0,
            avgBarsHeld: m.avg_bars_held ?? 0,
            targetHitRate: m.target_hit_rate ?? 0,
            stopHitRate: m.stop_hit_rate ?? 0,
          },
          isViable: strat.is_viable ?? false,
        });
      }

      // Walk-forward from first valid RR/side
      if (sideData.walk_forward && walkForward.folds === 0) {
        const wf = sideData.walk_forward;
        walkForward = {
          folds: wf.folds ?? 0,
          avgTestAuc: wf.avg_test_auc ?? 0.5,
          minTestAuc: wf.min_test_auc ?? 0.5,
          stability: wf.stability ?? 0,
          degradation: wf.degradation ?? 0,
          isValid: wf.is_valid ?? false,
          foldResults: (wf.fold_results ?? []).map((f: any) => ({
            fold: f.fold,
            trainAuc: f.train_auc ?? 0.5,
            testAuc: f.test_auc ?? 0.5,
            testPrecision: f.test_precision ?? 0,
            testRecall: f.test_recall ?? 0,
            nTrain: f.n_train ?? 0,
            nTest: f.n_test ?? 0,
          })),
        };
      }

      // Top features from first valid RR/side
      if (sideData.top_features?.length && topFeatures.length === 0) {
        topFeatures = sideData.top_features.map((f: any) => ({
          feature: f.feature ?? f.name ?? "",
          direction: (f.direction as "high" | "low") ?? "high",
          threshold: f.threshold ?? f.median_val ?? 0,
          importance: f.importance ?? 0,
          winRateAbove: f.win_rate_above ?? 0,
          winRateBelow: f.win_rate_below ?? 0,
          shapValue: f.shap_value ?? 0,
        }));
      }

      // Label stats
      const ls = sideData.label_stats;
      if (ls && labels.total === 0) {
        labels = {
          total: ls.total ?? 0,
          longWins: ls.long_wins ?? 0,
          shortWins: ls.short_wins ?? 0,
          noTrade: ls.no_trade ?? 0,
        };
      }
    }
  }

  const primaryRR = rrs[0] ?? "2:1";
  const rrCfgs: Record<string, { upPct: number; dnPct: number }> = {
    "3:1": { upPct: 0.015, dnPct: 0.005 },
    "2:1": { upPct: 0.010, dnPct: 0.005 },
    "1:1": { upPct: 0.010, dnPct: 0.010 },
    "1:2": { upPct: 0.005, dnPct: 0.010 },
    "1:3": { upPct: 0.005, dnPct: 0.015 },
  };
  const rrCfg = rrCfgs[primaryRR] ?? { upPct: 0.01, dnPct: 0.005 };

  return {
    symbol,
    config: {
      symbol,
      rr: primaryRR,
      upPct: rrCfg.upPct,
      dnPct: rrCfg.dnPct,
      leverage,
      horizon: 120,
      side: "both" as const,
    },
    strategies: allStrategies,
    walkForward,
    topFeatures,
    labels,
  };
}

export const researchRouter = createRouter({
  run: publicQuery
    .input(
      z.object({
        symbol: z.string().regex(/^[A-Z0-9]{3,12}$/).default("SOLUSDT"),
        days: z.number().int().min(7).max(365).default(60),
        leverage: z.number().min(1).max(125).default(10),
        horizon: z.number().int().min(30).max(480).default(120),
        rrs: z.array(z.string()).min(1).default(["2:1"]),
      })
    )
    .mutation(async ({ input }) => {
      const tmpFile = path.join(os.tmpdir(), `research_${crypto.randomBytes(8).toString("hex")}.json`);

      try {
        await runPython(input.symbol, input.days, input.leverage, input.horizon, tmpFile);

        if (!existsSync(tmpFile)) {
          throw new Error("Python engine did not produce output file");
        }

        const raw = JSON.parse(await readFile(tmpFile, "utf-8"));
        return mapResult(raw, input.symbol, input.rrs, input.leverage);
      } finally {
        if (existsSync(tmpFile)) {
          await unlink(tmpFile).catch(() => {});
        }
      }
    }),
});
