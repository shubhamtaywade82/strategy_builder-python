import { createRouter, publicQuery } from "./middleware";
import { aiRouter } from "./ai";
import { researchRouter } from "./research";
import { marketRouter } from "./market";

export const appRouter = createRouter({
  ping: publicQuery.query(() => ({ ok: true, ts: Date.now() })),
  ai: aiRouter,
  research: researchRouter,
  market: marketRouter,
});

export type AppRouter = typeof appRouter;
