import { createRouter, publicQuery } from "./middleware";
import { aiRouter } from "./ai";
import { researchRouter } from "./research";

export const appRouter = createRouter({
  ping: publicQuery.query(() => ({ ok: true, ts: Date.now() })),
  ai: aiRouter,
  research: researchRouter,
});

export type AppRouter = typeof appRouter;
