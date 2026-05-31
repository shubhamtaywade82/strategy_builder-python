import {
  sqliteTable,
  text,
  integer,
  real,
} from "drizzle-orm/sqlite-core";

// Chat messages with AI assistant
export const chatMessages = sqliteTable("chat_messages", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  sessionId: text("session_id").notNull(),
  role: text("role").notNull(), // user, assistant, system
  content: text("content").notNull(),
  model: text("model").default("qwen3.5:4b"),
  createdAt: integer("created_at", { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
});

// Research sessions (saved strategy research runs)
export const researchSessions = sqliteTable("research_sessions", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  sessionId: text("session_id").notNull().unique(),
  symbol: text("symbol").notNull(),
  rrConfig: text("rr_config").notNull(),
  leverage: integer("leverage").notNull().default(10),
  days: integer("days").notNull().default(60),
  status: text("status").notNull().default("pending"), // pending, running, completed, failed
  resultJson: text("result_json"),
  bestStrategy: text("best_strategy"),
  winRate: real("win_rate"),
  profitFactor: real("profit_factor"),
  expectancy: real("expectancy"),
  createdAt: integer("created_at", { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
  updatedAt: integer("updated_at", { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
});

// AI-generated strategy insights
export const strategyInsights = sqliteTable("strategy_insights", {
  id: integer("id").primaryKey({ autoIncrement: true }),
  sessionId: text("session_id").notNull(),
  strategyName: text("strategy_name").notNull(),
  insightType: text("insight_type").notNull(), // analysis, optimization, risk_assessment
  content: text("content").notNull(),
  model: text("model").default("qwen3.5:4b"),
  createdAt: integer("created_at", { mode: 'timestamp' }).notNull().$defaultFn(() => new Date()),
});
