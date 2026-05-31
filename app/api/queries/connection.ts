import { drizzle } from "drizzle-orm/better-sqlite3";
import Database from "better-sqlite3";
import * as schema from "@db/schema";

const fullSchema = { ...schema };

let instance: ReturnType<typeof drizzle<typeof fullSchema>>;

export function getDb() {
  if (!instance) {
    const sqlite = new Database("sqlite.db");
    instance = drizzle(sqlite, {
      schema: fullSchema,
    });
  }
  return instance;
}
