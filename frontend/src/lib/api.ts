// src/lib/api.ts
//
// Thin client for the FastAPI backend. Keeping all fetch calls here (rather
// than scattered across components) means if the API shape changes, we only
// update one file.

const API_BASE = "http://localhost:8000";

export interface Tile {
  tile_id: string;
  rank: number;
  ensemble_score: number;
  ae_score: number;
  baseline_score: number;
  flag_status: string | null;
}

export interface FlagPayload {
  status: "confirmed_debris" | "false_positive" | "unsure";
  note?: string;
}

export async function fetchTiles(limit = 50, offset = 0): Promise<Tile[]> {
  const res = await fetch(`${API_BASE}/tiles?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error(`Failed to fetch tiles: ${res.status}`);
  return res.json();
}

export function tileImageUrl(tileId: string): string {
  return `${API_BASE}/tiles/${tileId}/image`;
}
export function tileHeatmapUrl(tileId: string): string {
  return `${API_BASE}/tiles/${tileId}/heatmap`;
}
export async function submitFlag(tileId: string, payload: FlagPayload): Promise<void> {
  const res = await fetch(`${API_BASE}/tiles/${tileId}/flag`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(`Failed to submit flag: ${res.status}`);
}