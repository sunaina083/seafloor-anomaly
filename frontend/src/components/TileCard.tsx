// src/components/TileCard.tsx
//
// A single tile in the review grid: thumbnail, rank, and a score bar whose
// fill/color communicates how anomalous the model thinks this tile is.
// Clicking it opens the detail view (handled by the parent page).

import { Tile, tileImageUrl } from "@/lib/api";

interface TileCardProps {
  tile: Tile;
  onClick: () => void;
}

function scoreColor(score: number): string {
  // Higher score = more anomalous = warmer color, drawing the analyst's eye.
  if (score > 0.7) return "bg-red-400";
  if (score > 0.4) return "bg-amber-400";
  return "bg-teal-400";
}
// Each flag status gets its own icon and color, so an analyst can tell the
// outcome at a glance without opening the tile -- a plain checkmark for every
// status would look identical whether debris was confirmed, ruled out, or
// still unsure, defeating the point of a quick-scan review grid.
const FLAG_BADGES: Record<string, { icon: string; className: string }> = {
  confirmed_debris: { icon: "✓", className: "bg-red-400 text-black" },
  false_positive: { icon: "✕", className: "bg-ocean-600 text-white" },
  unsure: { icon: "?", className: "bg-amber-400 text-black" },
};

export default function TileCard({ tile, onClick }: TileCardProps) {
  return (
    <button
      onClick={onClick}
      className="group relative flex flex-col overflow-hidden rounded-lg border border-ocean-700 bg-ocean-800 transition-transform hover:scale-[1.03] hover:border-teal-400 focus:outline-none focus:ring-2 focus:ring-teal-400"
    >
      <div className="relative aspect-square w-full bg-black">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={tileImageUrl(tile.tile_id)}
          alt={`Sonar tile rank ${tile.rank}`}
          className="h-full w-full object-cover"
        />
        <span className="absolute left-1.5 top-1.5 rounded bg-black/70 px-1.5 py-0.5 text-xs font-mono text-teal-300">
          #{tile.rank}
        </span>
                {tile.flag_status && FLAG_BADGES[tile.flag_status] && (
          <span
            className={`absolute right-1.5 top-1.5 flex h-5 w-5 items-center justify-center rounded-full text-xs font-bold ${FLAG_BADGES[tile.flag_status].className}`}
            title={tile.flag_status.replace("_", " ")}
          >
            {FLAG_BADGES[tile.flag_status].icon}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2 px-2 py-1.5">
        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ocean-700">
          <div
            className={`h-full rounded-full ${scoreColor(tile.ensemble_score)}`}
            style={{ width: `${Math.round(tile.ensemble_score * 100)}%` }}
          />
        </div>
        <span className="font-mono text-xs text-ocean-300">
          {tile.ensemble_score.toFixed(2)}
        </span>
      </div>
    </button>
  );
}