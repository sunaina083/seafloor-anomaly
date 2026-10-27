// src/components/TileDetailModal.tsx
//
// Overlay shown when an analyst opens a tile: the image with a toggleable
// reconstruction-error heatmap, the score breakdown, and the flag buttons.

import { useState } from "react";
import { Tile, tileImageUrl, tileHeatmapUrl, submitFlag, FlagPayload } from "@/lib/api";

interface TileDetailModalProps {
  tile: Tile;
  onClose: () => void;
  onFlagged: (tileId: string, status: string) => void;
}

const FLAG_OPTIONS: { value: FlagPayload["status"]; label: string; color: string }[] = [
  { value: "confirmed_debris", label: "Confirm Debris", color: "bg-red-400 hover:bg-red-300" },
  { value: "false_positive", label: "False Positive", color: "bg-ocean-700 hover:bg-ocean-600" },
  { value: "unsure", label: "Unsure", color: "bg-amber-400 hover:bg-amber-300" },
];

export default function TileDetailModal({ tile, onClose, onFlagged }: TileDetailModalProps) {
  const [submitting, setSubmitting] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(true);

  async function handleFlag(status: FlagPayload["status"]) {
    setSubmitting(true);
    try {
      await submitFlag(tile.tile_id, { status });
      onFlagged(tile.tile_id, status);
    } catch (err) {
      console.error(err);
      alert("Failed to save flag -- check that the backend server is running.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-lg rounded-xl border border-ocean-700 bg-ocean-800 p-5"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-mono text-sm text-teal-300">Tile #{tile.rank}</h2>
          <button onClick={onClose} className="text-ocean-300 hover:text-white">
            ✕
          </button>
        </div>

        {/* Image with the heatmap stacked on top; both are the same square size */}
        <div className="relative mb-2 overflow-hidden rounded-lg border border-ocean-700">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={tileImageUrl(tile.tile_id)}
            alt={`Sonar tile ${tile.tile_id}`}
            className="block w-full"
          />
          {showHeatmap && (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={tileHeatmapUrl(tile.tile_id)}
              alt="Reconstruction error heatmap"
              className="pointer-events-none absolute inset-0 h-full w-full"
            />
          )}
        </div>

        <div className="mb-4 flex items-center justify-between text-xs text-ocean-300">
          <button
            onClick={() => setShowHeatmap((v) => !v)}
            className="rounded-full border border-ocean-600 px-3 py-1 hover:border-teal-400"
          >
            {showHeatmap ? "Hide heatmap" : "Show heatmap"}
          </button>
          <span>Brighter color = harder for the model to rebuild</span>
        </div>

        <div className="mb-4 grid grid-cols-3 gap-2 text-center font-mono text-xs">
          <div>
            <div className="text-ocean-300">Ensemble</div>
            <div className="text-teal-300">{tile.ensemble_score.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-ocean-300">Autoencoder</div>
            <div className="text-teal-300">{tile.ae_score.toFixed(3)}</div>
          </div>
          <div>
            <div className="text-ocean-300">Brightness</div>
            <div className="text-teal-300">{tile.baseline_score.toFixed(3)}</div>
          </div>
        </div>

        <div className="flex gap-2">
          {FLAG_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              disabled={submitting}
              onClick={() => handleFlag(opt.value)}
              className={`flex-1 rounded-lg px-2 py-2 text-xs font-medium text-black transition-colors disabled:opacity-50 ${opt.color}`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {tile.flag_status && (
          <p className="mt-2 text-center text-xs text-ocean-300">
            Current status: <span className="text-teal-300">{tile.flag_status}</span>
          </p>
        )}
      </div>
    </div>
  );
}