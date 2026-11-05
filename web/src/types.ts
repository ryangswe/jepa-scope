// Mirrors the JSON from GET /api/meta (see src/jepa_scope/server.py).

export interface Header {
  model_id: string
  hidden_size: number
  grid: { frames: number; crop: number; patch: number; tubelet: number }
  window: { frames: number; stride: number; context_frames: number; sample_fps: number; tubelet_seconds: number }
  preprocess: { shortest_edge: number | null; crop_size: number; hash: string }
  crop: { x: number; y: number; w: number; h: number }
  source: { path: string; width: number; height: number; duration: number; source_fps: number }
  run: { jepa_scope: string; device: string; dtype: string; seconds: number }
}

export interface Meta {
  header: Header
  timeline: {
    count: number
    grid_h: number
    grid_w: number
    time: number[]
    surprise_mean: (number | null)[]
    surprise_range: [number, number]
  }
}

export interface Data {
  meta: Meta
  pca: Uint8Array // [count, grid_h, grid_w, 3]
  surprise: Uint8Array // [count, grid_h, grid_w], 0..255
}

export type Mode = 'pca' | 'surprise' | 'off'

/** Index of the tubelet showing at `t` seconds, or -1 if `t` is outside the embedded range. */
export function tubeletAt(meta: Meta, t: number): number {
  const times = meta.timeline.time
  const end = times[times.length - 1] + meta.header.window.tubelet_seconds
  if (t < times[0] || t >= end) return -1
  let lo = 0
  let hi = times.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (times[mid] <= t) lo = mid
    else hi = mid - 1
  }
  return lo
}
