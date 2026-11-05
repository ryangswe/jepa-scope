import { useCallback, useEffect, useRef, useState } from 'react'
import { Overlay } from './overlay'
import { Timeline } from './Timeline'
import { tubeletAt, type Data, type Meta, type Mode } from './types'

async function load(): Promise<Data> {
  const [meta, pca, surprise] = await Promise.all([
    fetch('/api/meta').then((r) => r.json() as Promise<Meta>),
    fetch('/api/pca').then((r) => r.arrayBuffer()),
    fetch('/api/surprise').then((r) => r.arrayBuffer()),
  ])
  const { count, grid_h, grid_w } = meta.timeline
  const pcaBytes = new Uint8Array(pca)
  const surpriseBytes = new Uint8Array(surprise)
  if (pcaBytes.length !== count * grid_h * grid_w * 3 || surpriseBytes.length !== count * grid_h * grid_w) {
    throw new Error('array sizes do not match /api/meta; the server and viewer are out of sync')
  }
  return { meta, pca: pcaBytes, surprise: surpriseBytes }
}

export default function App() {
  const [data, setData] = useState<Data | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    load().then(setData, (e) => setError(String(e)))
  }, [])
  if (error) return <p className="message">Could not load: {error}</p>
  if (!data) return <p className="message">Loading latents…</p>
  return <Viewer data={data} />
}

function Viewer({ data }: { data: Data }) {
  const { meta } = data
  const h = meta.header
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const overlayRef = useRef<Overlay | null>(null)
  const [mode, setMode] = useState<Mode>('pca')
  const [opacity, setOpacity] = useState(0.6)
  const [smooth, setSmooth] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(h.source.duration)
  const [glError, setGlError] = useState<string | null>(null)
  const current = tubeletAt(meta, time)

  useEffect(() => {
    if (!canvasRef.current) return
    try {
      const { count, grid_w, grid_h } = meta.timeline
      overlayRef.current = new Overlay(canvasRef.current, count, grid_w, grid_h, data.pca, data.surprise)
    } catch (e) {
      setGlError(String(e))
    }
  }, [data, meta])

  // Follow the decoded frame, not a timer, so the overlay never drifts from the picture.
  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    let handle = 0
    const tick: VideoFrameRequestCallback = (_now, frame) => {
      setTime(frame.mediaTime)
      handle = v.requestVideoFrameCallback(tick)
    }
    handle = v.requestVideoFrameCallback(tick)
    const onSeek = () => setTime(v.currentTime)
    v.addEventListener('seeked', onSeek)
    return () => {
      v.cancelVideoFrameCallback(handle)
      v.removeEventListener('seeked', onSeek)
    }
  }, [])

  useEffect(() => {
    const draw = () =>
      overlayRef.current?.draw({
        layer: mode === 'off' ? -1 : current,
        mode: mode === 'surprise' ? 'surprise' : 'pca',
        opacity,
        smooth,
      })
    draw()
    window.addEventListener('resize', draw)
    return () => window.removeEventListener('resize', draw)
  }, [current, mode, opacity, smooth])

  const seek = useCallback((t: number) => {
    const v = videoRef.current
    if (v) v.currentTime = Math.max(0, Math.min(t, v.duration || t))
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement) return
      const v = videoRef.current
      if (!v) return
      if (e.key === ' ') {
        e.preventDefault()
        if (v.paused) void v.play()
        else v.pause()
      } else if (e.key === '1') setMode('pca')
      else if (e.key === '2') setMode('surprise')
      else if (e.key === '3') setMode('off')
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        e.preventDefault()
        v.pause()
        seek(v.currentTime + (e.key === 'ArrowRight' ? 1 : -1) * h.window.tubelet_seconds)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [seek, h.window.tubelet_seconds])

  const s = current >= 0 ? meta.timeline.surprise_mean[current] : null
  return (
    <main>
      <header>
        <h1>jepa-scope</h1>
        <span className="sub">{h.source.path.split('/').pop()}</span>
      </header>

      <div className="stage" style={{ aspectRatio: `${h.source.width} / ${h.source.height}` }}>
        <video
          ref={videoRef}
          src="/video"
          controls
          muted
          playsInline
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        />
        <canvas
          ref={canvasRef}
          className="overlay"
          style={{
            left: `${h.crop.x * 100}%`,
            top: `${h.crop.y * 100}%`,
            width: `${h.crop.w * 100}%`,
            height: `${h.crop.h * 100}%`,
          }}
        />
      </div>
      {glError && <p className="message">{glError}</p>}

      <Timeline meta={meta} current={current} duration={duration} time={time} onSeek={seek} />

      <section className="controls">
        <div className="segmented" role="radiogroup" aria-label="Overlay">
          {(['pca', 'surprise', 'off'] as const).map((m, i) => (
            <button key={m} role="radio" aria-checked={mode === m} className={mode === m ? 'on' : ''} onClick={() => setMode(m)}>
              {m === 'pca' ? 'Features (PCA)' : m === 'surprise' ? 'Surprise' : 'Off'} <kbd>{i + 1}</kbd>
            </button>
          ))}
        </div>
        <label>
          Opacity
          <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(+e.target.value)} />
        </label>
        <label>
          <input type="checkbox" checked={smooth} onChange={(e) => setSmooth(e.target.checked)} /> Smooth
        </label>
      </section>

      <dl className="facts">
        <dt>Tubelet</dt>
        <dd>{current >= 0 ? `${current + 1} / ${meta.timeline.count}` : 'outside embedded range'}</dd>
        <dt>Surprise</dt>
        <dd>{s === null ? 'n/a (context only)' : s.toFixed(4)}</dd>
        <dt>Model</dt>
        <dd>{h.model_id}</dd>
        <dt>Grid</dt>
        <dd>
          {h.grid.frames} frames → {h.grid.frames / h.grid.tubelet}×{h.grid.crop / h.grid.patch}×{h.grid.crop / h.grid.patch} tokens × {h.hidden_size}d
        </dd>
        <dt>Sampling</dt>
        <dd>
          {h.window.sample_fps} fps, windows of {h.window.frames} frames every {h.window.stride}
        </dd>
        <dt>Run</dt>
        <dd>
          {h.run.device} {h.run.dtype}, {h.run.seconds}s, preprocess {h.preprocess.hash}
        </dd>
      </dl>
      <p className="hint">
        Space plays or pauses. The arrow keys step one tubelet ({h.window.tubelet_seconds.toFixed(2)}s). Grey on the timeline is
        context the model never had to predict.
      </p>
    </main>
  )
}
