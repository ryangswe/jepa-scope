import type { Meta } from './types'

interface Props {
  meta: Meta
  current: number
  duration: number
  time: number
  onSeek: (t: number) => void
}

const H = 64

// Mean surprise per tubelet as bars on the video's own time axis. Click anywhere to seek.
export function Timeline({ meta, current, duration, time, onSeek }: Props) {
  const { time: times, surprise_mean } = meta.timeline
  const span = meta.header.window.tubelet_seconds
  const total = Math.max(duration, times[times.length - 1] + span)
  const finite = surprise_mean.filter((v): v is number => v !== null)
  // Scale from the clip's own minimum so small relative jumps stay visible. A bar's height is
  // relative to this clip only; compare raw values across clips in the facts panel.
  const max = finite.length ? Math.max(...finite) : 1
  const min = finite.length ? Math.min(...finite) : 0
  const range = max - min || 1
  const x = (t: number) => (t / total) * 1000

  return (
    <svg
      className="timeline"
      viewBox={`0 0 1000 ${H}`}
      preserveAspectRatio="none"
      role="slider"
      aria-label="Surprise over time; click to seek"
      aria-valuemin={0}
      aria-valuemax={total}
      aria-valuenow={time}
      onPointerDown={(e) => {
        const r = e.currentTarget.getBoundingClientRect()
        onSeek(((e.clientX - r.left) / r.width) * total)
      }}
    >
      {times.map((t, i) => {
        const v = surprise_mean[i]
        if (v === null) {
          return <rect key={i} className="nodata" x={x(t)} y={0} width={x(span)} height={H} />
        }
        const h = 3 + ((v - min) / range) * (H - 7)
        return (
          <rect key={i} className={i === current ? 'bar on' : 'bar'} x={x(t)} y={H - h} width={x(span) - 0.5} height={h} />
        )
      })}
      <line className="playhead" x1={x(time)} x2={x(time)} y1={0} y2={H} />
    </svg>
  )
}
