// Small SVG/CSS chart kit shared by the Overview, Root causes, Revenue and Evidence pages.
// Styling lives in styles.css under the `viz`, `v-*`, `bt`, `barlist`, `heat`, `bullet` and `key` classes.
import { useCallback, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties, KeyboardEvent, PointerEvent, ReactNode } from "react";

/** Format a 0–1 share as a percent. */
export const pct = (x: number | null | undefined, digits = 0) => (x == null ? "–" : `${(x * 100).toFixed(digits)}%`);

/** A round axis maximum at or above `v`, split into `steps` ticks. */
export function niceMax(v: number, steps = 4) {
  if (v <= 0) return steps;
  const raw = v / steps;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  return ([1, 2, 2.5, 5, 10].find((m) => m * mag >= raw - 1e-9) ?? 10) * mag * steps;
}

export const range = (n: number) => Array.from({ length: n }, (_, i) => i);

/** Small seeded random number generator, so layouts are stable between renders. */
export function seeded(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ---------- tooltip + width tracking ----------

export function useViz() {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);
  const [tip, setTip] = useState<{ content: ReactNode; x: number; y: number; below: boolean } | null>(null);

  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setWidth(Math.round(el.clientWidth));
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const showAt = useCallback((content: ReactNode, x: number, y: number) => {
    const w = ref.current?.clientWidth ?? 0;
    const half = Math.min(120, w / 2);
    setTip({ content, x: Math.max(half, Math.min(w - half, x)), y, below: y < 96 });
  }, []);

  const show = useCallback(
    (content: ReactNode, target: Element) => {
      const el = ref.current;
      if (!el) return;
      const box = el.getBoundingClientRect();
      const r = target.getBoundingClientRect();
      const below = r.top - box.top < 96;
      showAt(content, r.left + r.width / 2 - box.left, below ? r.bottom - box.top : r.top - box.top);
    },
    [showAt],
  );

  const hide = useCallback(() => setTip(null), []);

  const bind = useCallback(
    (content: ReactNode) => ({
      onPointerEnter: (e: PointerEvent<Element>) => show(content, e.currentTarget),
      onPointerLeave: hide,
      onFocus: (e: { currentTarget: Element }) => show(content, e.currentTarget),
      onBlur: hide,
    }),
    [show, hide],
  );

  const tipNode = tip ? (
    <div className={tip.below ? "viz-tip below" : "viz-tip"} style={{ left: tip.x, top: tip.y }} aria-hidden="true">
      {tip.content}
    </div>
  ) : null;

  return { ref, width, bind, show, showAt, hide, tipNode };
}

export interface TipRow {
  cls?: string;
  label: ReactNode;
  value: ReactNode;
}

export function Tip({ title, rows, foot }: { title: ReactNode; rows?: TipRow[]; foot?: ReactNode }) {
  return (
    <>
      <div className="tip-title">{title}</div>
      {rows?.map((r, i) => (
        <div key={i} className="tip-row">
          {r.cls !== undefined && <i className={`tip-key ${r.cls}`} />}
          <span>{r.label}</span>
          <b>{r.value}</b>
        </div>
      ))}
      {foot && <div className="tip-foot">{foot}</div>}
    </>
  );
}

// ---------- layout pieces ----------

export function Panel({ title, note, actions, className, children }: { title: ReactNode; note?: ReactNode; actions?: ReactNode; className?: string; children?: ReactNode }) {
  return (
    <div className={`card panel ${className ?? ""}`}>
      <div className="panel-head">
        <div>
          <h3>{title}</h3>
          {note && <p className="panel-note">{note}</p>}
        </div>
        {actions}
      </div>
      {children}
    </div>
  );
}

/** Placeholder while a chart loads, or a message when it has nothing to show. */
export function VizState({ height, children }: { height: number; children?: ReactNode }) {
  return (
    <div className={children ? "viz-state" : "viz-state loading"} style={{ minHeight: height }} role="status">
      {children ?? "Loading…"}
    </div>
  );
}

export interface LegendItem {
  label: ReactNode;
  kind?: "rect" | "tick" | "dot" | "line" | "light" | "ring" | "ref";
  cls?: string;
}

export function Legend({ items, className }: { items: LegendItem[]; className?: string }) {
  return (
    <ul className={`viz-legend ${className ?? ""}`}>
      {items.map((it, i) => (
        <li key={i}>
          <i className={`key key-${it.kind ?? "rect"} ${it.cls ?? ""}`} />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

/** The chart's numbers as a collapsible table, for screen readers and exact values. */
export function DataTable({ head, rows, label = "Show the numbers as a table" }: { head: string[]; rows: ReactNode[][]; label?: string }) {
  return (
    <details className="viz-table">
      <summary>{label}</summary>
      <table>
        <thead>
          <tr>
            {head.map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((c, j) => (
                <td key={j}>{c}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

// ---------- bars ----------

/** One horizontal bar. `marker` is a tick, `dot` a second value, `refAt` a reference line. */
export function Bar({ value, max = 1, cls = "", light, track, marker, dot, refAt, thin }: {
  value: number;
  max?: number;
  cls?: string;
  light?: boolean;
  track?: boolean;
  marker?: number;
  dot?: number;
  refAt?: number;
  thin?: boolean;
}) {
  const at = (v: number) => `${Math.max(0, Math.min(1, v / max)) * 100}%`;
  return (
    <div className={`bt ${cls} ${track ? "tracked" : ""} ${thin ? "thin" : ""}`}>
      <i className={light ? "light" : ""} style={{ width: at(value) }} />
      {refAt != null && <s style={{ left: at(refAt) }} />}
      {marker != null && <u style={{ left: at(marker) }} />}
      {dot != null && <b style={{ left: at(dot) }} />}
    </div>
  );
}

export interface BarRow {
  key: string;
  label: ReactNode;
  sub?: ReactNode;
  value: number;
  dot?: number;
  cls?: string;
  dim?: boolean;
  rule?: boolean;
  display: ReactNode;
  aria?: string;
  tip?: ReactNode;
}

export function BarList({ rows, max, track, refAt, refLabel, ariaLabel, swatch }: {
  rows: BarRow[];
  max?: number;
  track?: boolean;
  refAt?: number;
  refLabel?: ReactNode;
  ariaLabel: string;
  swatch?: boolean;
}) {
  const v = useViz();
  const top = max ?? Math.max(1e-9, ...rows.map((r) => Math.max(r.value, r.dot ?? 0)));
  return (
    <div className="viz" ref={v.ref}>
      <ul className="barlist" aria-label={ariaLabel}>
        {rows.map((r) => (
          <li key={r.key} className={`${r.dim ? "dim" : ""} ${r.rule ? "rule" : ""}`} tabIndex={0} aria-label={r.aria} {...(r.tip ? v.bind(r.tip) : {})}>
            <span className="bl-name">
              {swatch && <i className={`key key-rect ${r.cls ?? ""}`} />}
              <span>{r.label}</span>
              {r.sub && <small>{r.sub}</small>}
            </span>
            <Bar value={r.value} max={top} cls={r.cls} track={track} dot={r.dot} refAt={refAt} />
            <span className="bl-val">{r.display}</span>
          </li>
        ))}
      </ul>
      {refAt != null && refLabel && (
        <p className="viz-foot">
          <i className="key key-ref" /> {refLabel}
        </p>
      )}
      {v.tipNode}
    </div>
  );
}

// ---------- stacked columns ----------

export interface ColumnSeg {
  key: string;
  label: string;
  value: number;
  cls: string;
}

export interface Column {
  key: string;
  label: string;
  sub?: string;
  segs: ColumnSeg[];
}

export function StackedColumns({ data, height = 230, unit, ariaLabel }: { data: Column[]; height?: number; unit: string; ariaLabel: string }) {
  const v = useViz();
  const w = v.width;
  const m = { l: 30, r: 8, t: 24, b: 40 };
  const totals = data.map((d) => d.segs.reduce((a, s) => a + s.value, 0));
  const peak = Math.max(4, ...totals);
  const top = peak <= 40 ? Math.ceil(peak / 4) * 4 : niceMax(peak);
  const plotH = height - m.t - m.b;
  const y = (val: number) => m.t + plotH - (val / top) * plotH;
  const band = data.length ? (w - m.l - m.r) / data.length : 0;
  const barW = Math.min(24, band * 0.6);

  return (
    <div className="viz" ref={v.ref} style={{ height }}>
      {w > 0 && (
        <svg viewBox={`0 0 ${w} ${height}`} width={w} height={height} role="img" aria-label={ariaLabel}>
          {range(5).map((i) => {
            const val = (top / 4) * i;
            return (
              <g key={i}>
                <line x1={m.l} x2={w - m.r} y1={y(val)} y2={y(val)} className={i === 0 ? "v-base" : "v-grid"} />
                <text x={m.l - 8} y={y(val) + 4} textAnchor="end" className="v-tick">
                  {val}
                </text>
              </g>
            );
          })}
          <text x={0} y={11} className="v-axis">
            {unit}
          </text>
          {data.map((d, i) => {
            const cx = m.l + band * (i + 0.5);
            const x0 = cx - barW / 2;
            const segs = d.segs.filter((s) => s.value > 0);
            let acc = 0;
            return (
              <g key={d.key} className="v-col" style={{ animationDelay: `${i * 60}ms` }}>
                {segs.map((s, j) => {
                  const yTop = y(acc + s.value);
                  const yBot = y(acc);
                  acc += s.value;
                  const last = j === segs.length - 1;
                  const h = Math.max(1, yBot - yTop - (last ? 0 : 2));
                  const top0 = yBot - h;
                  const r = last ? Math.min(4, h) : 0;
                  const path = `M${x0},${top0 + h}V${top0 + r}Q${x0},${top0} ${x0 + r},${top0}H${x0 + barW - r}Q${x0 + barW},${top0} ${x0 + barW},${top0 + r}V${top0 + h}Z`;
                  return (
                    <g
                      key={s.key}
                      className={`v-seg ${s.cls}`}
                      tabIndex={0}
                      role="img"
                      aria-label={`${d.label}: ${s.value} ${s.label}`}
                      {...v.bind(<Tip title={d.label} rows={[{ cls: s.cls, label: s.label, value: s.value }, { label: "All absences", value: totals[i] }]} />)}
                    >
                      <rect x={cx - band / 2 + 2} y={top0 - 1} width={band - 4} height={h + 2} className="v-hit" />
                      <path d={path} />
                    </g>
                  );
                })}
                {totals[i] > 0 && (
                  <text x={cx} y={y(totals[i]) - 7} textAnchor="middle" className="v-val">
                    {totals[i]}
                  </text>
                )}
                <text x={cx} y={height - m.b + 17} textAnchor="middle" className="v-cat">
                  {d.label}
                </text>
                {d.sub && (
                  <text x={cx} y={height - m.b + 32} textAnchor="middle" className="v-tick">
                    {d.sub}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      )}
      {v.tipNode}
    </div>
  );
}

// ---------- heatmap ----------

export interface HeatCell {
  value: number;
  flagged?: boolean;
  aria: string;
  tip?: ReactNode;
}

export function Heatmap({ rows, cols, cells, rowTotals, max, ariaLabel, totalLabel }: {
  rows: { key: string; label: string; sub?: string }[];
  cols: { key: string; label: string; sub?: string }[];
  cells: HeatCell[][];
  rowTotals?: number[];
  max: number;
  ariaLabel: string;
  totalLabel?: string;
}) {
  const v = useViz();
  const step = (n: number) => (n <= 0 ? 0 : Math.max(1, Math.min(5, Math.ceil((n / max) * 5))));
  return (
    <div className="viz" ref={v.ref}>
      <div className="heat" role="table" aria-label={ariaLabel} style={{ "--cols": cols.length } as CSSProperties}>
        <div role="row" className="heat-row heat-head">
          <span role="columnheader" />
          {cols.map((c) => (
            <span key={c.key} role="columnheader">
              {c.label}
              {c.sub && <small>{c.sub}</small>}
            </span>
          ))}
          {rowTotals && (
            <span role="columnheader" className="heat-total">
              {totalLabel ?? "Total"}
            </span>
          )}
        </div>
        {rows.map((r, i) => (
          <div key={r.key} role="row" className="heat-row">
            <span role="rowheader" className="heat-name">
              {r.label}
              {r.sub && <small>{r.sub}</small>}
            </span>
            {cols.map((c, j) => {
              const cell = cells[i][j];
              return (
                <span
                  key={c.key}
                  role="cell"
                  className={`heat-cell ${cell.flagged ? "flagged" : ""}`}
                  data-step={step(cell.value)}
                  aria-label={cell.aria}
                  tabIndex={cell.value > 0 ? 0 : undefined}
                  {...(cell.tip && cell.value > 0 ? v.bind(cell.tip) : {})}
                >
                  {cell.value > 0 ? cell.value : ""}
                </span>
              );
            })}
            {rowTotals && (
              <span role="cell" className="heat-total">
                {rowTotals[i]}
              </span>
            )}
          </div>
        ))}
      </div>
      {v.tipNode}
    </div>
  );
}

export function HeatScale({ max, flaggedLabel }: { max: number; flaggedLabel?: string }) {
  const bounds = range(5).map((i) => Math.max(1, Math.ceil(((i + 1) / 5) * max)));
  const steps = range(5).filter((i) => i === 0 || bounds[i] !== bounds[i - 1]);
  return (
    <div className="heat-scale">
      <span>0</span>
      <i className="heat-cell" data-step={0} />
      {steps.map((i) => (
        <i key={i} className="heat-cell" data-step={i + 1} />
      ))}
      <span>{max} absent</span>
      {flaggedLabel && (
        <span className="heat-flag">
          <i className="heat-cell flagged" data-step={0} /> {flaggedLabel}
        </span>
      )}
    </div>
  );
}

// ---------- bullet: observed against expected ----------

export function Bullet({ observed, expected, cls = "", unit = "students", compact }: { observed: number; expected: number; cls?: string; unit?: string; compact?: boolean }) {
  const top = niceMax(Math.max(observed, expected) * 1.05, 4);
  const at = (v: number) => `${Math.min(100, (v / top) * 100)}%`;
  const ratio = expected > 0 ? observed / expected : null;
  const exp = expected.toFixed(1);
  return (
    <div
      className={`bullet ${cls} ${compact ? "compact" : ""}`}
      role="img"
      aria-label={`${observed} ${unit} absent, ${exp} expected${ratio ? `, ${ratio.toFixed(1)} times the expected number` : ""}`}
    >
      <div className="bullet-figs">
        <span>
          <b>{observed}</b> observed
        </span>
        <span>
          <b>{exp}</b> expected
        </span>
        {ratio !== null && (
          <span className="bullet-ratio">
            <b>{ratio.toFixed(1)}×</b> the usual
          </span>
        )}
      </div>
      <div className="bullet-track">
        <i style={{ width: at(observed) }} />
        <u style={{ left: at(expected) }} title={`Expected: ${exp}`} />
      </div>
      <div className="bullet-axis">
        {range(5).map((i) => (
          <span key={i}>{+((top / 4) * i).toFixed(1)}</span>
        ))}
      </div>
      {!compact && (
        <Legend
          items={[
            { label: `Observed ${unit}`, kind: "rect", cls },
            { label: "Expected from the school-wide rate", kind: "tick" },
          ]}
        />
      )}
    </div>
  );
}

// ---------- dumbbell: before and after per row ----------

export interface DumbbellRow {
  key: string;
  label: string;
  sub?: string;
  from: number;
  to: number;
}

export function Dumbbell({ rows, fromLabel, toLabel, fromCls, toCls, ariaLabel }: {
  rows: DumbbellRow[];
  fromLabel: string;
  toLabel: string;
  fromCls: string;
  toCls: string;
  ariaLabel: string;
}) {
  const v = useViz();
  const w = v.width;
  const narrow = w < 430;
  const rowH = narrow ? 50 : 38;
  const m = { l: narrow ? 12 : 172, r: 40, t: 26, b: 6 };
  const height = m.t + rows.length * rowH + m.b;
  const x = (f: number) => m.l + f * (w - m.l - m.r);

  return (
    <div className="viz" ref={v.ref} style={{ minHeight: height }}>
      {w > 0 && (
        <svg viewBox={`0 0 ${w} ${height}`} width={w} height={height} role="img" aria-label={ariaLabel}>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={m.t - 6} y2={height - m.b} className={t === 0 ? "v-base" : "v-grid"} />
              <text x={x(t)} y={m.t - 12} textAnchor="middle" className="v-tick">
                {pct(t)}
              </text>
            </g>
          ))}
          {rows.map((r, i) => {
            const cy = m.t + i * rowH + (narrow ? 34 : rowH / 2);
            const same = Math.abs(r.to - r.from) < 5e-3;
            const lo = Math.min(r.from, r.to);
            const hi = Math.max(r.from, r.to);
            const delta = Math.round((r.to - r.from) * 100);
            return (
              <g
                key={r.key}
                className="v-row"
                tabIndex={0}
                role="img"
                aria-label={`${r.label}: ${fromLabel} ${pct(r.from)}, ${toLabel} ${pct(r.to)}`}
                {...v.bind(
                  <Tip
                    title={r.label}
                    rows={[
                      { cls: toCls, label: toLabel, value: pct(r.to) },
                      { cls: fromCls, label: fromLabel, value: pct(r.from) },
                    ]}
                    foot={`${same ? "No change" : `${delta > 0 ? "+" : ""}${delta} points`}${r.sub ? ` · ${r.sub}` : ""}`}
                  />,
                )}
              >
                <rect x={0} y={m.t + i * rowH} width={w} height={rowH} className="v-hit" />
                <text x={narrow ? m.l : m.l - 14} y={narrow ? cy - 16 : cy - 1} textAnchor={narrow ? "start" : "end"} className="v-cat">
                  {r.label}
                </text>
                {r.sub && (
                  <text x={narrow ? w - m.r : m.l - 14} y={narrow ? cy - 16 : cy + 13} textAnchor="end" className="v-tick">
                    {r.sub}
                  </text>
                )}
                {!same && <line x1={x(lo)} x2={x(hi)} y1={cy} y2={cy} className={`v-link ${toCls}`} />}
                {!same && <circle cx={x(r.from)} cy={cy} r={5.5} className={`v-dot ${fromCls}`} />}
                <circle cx={x(r.to)} cy={cy} r={6.5} className={`v-dot ${toCls}`} />
                {!same && (
                  <text x={x(lo) - 11} y={cy + 4} textAnchor="end" className="v-tick">
                    {pct(lo)}
                  </text>
                )}
                <text x={x(hi) + 12} y={cy + 4} className="v-val">
                  {pct(hi)}
                </text>
              </g>
            );
          })}
        </svg>
      )}
      {v.tipNode}
    </div>
  );
}

// ---------- line chart ----------

export interface Series {
  key: string;
  label: string;
  cls: string;
  bold?: boolean;
  points: { x: number; y: number }[];
}

export interface Axis {
  domain: [number, number];
  ticks: number[];
  fmt: (v: number) => string;
  label: string;
  log?: boolean;
}

export function LineChart({ series, x: xa, y: ya, height = 280, dots, interpolate, marks, refX, ariaLabel, tipTitle }: {
  series: Series[];
  x: Axis;
  y: Axis;
  height?: number;
  dots?: boolean;
  interpolate?: boolean;
  marks?: { series: string; x: number; y: number }[];
  refX?: { v: number; label: string };
  ariaLabel: string;
  tipTitle?: (x: number) => string;
}) {
  const v = useViz();
  const w = v.width;
  const [hover, setHover] = useState<number | null>(null);
  const labelsRight = w >= 470;
  const longest = Math.max(0, ...series.map((s) => s.label.length));
  const m = { l: 46, r: labelsRight ? 20 + longest * 6.6 : 16, t: 30, b: 46 };
  const tx = (val: number) => (xa.log ? Math.log10(val) : val);
  const ty = (val: number) => (ya.log ? Math.log10(val) : val);
  const sx = (val: number) => m.l + ((tx(val) - tx(xa.domain[0])) / (tx(xa.domain[1]) - tx(xa.domain[0]))) * (w - m.l - m.r);
  const sy = (val: number) => height - m.b - ((ty(val) - ty(ya.domain[0])) / (ty(ya.domain[1]) - ty(ya.domain[0]))) * (height - m.t - m.b);

  const clipped = useMemo(
    () =>
      series.map((s) => ({
        ...s,
        points: s.points.filter((p) => p.x >= xa.domain[0] && p.x <= xa.domain[1]).sort((a, b) => a.x - b.x || a.y - b.y),
      })),
    [series, xa.domain],
  );

  const stops = useMemo(
    () =>
      interpolate
        ? range(25).map((i) => {
            const t = tx(xa.domain[0]) + ((tx(xa.domain[1]) - tx(xa.domain[0])) * i) / 24;
            return xa.log ? Math.pow(10, t) : t;
          })
        : [...new Set(clipped.flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [clipped, interpolate, xa.domain, xa.log],
  );

  const valueAt = (s: Series, at: number): number | null => {
    const pts = s.points;
    if (!pts.length) return null;
    if (!interpolate) return pts.find((p) => p.x === at)?.y ?? null;
    if (at < pts[0].x || at > pts[pts.length - 1].x) return null;
    for (let i = pts.length - 1; i >= 0; i--) {
      if (pts[i].x <= at) {
        const next = pts[i + 1];
        if (!next || next.x === pts[i].x) return pts[i].y;
        const f = (tx(at) - tx(pts[i].x)) / (tx(next.x) - tx(pts[i].x));
        return pts[i].y + (next.y - pts[i].y) * f;
      }
    }
    return null;
  };

  const focusAt = (at: number | null) => {
    setHover(at);
    if (at === null) return v.hide();
    const vals = clipped
      .map((s) => ({ s, val: valueAt(s, at) }))
      .filter((r): r is { s: Series; val: number } => r.val !== null)
      .sort((a, b) => b.val - a.val);
    const top = Math.min(height - m.b, ...vals.map((r) => sy(r.val)));
    v.showAt(
      <Tip title={tipTitle ? tipTitle(at) : `${xa.fmt(at)} ${xa.label.toLowerCase()}`} rows={vals.map((r) => ({ cls: `line ${r.s.cls}`, label: r.s.label, value: ya.fmt(r.val) }))} />,
      sx(at),
      top - 4,
    );
  };

  const nearest = (val: number) => stops.reduce((best, s) => (Math.abs(tx(s) - tx(val)) < Math.abs(tx(best) - tx(val)) ? s : best), stops[0]);

  const onMove = (e: PointerEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    const t = tx(xa.domain[0]) + f * (tx(xa.domain[1]) - tx(xa.domain[0]));
    const val = xa.log ? Math.pow(10, t) : t;
    focusAt(interpolate ? val : nearest(val));
  };

  const onKey = (e: KeyboardEvent<SVGSVGElement>) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const i = hover === null ? 0 : stops.indexOf(nearest(hover));
    focusAt(stops[Math.max(0, Math.min(stops.length - 1, i + (e.key === "ArrowRight" ? 1 : -1)))]);
  };

  // End-of-line labels, nudged apart so they do not overlap.
  const labels = clipped
    .filter((s) => s.points.length)
    .map((s) => ({ s, y0: sy(s.points[s.points.length - 1].y), x0: sx(s.points[s.points.length - 1].x), y: 0 }))
    .sort((a, b) => a.y0 - b.y0);
  labels.forEach((l, i) => (l.y = i === 0 ? l.y0 : Math.max(l.y0, labels[i - 1].y + 15)));
  if ((labels.length ? labels[labels.length - 1].y - (height - m.b) : 0) > 0) {
    for (let i = labels.length - 1; i >= 0; i--) labels[i].y = Math.min(labels[i].y, height - m.b - (labels.length - 1 - i) * 15);
  }

  return (
    <div className="viz" ref={v.ref} style={{ height }}>
      {w > 0 && (
        <svg
          viewBox={`0 0 ${w} ${height}`}
          width={w}
          height={height}
          role="img"
          tabIndex={0}
          className="v-line-chart"
          aria-label={`${ariaLabel}. Use the left and right arrow keys to read values.`}
          onKeyDown={onKey}
          onFocus={() => hover === null && focusAt(stops[stops.length - 1])}
          onBlur={() => focusAt(null)}
        >
          {ya.ticks.map((t, i) => (
            <g key={t}>
              <line x1={m.l} x2={w - m.r} y1={sy(t)} y2={sy(t)} className={i === 0 ? "v-base" : "v-grid"} />
              <text x={m.l - 8} y={sy(t) + 4} textAnchor="end" className="v-tick">
                {ya.fmt(t)}
              </text>
            </g>
          ))}
          {xa.ticks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={height - m.b} y2={height - m.b + 4} className="v-base" />
              <text x={sx(t)} y={height - m.b + 18} textAnchor="middle" className="v-tick">
                {xa.fmt(t)}
              </text>
            </g>
          ))}
          <text x={0} y={12} className="v-axis">
            {ya.label}
          </text>
          <text x={m.l + (w - m.l - m.r) / 2} y={height - 6} textAnchor="middle" className="v-axis">
            {xa.label}
          </text>
          {refX && (
            <g>
              <line x1={sx(refX.v)} x2={sx(refX.v)} y1={m.t} y2={height - m.b} className="v-ref" />
              <text x={sx(refX.v) + 6} y={m.t + 10} className="v-tick">
                {refX.label}
              </text>
            </g>
          )}
          {clipped.map((s) => (
            <g key={s.key} className={`v-series ${s.cls} ${s.bold ? "bold" : ""}`}>
              <path d={s.points.map((p, i) => `${i ? "L" : "M"}${sx(p.x).toFixed(1)},${sy(p.y).toFixed(1)}`).join("")} className="v-line" />
              {dots && s.points.map((p) => <circle key={p.x} cx={sx(p.x)} cy={sy(p.y)} r={4} className="v-dot" />)}
            </g>
          ))}
          {marks
            ?.filter((mk) => mk.x >= xa.domain[0] && mk.x <= xa.domain[1])
            .map((mk) => {
              const s = clipped.find((c) => c.key === mk.series);
              return <circle key={mk.series} cx={sx(mk.x)} cy={sy(mk.y)} r={6} className={`v-mark ${s?.cls ?? ""}`} />;
            })}
          {labelsRight &&
            labels.map((l) => {
              const moved = Math.abs(l.y - l.y0) > 3;
              return (
                <g key={l.s.key}>
                  {moved && <line x1={l.x0 + 5} x2={l.x0 + 13} y1={l.y0} y2={l.y} className="v-leader" />}
                  <text x={l.x0 + (moved ? 16 : 10)} y={l.y + 4} className={l.s.bold ? "v-val" : "v-cat"}>
                    {l.s.label}
                  </text>
                </g>
              );
            })}
          {hover !== null && (
            <g className="v-cross">
              <line x1={sx(hover)} x2={sx(hover)} y1={m.t} y2={height - m.b} />
              {clipped.map((s) => {
                const val = valueAt(s, hover);
                return val === null ? null : <circle key={s.key} cx={sx(hover)} cy={sy(val)} r={4.5} className={`v-dot ${s.cls}`} />;
              })}
            </g>
          )}
          <rect x={m.l} y={m.t} width={Math.max(0, w - m.l - m.r)} height={height - m.t - m.b} className="v-hit" onPointerMove={onMove} onPointerLeave={() => focusAt(null)} />
        </svg>
      )}
      {v.tipNode}
    </div>
  );
}
