// School network graph: one dot per student, lines for shared links (classroom, bus route,
// household), and a shaded hull around each flagged cluster. The layout is a small seeded
// force simulation, so it is the same on every render for the same students and links.
import { useMemo, useState } from "react";
import type { ReactNode } from "react";
import { range, seeded, useViz } from "./charts";

export interface NetNode {
  id: string;
  groups: Record<string, string | null>;
  cls?: string;
  glyph?: string;
  href: boolean;
  legend?: string;
  aria: string;
  tip: ReactNode;
}

export interface NetHull {
  key: string;
  ids: string[];
  cls: string;
  label: string;
}

interface Link {
  a: number;
  b: number;
  layer: string;
  size: number;
}

type Pt = { x: number; y: number };
type XY = [number, number];

/** Links for one layer. Small groups are fully connected; large groups get a ring of near neighbours. */
function layerLinks(nodes: NetNode[], layer: string): Link[] {
  const groups = new Map<string, number[]>();
  nodes.forEach((n, i) => {
    const g = n.groups[layer];
    if (g) groups.set(g, [...(groups.get(g) ?? []), i]);
  });
  const links: Link[] = [];
  groups.forEach((members) => {
    const n = members.length;
    if (n < 2) return;
    if (n <= 8) {
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) links.push({ a: members[i], b: members[j], layer, size: n });
    } else {
      for (let i = 0; i < n; i++)
        for (const k of [1, 2, 3]) links.push({ a: members[i], b: members[(i + k) % n], layer, size: 7 });
    }
  });
  return links;
}

/** Force layout. Returns positions normalised to 0–1 on both axes. */
function layout(nodes: NetNode[], links: Link[], layers: string[], iterations = 340): Pt[] {
  const n = nodes.length;
  if (n === 0) return [];
  const rand = seeded(20261114 + n);
  const spread = 560 * Math.sqrt(Math.max(1, n / 28));
  const primary = layers[0];
  const groupNames = [...new Set(nodes.map((d) => (primary ? d.groups[primary] : null)).filter((g): g is string => !!g))].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }),
  );
  // Start each group on its own spoke of a circle so groups separate quickly.
  const pos = nodes.map((d) => {
    const g = primary ? d.groups[primary] : null;
    const gi = g ? groupNames.indexOf(g) : -1;
    const angle = gi >= 0 ? (gi / groupNames.length) * Math.PI * 2 - Math.PI / 2 : rand() * Math.PI * 2;
    const radius = gi >= 0 ? spread * 0.36 : spread * 0.08;
    const jitter = rand() * Math.PI * 2;
    return { x: Math.cos(angle) * radius * 1.35 + Math.cos(jitter) * 26, y: Math.sin(angle) * radius + Math.sin(jitter) * 26 };
  });

  for (let it = 0; it < iterations; it++) {
    const cool = Math.pow(1 - it / iterations, 1.5);
    const fx = new Float64Array(n);
    const fy = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = pos[j].x - pos[i].x;
        let dy = pos[j].y - pos[i].y;
        let d2 = dx * dx + dy * dy;
        if (d2 < 0.01) {
          dx = (rand() - 0.5) * 2;
          dy = (rand() - 0.5) * 2;
          d2 = dx * dx + dy * dy;
        }
        const d = Math.sqrt(d2);
        const f = Math.min(12, 2600 / d2) * cool;
        fx[i] -= (dx / d) * f;
        fy[i] -= (dy / d) * f;
        fx[j] += (dx / d) * f;
        fy[j] += (dy / d) * f;
      }
      // Gentle pull to the centre, stronger vertically so the graph stays wide.
      fx[i] -= pos[i].x * 0.02 * cool;
      fy[i] -= pos[i].y * 0.034 * cool;
    }
    for (const l of links) {
      const dx = pos[l.b].x - pos[l.a].x;
      const dy = pos[l.b].y - pos[l.a].y;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const rest = 40 + (l.size - 2) * 7;
      const f = ((d - rest) * 0.45 * cool) / Math.max(1, l.size - 1);
      fx[l.a] += (dx / d) * f;
      fy[l.a] += (dy / d) * f;
      fx[l.b] -= (dx / d) * f;
      fy[l.b] -= (dy / d) * f;
    }
    for (let i = 0; i < n; i++) {
      pos[i].x += fx[i];
      pos[i].y += fy[i];
    }
    // Keep dots from overlapping.
    for (let i = 0; i < n; i++)
      for (let j = i + 1; j < n; j++) {
        const dx = pos[j].x - pos[i].x;
        const dy = pos[j].y - pos[i].y;
        const d = Math.sqrt(dx * dx + dy * dy) || 1;
        if (d < 30) {
          const push = (30 - d) / 2;
          pos[i].x -= (dx / d) * push;
          pos[i].y -= (dy / d) * push;
          pos[j].x += (dx / d) * push;
          pos[j].y += (dy / d) * push;
        }
      }
  }

  const xs = pos.map((p) => p.x);
  const ys = pos.map((p) => p.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  return pos.map((p) => ({ x: x1 > x0 ? (p.x - x0) / (x1 - x0) : 0.5, y: y1 > y0 ? (p.y - y0) / (y1 - y0) : 0.5 }));
}

/** Convex hull (monotone chain). */
function hull(points: XY[]): XY[] {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (pts.length < 3) return pts;
  const cross = (o: XY, a: XY, b: XY) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower: XY[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper: XY[] = [];
  for (const p of [...pts].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

/** Points on a circle of radius `r` around each point, so the hull has rounded padding. */
const padded = (pts: Pt[], r: number): XY[] =>
  pts.flatMap((p) => range(18).map((i): XY => [p.x + Math.cos((i / 18) * Math.PI * 2) * r, p.y + Math.sin((i / 18) * Math.PI * 2) * r]));

const toPath = (pts: XY[]) => pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("") + "Z";

/**
 * Outline for a cluster. One convex hull when it would not swallow other students;
 * otherwise a chain of capsules along a minimum spanning tree of the members.
 */
function clusterPaths(members: Pt[], others: Pt[], r: number): string[] {
  const h = hull(padded(members, r));
  const inside = (p: Pt) => {
    let sign = 0;
    for (let i = 0; i < h.length; i++) {
      const a = h[i];
      const b = h[(i + 1) % h.length];
      const c = (b[0] - a[0]) * (p.y - a[1]) - (b[1] - a[1]) * (p.x - a[0]);
      if (Math.abs(c) < 1e-9) continue;
      if (sign === 0) sign = Math.sign(c);
      else if (Math.sign(c) !== sign) return false;
    }
    return true;
  };
  if (members.length < 3 || !others.some(inside)) return [toPath(h)];

  const inTree = [0];
  const paths: string[] = [];
  while (inTree.length < members.length) {
    let best: [number, number, number] = [-1, -1, Infinity];
    for (const i of inTree)
      members.forEach((p, j) => {
        if (inTree.includes(j)) return;
        const d = Math.hypot(p.x - members[i].x, p.y - members[i].y);
        if (d < best[2]) best = [i, j, d];
      });
    inTree.push(best[1]);
    paths.push(toPath(hull(padded([members[best[0]], members[best[1]]], r))));
  }
  return paths;
}

interface Label {
  key: string;
  text: string;
  cls: string;
  x: number;
  y: number;
  w: number;
}

export function NetworkGraph({ nodes, layers, initial, hulls, onOpen, legend, ariaLabel }: {
  nodes: NetNode[];
  layers: { key: string; label: string }[];
  initial: string[];
  hulls: NetHull[];
  onOpen: (id: string) => void;
  legend: { key: string; label: string; cls: string; glyph: string; count: number }[];
  ariaLabel: string;
}) {
  const v = useViz();
  const [on, setOn] = useState(initial);
  const [hoverNode, setHoverNode] = useState<string | null>(null);
  const [hoverLegend, setHoverLegend] = useState<string | null>(null);
  const width = v.width;
  const height = width < 520 ? 460 : 540;

  const active = layers.map((l) => l.key).filter((k) => on.includes(k));
  const activeKey = active.join("+");
  const signature = nodes.map((n) => `${n.id}:${layers.map((l) => n.groups[l.key] ?? "").join("|")}`).join(";");
  const { links, unit } = useMemo(() => {
    const links = active.flatMap((k) => layerLinks(nodes, k));
    return { links, unit: layout(nodes, links, active) };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signature, activeKey]);

  const pad = { x: 30, t: 34, b: 26 };
  const pos = unit.map((p) => ({ x: pad.x + p.x * Math.max(1, width - pad.x * 2), y: pad.t + p.y * (height - pad.t - pad.b) }));
  const index = new Map(nodes.map((n, i) => [n.id, i]));

  const near = new Set<number>();
  if (hoverNode !== null) {
    const i = index.get(hoverNode);
    if (i !== undefined) {
      near.add(i);
      links.forEach((l) => {
        if (l.a === i) near.add(l.b);
        if (l.b === i) near.add(l.a);
      });
    }
  }
  const dimmed = (i: number) => (hoverNode !== null ? !near.has(i) : hoverLegend !== null ? nodes[i].legend !== hoverLegend : false);

  // Place hull and group labels where they do not cover dots or each other.
  const primary = active[0];
  const labels: Label[] = [];
  if (width > 0 && pos.length) {
    const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x));
    const collides = (x: number, y: number, w: number) =>
      pos.some((p) => Math.abs(p.x - x) < w / 2 + 14 && Math.abs(p.y - (y - 4)) < 19) ||
      labels.some((l) => Math.abs(l.x - x) < (l.w + w) / 2 + 8 && Math.abs(l.y - y) < 16);
    const place = (key: string, text: string, cls: string, members: number[], off: number) => {
      const pts = members.map((i) => pos[i]);
      const w = text.length * (cls === "net-hull-label" ? 6.9 : 6.3);
      const xs = pts.map((p) => p.x);
      const ys = pts.map((p) => p.y);
      const cx = xs.reduce((a, b) => a + b, 0) / xs.length;
      const cy = ys.reduce((a, b) => a + b, 0) / ys.length;
      const top = pts[ys.indexOf(Math.min(...ys))];
      const bottom = pts[ys.indexOf(Math.max(...ys))];
      const candidates: XY[] = (
        [
          [cx, Math.min(...ys) - off],
          [top.x, top.y - off],
          [cx, Math.max(...ys) + off + 9],
          [bottom.x, bottom.y + off + 9],
          [Math.min(...xs) - w / 2 - off, cy + 4],
          [Math.max(...xs) + w / 2 + off, cy + 4],
          [top.x - w / 2, top.y - off],
          [top.x + w / 2, top.y - off],
        ] as XY[]
      ).map(([x, y]): XY => [clamp(x, w / 2 + 2, width - w / 2 - 2), clamp(y, 12, height - 5)]);
      const [x, y] = candidates.find(([x, y]) => !collides(x, y, w)) ?? candidates[0];
      labels.push({ key, text, cls, x, y, w });
    };

    hulls.forEach((h) => {
      const members = h.ids.map((id) => index.get(id)).filter((i): i is number => i !== undefined);
      if (members.length) place(h.key, h.label, "net-hull-label", members, 30);
    });
    if (primary) {
      const groups = new Map<string, number[]>();
      nodes.forEach((n, i) => {
        const g = n.groups[primary];
        if (g) groups.set(g, [...(groups.get(g) ?? []), i]);
      });
      groups.forEach((members, g) => {
        if (members.length >= 2 && !hulls.some((h) => h.label.startsWith(g))) place(`g-${g}`, g, "net-group", members, 18);
      });
    }
  }

  // At least one layer always stays on.
  const toggle = (k: string) => setOn((cur) => (cur.includes(k) ? (cur.length > 1 ? cur.filter((x) => x !== k) : cur) : [...cur, k]));

  return (
    <div className="net">
      <div className="net-bar">
        <span id="net-links">Links</span>
        <div className="seg-toggle" role="group" aria-labelledby="net-links">
          {layers.map((l) => (
            <button key={l.key} type="button" className={on.includes(l.key) ? "on" : ""} aria-pressed={on.includes(l.key)} onClick={() => toggle(l.key)}>
              <i className={`key key-link l-${l.key}`} />
              {l.label}
            </button>
          ))}
        </div>
      </div>
      <div className="viz" ref={v.ref} style={{ height }}>
        {width > 0 && (
          <svg
            viewBox={`0 0 ${width} ${height}`}
            width={width}
            height={height}
            role="group"
            aria-label={ariaLabel}
            className={hoverNode !== null || hoverLegend !== null ? "net-svg focus" : "net-svg"}
          >
            <g key={`h-${activeKey}`} className="net-fade">
              {hulls.map((h) => {
                const members = h.ids
                  .map((id) => index.get(id))
                  .filter((i): i is number => i !== undefined)
                  .map((i) => pos[i]);
                if (!members.length) return null;
                const others = nodes.filter((n) => !h.ids.includes(n.id)).map((n) => pos[index.get(n.id)!]);
                const paths = clusterPaths(members, others, 21);
                return (
                  <g key={h.key} className={`net-hull ${h.cls}`}>
                    {paths.map((d, i) => (
                      <path key={i} d={d} className="hull-edge" />
                    ))}
                    {paths.map((d, i) => (
                      <path key={i} d={d} className="hull-mask" />
                    ))}
                    <g className="hull-fill">
                      {paths.map((d, i) => (
                        <path key={i} d={d} />
                      ))}
                    </g>
                  </g>
                );
              })}
            </g>
            <g key={`l-${activeKey}`} className="net-fade">
              {links.map((l, i) => (
                <line
                  key={i}
                  x1={pos[l.a].x}
                  y1={pos[l.a].y}
                  x2={pos[l.b].x}
                  y2={pos[l.b].y}
                  className={`net-link l-${l.layer} ${hoverNode !== null && (nodes[l.a].id === hoverNode || nodes[l.b].id === hoverNode) ? "on" : ""}`}
                />
              ))}
            </g>
            {nodes.map((n, i) => {
              const inHull = hulls.find((h) => h.ids.includes(n.id));
              const open = () => n.href && onOpen(n.id);
              const enter = (el: Element) => {
                setHoverNode(n.id);
                v.show(n.tip, el.querySelector(".net-dot") ?? el);
              };
              const leave = () => {
                setHoverNode(null);
                v.hide();
              };
              return (
                <g
                  key={n.id}
                  className={`net-node ${n.cls ?? "present"} ${n.href ? "link" : ""} ${dimmed(i) ? "dim" : ""}`}
                  style={{ transform: `translate(${pos[i].x}px, ${pos[i].y}px)` }}
                  tabIndex={n.href ? 0 : undefined}
                  role={n.href ? "link" : "img"}
                  aria-label={n.aria}
                  onClick={open}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      open();
                    }
                  }}
                  onPointerEnter={(e) => enter(e.currentTarget)}
                  onPointerLeave={leave}
                  onFocus={(e) => enter(e.currentTarget)}
                  onBlur={leave}
                >
                  <circle r={15} className="v-hit" />
                  {inHull && n.cls && <circle r={13.5} className={`net-ring ${inHull.cls}`} />}
                  <circle r={n.cls ? 9.5 : 5} className="net-dot" />
                  {n.glyph && (
                    <text y={3.6} textAnchor="middle" className="net-glyph">
                      {n.glyph}
                    </text>
                  )}
                </g>
              );
            })}
            <g key={`t-${activeKey}`} className="net-fade net-labels">
              {labels.map((l) => (
                <text key={l.key} x={l.x} y={l.y} textAnchor="middle" className={l.cls}>
                  {l.text}
                </text>
              ))}
            </g>
          </svg>
        )}
        {v.tipNode}
      </div>
      <ul className="viz-legend net-legend">
        {legend.map((l) => (
          <li key={l.key} className={hoverLegend === l.key ? "hot" : ""} onPointerEnter={() => setHoverLegend(l.key)} onPointerLeave={() => setHoverLegend(null)}>
            <i className={`key key-node ${l.cls}`}>{l.glyph}</i>
            {l.label} <b>{l.count}</b>
          </li>
        ))}
        <li>
          <i className="key key-present" />
          Present
        </li>
        {hulls.length > 0 && (
          <li>
            <i className="key key-hull" />
            Flagged cluster
          </li>
        )}
      </ul>
    </div>
  );
}
