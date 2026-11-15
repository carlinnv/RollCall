import { useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, fmtDate, money } from "../api";
import type { Barrier, Cluster, QueueRow, Student } from "../api";
import { useLoad } from "../hooks";
import { ErrorNote, Page } from "../components/ui";
import { BarList, Bullet, Heatmap, HeatScale, Legend, DataTable, Panel, StackedColumns, Tip, VizState } from "../components/charts";
import type { BarRow, Column } from "../components/charts";
import { NetworkGraph } from "../components/Network";
import type { NetHull, NetNode } from "../components/Network";

/** First word of a barrier, which is also its CSS class suffix (`barrier-illness`). */
const short = (b: Barrier | null | undefined) => (b ?? "unknown").split(" ")[0];

const NAME: Record<string, string> = {
  illness: "Illness",
  transportation: "Transportation",
  housing: "Housing or food",
  caregiving: "Caregiving",
  safety: "Safety",
  disengagement: "Disengagement",
  appointment: "Appointment or travel",
  unknown: "Unclear reply",
  awaiting: "Awaiting reply",
};

const GLYPH: Record<string, string> = {
  illness: "I",
  transportation: "T",
  housing: "H",
  caregiving: "C",
  safety: "S",
  disengagement: "D",
  appointment: "A",
  unknown: "?",
  awaiting: "",
};

const ORDER = ["illness", "transportation", "caregiving", "housing", "disengagement", "appointment", "safety", "unknown", "awaiting"];

const reasonOf = (r: QueueRow) => (r.classification ? short(r.classification.barrier) : "awaiting");
const byName = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true });
/** "Mon, Nov 9" → ["Mon", " Nov 9"] */
const dayParts = (iso: string) => fmtDate(iso).split(",");
const fmtP = (p: number) => (p < 0.001 ? p.toExponential(1) : p.toFixed(3));

function AbsencesByDay({ rows }: { rows: QueueRow[] }) {
  const days = [...new Set(rows.map((r) => r.date))].sort().slice(-5);
  const data: Column[] = days.map((d) => {
    const today = rows.filter((r) => r.date === d);
    const count = (s: string) => today.filter((r) => r.status === s).length;
    return {
      key: d,
      label: dayParts(d)[0],
      sub: dayParts(d)[1]?.trim(),
      segs: [
        { key: "captured", label: "reason captured", value: count("reason captured"), cls: "series-ok" },
        { key: "unknown", label: "reply unclear", value: count("unknown"), cls: "series-muted" },
        { key: "awaiting", label: "awaiting a reply", value: count("awaiting reply"), cls: "barrier-awaiting" },
      ],
    };
  });
  return (
    <>
      <StackedColumns data={data} height={210} unit="Absences" ariaLabel="Absences per school day, split by whether the reason is known" />
      <Legend
        items={[
          { label: "Reason captured", cls: "series-ok" },
          { label: "Reply unclear", cls: "series-muted" },
          { label: "Awaiting a reply", cls: "barrier-awaiting" },
        ]}
      />
      <DataTable
        head={["Day", "Reason captured", "Reply unclear", "Awaiting a reply", "All"]}
        rows={data.map((d) => [`${d.label} ${d.sub ?? ""}`, ...d.segs.map((s) => s.value), d.segs.reduce((a, s) => a + s.value, 0)])}
      />
    </>
  );
}

function WhyOut({ rows }: { rows: QueueRow[] }) {
  const total = rows.length;
  const n: Record<string, number> = {};
  rows.forEach((r) => (n[reasonOf(r)] = (n[reasonOf(r)] ?? 0) + 1));
  const known = ORDER.filter((k) => n[k] && k !== "unknown" && k !== "awaiting").sort((a, b) => n[b] - n[a]);
  const unknown = ["unknown", "awaiting"].filter((k) => n[k]);
  const share = (k: string) => Math.round((n[k] / total) * 100);
  const bars: BarRow[] = [...known, ...unknown].map((k, i) => ({
    key: k,
    label: NAME[k],
    value: n[k],
    cls: `barrier-${k}`,
    dim: i >= known.length,
    rule: i === known.length && known.length > 0,
    display: (
      <>
        {n[k]}
        <small>{share(k)}%</small>
      </>
    ),
    aria: `${NAME[k]}: ${n[k]} absences, ${share(k)}% of the week`,
    tip: (
      <Tip
        title={NAME[k]}
        rows={[
          { label: "Absences", value: n[k] },
          { label: "Share of the week", value: `${share(k)}%` },
        ]}
      />
    ),
  }));
  return <BarList rows={bars} swatch ariaLabel="Absences this week by reason, largest first" />;
}

type GroupBy = "classroom" | "bus_route";

function WhereAndWhen({ students, rows, clusters }: { students: Student[]; rows: QueueRow[]; clusters: Cluster[] }) {
  const [by, setBy] = useState<GroupBy>("classroom");
  const days = [...new Set(rows.map((r) => r.date))].sort().slice(-5);
  const groupOf = (s: Student) => (by === "classroom" ? s.classroom : s.bus_route ?? "No bus");
  const groups = [...new Set(students.map(groupOf))].sort((a, b) => (a === "No bus" ? 1 : b === "No bus" ? -1 : byName(a, b)));
  const size = (g: string) => students.filter((s) => groupOf(s) === g).length;

  const cells = groups.map((g) =>
    days.map((d) => {
      const out = rows.filter((r) => r.date === d && groupOf(r.student) === g);
      const flag = clusters.find((c) => c.link_type === by && c.link_value === g && d >= c.first_date && d <= c.last_date);
      const reasons = ORDER.map((k) => ({ k, n: out.filter((r) => reasonOf(r) === k).length })).filter((x) => x.n);
      return {
        value: out.length,
        flagged: !!flag && out.length > 0,
        aria: `${g}, ${fmtDate(d)}: ${out.length} of ${size(g)} absent${flag && out.length ? ", part of a flagged cluster" : ""}`,
        tip: (
          <Tip
            title={`${g} · ${fmtDate(d)}`}
            rows={[{ label: "Absent", value: `${out.length} of ${size(g)}` }, ...reasons.map((x) => ({ cls: `barrier-${x.k}`, label: NAME[x.k], value: x.n }))]}
            foot={flag && out.length ? `Flagged: ${flag.likely_cause ?? flag.barrier}` : undefined}
          />
        ),
      };
    }),
  );
  const max = Math.max(3, ...cells.flat().map((c) => c.value));

  return (
    <Panel
      title="Where and when"
      note="Absences per day in each group. A dark run along a row points at that group."
      actions={
        <div className="seg-toggle" role="group" aria-label="Rows">
          <button type="button" className={by === "classroom" ? "on" : ""} aria-pressed={by === "classroom"} onClick={() => setBy("classroom")}>
            Classrooms
          </button>
          <button type="button" className={by === "bus_route" ? "on" : ""} aria-pressed={by === "bus_route"} onClick={() => setBy("bus_route")}>
            Bus routes
          </button>
        </div>
      }
    >
      {rows.length === 0 ? (
        <VizState height={300}>No absences logged yet this week.</VizState>
      ) : (
        <>
          <Heatmap
            ariaLabel={`Absences by ${by === "classroom" ? "classroom" : "bus route"} and day`}
            max={max}
            totalLabel="Week"
            rows={groups.map((g) => ({ key: g, label: g, sub: `${size(g)} students` }))}
            cols={days.map((d) => ({ key: d, label: dayParts(d)[0], sub: dayParts(d)[1]?.trim() }))}
            cells={cells}
            rowTotals={cells.map((r) => r.reduce((a, c) => a + c.value, 0))}
          />
          <HeatScale max={max} flaggedLabel="outlined = part of a flagged cluster" />
        </>
      )}
    </Panel>
  );
}

function SchoolNetwork({ students, rows, clusters }: { students: Student[]; rows: QueueRow[]; clusters: Cluster[] }) {
  const nav = useNavigate();
  // Each student's most recent absence this week.
  const latest = useMemo(() => {
    const m = new Map<string, QueueRow>();
    [...rows].sort((a, b) => a.date.localeCompare(b.date)).forEach((r) => m.set(r.student_id, r));
    return m;
  }, [rows]);
  const daysOut = useMemo(() => {
    const m = new Map<string, number>();
    rows.forEach((r) => m.set(r.student_id, (m.get(r.student_id) ?? 0) + 1));
    return m;
  }, [rows]);

  const nodes: NetNode[] = useMemo(() => {
    const household = new Map<string, number>();
    students.forEach((s) => household.set(s.household_id, (household.get(s.household_id) ?? 0) + 1));
    return students.map((s) => {
      const r = latest.get(s.id);
      const reason = r ? reasonOf(r) : null;
      const name = `${s.first_name} ${s.last_name}`;
      const cluster = clusters.find((c) => c.student_ids.includes(s.id));
      const where = `${s.classroom} · ${s.bus_route ?? "no bus"} · grade ${s.grade}`;
      return {
        id: s.id,
        groups: {
          classroom: s.classroom,
          bus_route: s.bus_route,
          household: (household.get(s.household_id) ?? 0) > 1 ? s.household_id : null,
        },
        cls: reason ? `barrier-${reason}` : undefined,
        glyph: reason ? GLYPH[reason] : undefined,
        href: !!r,
        legend: reason ?? undefined,
        aria: r ? `${name}, ${where}. Absent ${fmtDate(r.date)}: ${NAME[reason!]}. Open the case.` : `${name}, ${where}. Present.`,
        tip: (
          <Tip
            title={name}
            rows={
              r
                ? [
                    { cls: `barrier-${reason}`, label: NAME[reason!], value: fmtDate(r.date) },
                    { label: "Days out this week", value: daysOut.get(s.id) ?? 1 },
                  ]
                : [{ label: "Present all week", value: "" }]
            }
            foot={
              <>
                {where}
                {cluster && (
                  <>
                    <br />
                    In the {cluster.link_value} cluster
                  </>
                )}
                {r && (
                  <>
                    <br />
                    Click to open the case
                  </>
                )}
              </>
            }
          />
        ),
      };
    });
  }, [students, latest, daysOut, clusters]);

  const legend = ORDER.map((k) => ({ key: k, label: NAME[k], cls: `barrier-${k}`, glyph: GLYPH[k], count: nodes.filter((n) => n.legend === k).length })).filter((l) => l.count);
  const hulls: NetHull[] = clusters.map((c) => ({
    key: c.id,
    ids: c.student_ids,
    cls: `barrier-${short(c.likely_cause ?? c.barrier)}`,
    label: `${c.link_value} · ${c.likely_cause ?? c.barrier}`,
  }));

  return (
    <NetworkGraph
      nodes={nodes}
      hulls={hulls}
      legend={legend}
      initial={["classroom", "household"]}
      layers={[
        { key: "classroom", label: "Classroom" },
        { key: "bus_route", label: "Bus route" },
        { key: "household", label: "Household" },
      ]}
      onOpen={(id) => {
        const r = latest.get(id);
        if (r) nav(`/cases/${r.id}`);
      }}
      ariaLabel={`School network: ${students.length} students, ${latest.size} absent this week. Absent students are links to their case.`}
    />
  );
}

export default function Overview() {
  const queue = useLoad(api.queue);
  const clusterLoad = useLoad(api.clusters);
  const revenue = useLoad(api.revenue);
  const students = useLoad(api.students);
  const escalations = useLoad(api.escalations);

  const rows = queue.data ?? [];
  const total = rows.length;
  const replied = rows.filter((r) => r.reply).length;
  const captured = rows.filter((r) => r.status === "reason captured").length;
  const awaiting = rows.filter((r) => r.status === "awaiting reply").length;
  const openSafety = (escalations.data ?? []).filter((e) => e.status !== "closed").length;
  const clusters = clusterLoad.data ?? [];
  const linked = rows.filter((r) => clusters.some((c) => c.absence_ids.includes(r.id))).length;
  const loading = queue.data === null;

  const funnel: BarRow[] = (
    [
      ["Absences logged", total],
      ["Family replied", replied],
      ["Reason captured", captured],
      ["Linked to a root cause", linked],
    ] as [string, number][]
  ).map(([label, value], i, all) => {
    const share = total ? Math.round((value / total) * 100) : 0;
    const prev = i > 0 ? all[i - 1][1] : 0;
    return {
      key: label,
      label,
      value,
      cls: "series-brand",
      display: (
        <>
          {value}
          <small>{share}%</small>
        </>
      ),
      aria: `${label}: ${value}, ${share}% of absences`,
      tip: (
        <Tip
          title={label}
          rows={[
            { label: "Absences", value },
            { label: "Of all logged", value: `${share}%` },
            ...(i > 0 && prev ? [{ label: "Of the step before", value: `${Math.round((value / prev) * 100)}%` }] : []),
          ]}
        />
      ),
    };
  });

  return (
    <Page
      title="Overview"
      subtitle="This week at a glance: who is out, why, and which absences share a cause."
      actions={
        <>
          <Link className="button ghost" to="/queue">
            Open case queue
          </Link>
          <Link className="button" to="/clusters">
            Review root causes
          </Link>
        </>
      }
    >
      <ErrorNote error={queue.error} />
      <div className="kpis">
        <div className="kpi">
          <span>Absences this week</span>
          <b>{total}</b>
          <em>{students.data ? `${new Set(rows.map((r) => r.student_id)).size} of ${students.data.length} students` : " "}</em>
        </div>
        <div className="kpi">
          <span>Reasons captured</span>
          <b>{total ? Math.round((captured / total) * 100) : 0}%</b>
          <div className="meter" role="img" aria-label={`${captured} of ${total} absences have a reason`}>
            <i style={{ width: `${total ? (captured / total) * 100 : 0}%` }} />
          </div>
          <em>
            {captured} of {total} absences
          </em>
        </div>
        <div className="kpi">
          <span>Awaiting a reply</span>
          <b className={awaiting ? "warn" : ""}>{awaiting}</b>
          <em>families not heard from yet</em>
        </div>
        <div className="kpi">
          <span>Open safety cases</span>
          <b className={openSafety ? "bad" : ""}>{openSafety}</b>
          <em>{openSafety ? <Link to="/escalations">with a counselor →</Link> : "none open"}</em>
        </div>
        <div className="kpi">
          <span>Funding at risk</span>
          <b>{revenue.data ? money(revenue.data.dollars_at_risk_week) : "–"}</b>
          <em>{revenue.data ? `${money(revenue.data.dollars_recovered_to_date)} recovered to date` : " "}</em>
        </div>
      </div>

      <div className="dash">
        <Panel
          className="span2"
          title="School network"
          note="Each dot is a student. A line joins two students who share a link. Colored and lettered = absent this week, by reason."
        >
          {students.data === null ? (
            <VizState height={520} />
          ) : students.data.length === 0 ? (
            <VizState height={520}>No students loaded.</VizState>
          ) : (
            <SchoolNetwork students={students.data} rows={rows} clusters={clusters} />
          )}
        </Panel>

        <div className="stack">
          <Panel
            title="Root causes found"
            actions={
              <Link to="/clusters" className="small">
                All →
              </Link>
            }
          >
            {clusterLoad.data === null ? (
              <VizState height={200} />
            ) : (
              clusters.length === 0 && (
                <VizState height={200}>No shared cause detected yet. A cluster appears when several absences share a reason and a group.</VizState>
              )
            )}
            {clusters.map((c) => {
              const observed = c.observed ?? c.student_ids.length;
              const cause = c.likely_cause ?? c.barrier;
              return (
                <Link key={c.id} to="/clusters" className="cause">
                  <div className="row between">
                    <b>{c.link_value}</b>
                    <span className={`tag barrier-${short(cause)}`}>{cause}</span>
                  </div>
                  {c.expected != null && <Bullet observed={observed} expected={c.expected} cls={`barrier-${short(cause)}`} compact />}
                  <div className="row between small muted">
                    <span>
                      {c.student_ids.length} students{c.p_value != null ? ` · p = ${fmtP(c.p_value)}` : ""}
                    </span>
                    <span className="money">{money(c.dollars_at_risk_week)}</span>
                  </div>
                </Link>
              );
            })}
            {clusters.length > 0 && (
              <Legend
                items={[
                  { label: "Observed absent", cls: "series-muted" },
                  { label: "Expected", kind: "tick" },
                ]}
              />
            )}
          </Panel>
          <Panel title="Absences by day" note="How many were out, and how much we know.">
            {loading ? <VizState height={270} /> : total === 0 ? <VizState height={270}>No absences logged yet this week.</VizState> : <AbsencesByDay rows={rows} />}
          </Panel>
        </div>

        <div className="span2">
          <WhereAndWhen students={students.data ?? []} rows={rows} clusters={clusters} />
        </div>

        <Panel title="Why students were out" note="Absences this week by reason, largest first.">
          {loading ? <VizState height={300} /> : total === 0 ? <VizState height={300}>No absences logged yet this week.</VizState> : <WhyOut rows={rows} />}
        </Panel>

        <Panel
          className="span3"
          title="From absence to answer"
          note="Each step counts the absences that got that far. Bars share one scale: the first row is 100%."
        >
          {loading ? (
            <VizState height={150} />
          ) : total === 0 ? (
            <VizState height={150}>No absences logged yet this week.</VizState>
          ) : (
            <BarList rows={funnel} max={total} track ariaLabel="Absences at each step from logged to linked to a root cause" />
          )}
        </Panel>
      </div>
    </Page>
  );
}
