import React from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE = import.meta.env.VITE_API_BASE || "";

async function getJson(path) {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed (${response.status})`);
  }
  return response.json();
}

const formatPercent = (value) => `${((value || 0) * 100).toFixed(1)}%`;
const formatNumber = (value, digits = 2) =>
  Number.isFinite(value) ? value.toFixed(digits) : "—";

function MetricCard({ label, value, tone = "" }) {
  return (
    <article className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function EmptyState({ children = "No evaluation data is available yet." }) {
  return <div className="empty-state">{children}</div>;
}

function Filters({ values, onChange, benchmarks }) {
  const fields = [
    ["agent", "Agent"],
    ["environment", "Environment"],
    ["task", "Task"],
    ["difficulty", "Difficulty"],
  ];
  return (
    <div className="filters">
      {fields.map(([key, label]) => (
        <label key={key}>
          {label}
          <select value={values[key]} onChange={(event) => onChange(key, event.target.value)}>
            <option value="">All</option>
            {[...new Set(values.options[key] || [])].map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
        </label>
      ))}
      <label>
        Benchmark run
        <select value={values.benchmark_run} onChange={(event) => onChange("benchmark_run", event.target.value)}>
          <option value="">All</option>
          {benchmarks.map((benchmark) => (
            <option key={benchmark.benchmark_run_id} value={benchmark.benchmark_run_id}>
              {benchmark.benchmark_run_id.slice(0, 10)}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

function Summary({ records }) {
  const total = records.length;
  const successful = records.filter((record) => record.result?.completed).length;
  const timeouts = records.filter((record) => record.result?.metrics?.timed_out).length;
  const score = records.reduce((sum, record) => sum + (record.result?.grading?.score || 0), 0);
  const steps = records.reduce((sum, record) => sum + (record.result?.steps || 0), 0);
  const elapsed = records.reduce((sum, record) => sum + (record.result?.metrics?.elapsed_time || 0), 0);
  return (
    <div className="metrics-grid">
      <MetricCard label="Success rate" value={formatPercent(total ? successful / total : 0)} tone="success" />
      <MetricCard label="Average score" value={formatNumber(total ? score / total : 0)} />
      <MetricCard label="Average steps" value={formatNumber(total ? steps / total : 0, 1)} />
      <MetricCard label="Avg execution time" value={`${formatNumber(total ? elapsed / total : 0, 3)}s`} />
      <MetricCard label="Failure rate" value={formatPercent(total ? (total - successful) / total : 0)} tone="warning" />
      <MetricCard label="Timeout rate" value={formatPercent(total ? timeouts / total : 0)} />
    </div>
  );
}

function ComparisonTable({ records }) {
  const groups = Object.values(records.reduce((result, record) => {
    const name = record.agent || "unknown";
    const group = result[name] || { name, records: [] };
    group.records.push(record);
    result[name] = group;
    return result;
  }, {}));
  if (!groups.length) return <EmptyState />;
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Agent / model</th><th>Success</th><th>Score</th><th>Avg steps</th><th>Time</th><th>Failures</th></tr></thead>
        <tbody>{groups.map(({ name, records: rows }) => {
          const success = rows.filter((row) => row.result?.completed).length;
          const failures = rows.length - success;
          return <tr key={name}>
            <td><strong>{name}</strong><small>{rows.find((row) => row.model_name)?.model_name || "model not reported"}</small></td>
            <td>{formatPercent(success / rows.length)}</td>
            <td>{formatNumber(rows.reduce((sum, row) => sum + (row.result?.grading?.score || 0), 0) / rows.length)}</td>
            <td>{formatNumber(rows.reduce((sum, row) => sum + (row.result?.steps || 0), 0) / rows.length, 1)}</td>
            <td>{formatNumber(rows.reduce((sum, row) => sum + (row.result?.metrics?.elapsed_time || 0), 0) / rows.length, 3)}s</td>
            <td>{failures}</td>
          </tr>;
        })}</tbody>
      </table>
    </div>
  );
}

function PerformanceTable({ records }) {
  const groups = Object.values(records.reduce((result, record) => {
    const key = `${record.environment}:${record.task_id}`;
    const group = result[key] || { environment: record.environment, task: record.task_id, records: [] };
    group.records.push(record);
    result[key] = group;
    return result;
  }, {}));
  if (!groups.length) return <EmptyState />;
  return (
    <div className="table-wrap">
      <table><thead><tr><th>Environment</th><th>Task</th><th>Runs</th><th>Success</th><th>Score</th></tr></thead>
        <tbody>{groups.map((group) => <tr key={`${group.environment}:${group.task}`}>
          <td>{group.environment}</td><td>{group.task}</td><td>{group.records.length}</td>
          <td>{formatPercent(group.records.filter((row) => row.result?.completed).length / group.records.length)}</td>
          <td>{formatNumber(group.records.reduce((sum, row) => sum + (row.result?.grading?.score || 0), 0) / group.records.length)}</td>
        </tr>)}</tbody>
      </table>
    </div>
  );
}

function RunDetails({ record, onClose }) {
  const result = record.result;
  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="drawer" onClick={(event) => event.stopPropagation()}>
        <button className="close-button" onClick={onClose}>Close</button>
        <p className="eyebrow">{record.environment} / {record.task_id}</p>
        <h2>{record.agent} <span>{record.model_name || "model not reported"}</span></h2>
        <div className="detail-grid">
          <div><span>Score</span><strong>{formatNumber(result.grading?.score)}</strong></div>
          <div><span>Steps</span><strong>{result.steps}</strong></div>
          <div><span>Termination</span><strong>{result.termination_reason}</strong></div>
          <div><span>Failure</span><strong>{result.failure?.primary_reason || "None"}</strong></div>
        </div>
        <h3>Metrics</h3>
        <pre>{JSON.stringify(result.metrics, null, 2)}</pre>
        <h3>Failure analysis</h3>
        {result.failure?.categories?.length ? <div className="tag-list">{result.failure.categories.map((category) => <span key={category}>{category}</span>)}</div> : <EmptyState>No failure categories.</EmptyState>}
        <h3>Trajectory</h3>
        <div className="trajectory">{(result.trajectory || []).map((step) => <div className="trajectory-step" key={step.step_number}>
          <div className="step-marker">{step.step_number}</div>
          <div><strong>Observation</strong><pre>{JSON.stringify(step.observation, null, 2)}</pre></div>
          <div><strong>Action</strong><pre>{JSON.stringify(step.action, null, 2)}</pre></div>
          <div><strong>Environment response</strong><pre>{JSON.stringify(step.environment_response, null, 2)}</pre></div>
          {step.error && <div className="error-text">{step.error.message}</div>}
        </div>)}</div>
      </aside>
    </div>
  );
}

function App() {
  const [records, setRecords] = React.useState([]);
  const [benchmarks, setBenchmarks] = React.useState([]);
  const [selected, setSelected] = React.useState(null);
  const [filters, setFilters] = React.useState({ agent: "", environment: "", task: "", difficulty: "", benchmark_run: "", options: {} });
  const [status, setStatus] = React.useState("loading");
  const [error, setError] = React.useState("");

  const load = React.useCallback(async () => {
    setStatus("loading");
    try {
      const [runData, benchmarkData] = await Promise.all([getJson("/results/runs?limit=1000"), getJson("/results/benchmarks?limit=100")]);
      setRecords(runData);
      setBenchmarks(benchmarkData);
      setFilters((current) => ({ ...current, options: {
        agent: runData.map((record) => record.agent),
        environment: runData.map((record) => record.environment),
        task: runData.map((record) => record.task_id),
        difficulty: runData.map((record) => record.difficulty).filter(Boolean),
      }}));
      setStatus("ready");
    } catch (loadError) {
      setError(loadError.message);
      setStatus("error");
    }
  }, []);

  React.useEffect(() => { load(); }, [load]);
  const updateFilter = (key, value) => setFilters((current) => ({ ...current, [key]: value }));
  const filtered = records.filter((record) =>
    (!filters.agent || record.agent === filters.agent) &&
    (!filters.environment || record.environment === filters.environment) &&
    (!filters.task || record.task_id === filters.task) &&
    (!filters.difficulty || record.difficulty === filters.difficulty) &&
    (!filters.benchmark_run || record.benchmark_run_id === filters.benchmark_run)
  );

  return <main className="app-shell">
    <header className="topbar"><div><p className="eyebrow">OpenEnv Workbench</p><h1>Evaluation dashboard</h1></div><button className="refresh-button" onClick={load}>Refresh data</button></header>
    {status === "loading" && <div className="state-panel">Loading evaluation results…</div>}
    {status === "error" && <div className="state-panel error-state">Could not load results: {error}<button onClick={load}>Retry</button></div>}
    {status === "ready" && <>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">Overview</p><h2>Benchmark summary</h2></div><span className="run-count">{filtered.length} runs</span></div><Summary records={filtered} /></section>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">Compare</p><h2>Agent and model performance</h2></div></div><ComparisonTable records={filtered} /></section>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">Coverage</p><h2>Environment and task performance</h2></div></div><PerformanceTable records={filtered} /></section>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">History</p><h2>Benchmark runs</h2></div></div><Filters values={filters} onChange={updateFilter} benchmarks={benchmarks} /><div className="history">{benchmarks.map((benchmark) => <div className="history-row" key={benchmark.benchmark_run_id}><span>{new Date(benchmark.created_at).toLocaleString()}</span><code>{benchmark.benchmark_run_id.slice(0, 12)}</code><span>{benchmark.kind}</span><strong>{benchmark.summary?.success_rate !== undefined ? formatPercent(benchmark.summary.success_rate) : "Comparison"}</strong></div>)}</div></section>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">Runs</p><h2>Individual evaluations</h2></div></div>{filtered.length ? <div className="run-list">{filtered.map((record) => <button className="run-row" key={record.result.run_id} onClick={() => setSelected(record)}><span><strong>{record.task_id || "Untitled task"}</strong><small>{record.environment} · {record.agent}</small></span><span>{formatNumber(record.result.grading?.score)}</span><span className={record.result.completed ? "status-success" : "status-failure"}>{record.result.termination_reason}</span></button>)}</div> : <EmptyState />}</section>
    </>}
    {selected && <RunDetails record={selected} onClose={() => setSelected(null)} />}
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
