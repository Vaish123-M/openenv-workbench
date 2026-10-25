import React from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { formatNumber, formatPercent } from "./format.js";

const API_BASE = import.meta.env.VITE_API_BASE || "";

async function getJson(path, credentials) {
  const headers = credentials?.username
    ? { Authorization: `Basic ${btoa(`${credentials.username}:${credentials.password}`)}` }
    : {};
  const response = await fetch(`${API_BASE}${path}`, { headers });
  if (!response.ok) {
    const error = new Error(`Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

async function postJson(path, body, credentials) {
  const headers = {
    "Content-Type": "application/json",
    ...(credentials?.username ? { Authorization: `Basic ${btoa(`${credentials.username}:${credentials.password}`)}` } : {}),
  };
  const response = await fetch(`${API_BASE}${path}`, { method: "POST", headers, body: JSON.stringify(body) });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    const error = new Error(detail?.detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

function MetricCard({ label, value, tone = "" }) {
  return (
    <article className={`metric-card ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function scoreFor(record) {
  return record.result?.grading?.score || 0;
}

function downloadCsv(records) {
  const headers = ["run_id", "agent", "model", "environment", "task", "difficulty", "completed", "score", "steps", "elapsed_seconds", "termination"];
  const rows = records.map((record) => [
    record.result?.run_id,
    record.agent,
    record.model_name || "",
    record.environment,
    record.task_id,
    record.difficulty || "",
    record.result?.completed,
    scoreFor(record),
    record.result?.steps || 0,
    record.result?.metrics?.elapsed_time || 0,
    record.result?.termination_reason || "",
  ]);
  const csv = [headers, ...rows].map((row) => row.map((value) => `"${String(value ?? "").replaceAll('"', '""')}"`).join(",")).join("\n");
  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `openenv-runs-${new Date().toISOString().slice(0, 10)}.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function TrendChart({ records }) {
  const points = records.slice().reverse().slice(-12);
  if (points.length < 2) return <EmptyState>Run at least two evaluations to see a quality trend.</EmptyState>;
  const width = 760;
  const height = 180;
  const chartPoints = points.map((record, index) => {
    const x = 18 + (index / (points.length - 1)) * (width - 36);
    const y = height - 24 - scoreFor(record) * (height - 48);
    return `${x},${y}`;
  }).join(" ");
  return <div className="trend-chart">
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Evaluation score trend">
      <defs><linearGradient id="trend-fill" x1="0" x2="0" y1="0" y2="1"><stop offset="0%" stopColor="#43d6a2" stopOpacity=".28" /><stop offset="100%" stopColor="#43d6a2" stopOpacity="0" /></linearGradient></defs>
      {[0, .5, 1].map((value) => <line key={value} x1="18" x2={width - 18} y1={height - 24 - value * (height - 48)} y2={height - 24 - value * (height - 48)} className="chart-grid" />)}
      <polyline points={`18,${height - 24} ${chartPoints} ${width - 18},${height - 24}`} className="chart-area" />
      <polyline points={chartPoints} className="chart-line" />
      {points.map((record, index) => {
        const [x, y] = chartPoints.split(" ")[index].split(",");
        return <circle key={record.result?.run_id} cx={x} cy={y} r="4" className="chart-point"><title>{`${record.task_id}: ${formatPercent(scoreFor(record))}`}</title></circle>;
      })}
    </svg>
    <div className="chart-labels"><span>Older runs</span><strong>Score signal</strong><span>Latest runs</span></div>
  </div>;
}

function QualitySignal({ records }) {
  const total = records.length;
  const successful = records.filter((record) => record.result?.completed).length;
  const successRate = total ? successful / total : 0;
  const avgScore = total ? records.reduce((sum, record) => sum + scoreFor(record), 0) / total : 0;
  const signal = successRate >= .8 && avgScore >= .8 ? "Strong and reliable" : successRate >= .5 ? "Promising, needs iteration" : "Needs investigation";
  const tone = successRate >= .8 && avgScore >= .8 ? "positive" : successRate >= .5 ? "neutral" : "negative";
  return <div className={`quality-signal ${tone}`}>
    <div className="signal-ring"><strong>{formatPercent(avgScore)}</strong><span>quality</span></div>
    <div><p className="eyebrow">Quality signal</p><h3>{signal}</h3><p>{successful} of {total} evaluations completed successfully. Use the run explorer below to inspect the exact trajectory behind every result.</p></div>
    <div className="signal-meta"><span>Reliability</span><strong>{formatPercent(successRate)}</strong></div>
  </div>;
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

function RunEvaluation({ credentials, onComplete }) {
  const [form, setForm] = React.useState({
    environment: "openenv",
    task: "email_classification",
    agent: "mock",
    max_steps: 10,
    timeout: "",
    actions: '[\n  {\n    "content": "{\\"label\\": \\"spam\\", \\"reason\\": \\"Verify suspension card click immediately form.\\}",\n    "format": "json"\n  }\n]',
  });
  const [runStatus, setRunStatus] = React.useState("idle");
  const [runError, setRunError] = React.useState("");

  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));
  const submit = async (event) => {
    event.preventDefault();
    setRunStatus("running");
    setRunError("");
    try {
      let actions;
      if (form.agent === "mock") {
        try {
          actions = JSON.parse(form.actions);
        } catch {
          throw new Error("Actions must be valid JSON.");
        }
        if (!Array.isArray(actions) || !actions.length) throw new Error("Add at least one mock action.");
      }
      const payload = {
        environment: form.environment,
        task: form.task,
        agent: form.agent,
        max_steps: Number(form.max_steps),
        ...(form.timeout ? { timeout: Number(form.timeout) } : {}),
        ...(actions ? { actions } : {}),
      };
      await postJson("/runs", payload, credentials);
      setRunStatus("success");
      onComplete();
    } catch (error) {
      setRunStatus("error");
      setRunError(error.message);
    }
  };
  return <section className="panel run-panel">
    <div className="section-heading"><div><p className="eyebrow">Execute</p><h2>Run a new evaluation</h2><p className="section-help">Use the same controlled runner as the CLI, then inspect the result in the dashboard.</p></div><span className={`run-state ${runStatus}`}>{runStatus === "running" ? "● running" : runStatus === "success" ? "✓ completed" : "● ready"}</span></div>
    <form className="run-form" onSubmit={submit}>
      <label>Environment<select value={form.environment} onChange={(event) => update("environment", event.target.value)}><option value="openenv">OpenEnv tasks</option><option value="coding">Coding tasks</option></select></label>
      <label>Task<select value={form.task} onChange={(event) => update("task", event.target.value)}><option value="email_classification">Email classification</option><option value="data_cleaning">Data cleaning</option><option value="customer_support_reply">Customer support reply</option><option value="fix_addition">Fix addition</option><option value="fix_email_normalization">Fix email normalization</option><option value="fix_safe_division">Fix safe division</option></select></label>
      <label>Agent<select value={form.agent} onChange={(event) => update("agent", event.target.value)}><option value="mock">Mock agent (deterministic)</option><option value="openai">OpenAI agent</option></select></label>
      <label>Max steps<input type="number" min="1" max="100" value={form.max_steps} onChange={(event) => update("max_steps", event.target.value)} /></label>
      <label>Timeout (seconds)<input type="number" min="0.1" step="0.1" placeholder="Optional" value={form.timeout} onChange={(event) => update("timeout", event.target.value)} /></label>
      {form.agent === "mock" && <label className="action-editor">Mock actions (JSON)<textarea value={form.actions} onChange={(event) => update("actions", event.target.value)} spellCheck="false" /></label>}
      <div className="run-form-footer"><span className="form-hint">Results are persisted to SQLite and appear in history automatically.</span><button className="refresh-button" type="submit" disabled={runStatus === "running"}>{runStatus === "running" ? "Running evaluation…" : "▶ Run evaluation"}</button></div>
    </form>
    {runError && <p className="error-text form-error">{runError}</p>}
  </section>;
}

function Summary({ records }) {
  const total = records.length;
  const successful = records.filter((record) => record.result?.completed).length;
  const timeouts = records.filter((record) => record.result?.metrics?.timed_out).length;
  const score = records.reduce((sum, record) => sum + scoreFor(record), 0);
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

function benchmarkStats(records, benchmarkId) {
  const rows = records.filter((record) => record.benchmark_run_id === benchmarkId);
  const total = rows.length;
  return {
    total,
    success: total ? rows.filter((record) => record.result?.completed).length / total : 0,
    score: total ? rows.reduce((sum, record) => sum + scoreFor(record), 0) / total : 0,
    steps: total ? rows.reduce((sum, record) => sum + (record.result?.steps || 0), 0) / total : 0,
    time: total ? rows.reduce((sum, record) => sum + (record.result?.metrics?.elapsed_time || 0), 0) / total : 0,
    failures: rows.reduce((sum, record) => sum + (record.result?.failure?.categories?.length || 0), 0),
  };
}

function Delta({ value, inverse = false, formatter = formatPercent }) {
  const improved = inverse ? value < 0 : value > 0;
  return <span className={`delta ${value === 0 ? "flat" : improved ? "improved" : "regressed"}`}>{value > 0 ? "+" : ""}{formatter(value)}</span>;
}

function ExperimentComparison({ benchmarks, records }) {
  const available = benchmarks.filter((benchmark) => records.some((record) => record.benchmark_run_id === benchmark.benchmark_run_id));
  const [baseline, setBaseline] = React.useState("");
  const [candidate, setCandidate] = React.useState("");
  React.useEffect(() => {
    if (available.length >= 2) {
      setCandidate((current) => current || available[0].benchmark_run_id);
      setBaseline((current) => current || available[1].benchmark_run_id);
    }
  }, [available.length, available[0]?.benchmark_run_id, available[1]?.benchmark_run_id]);
  if (available.length < 2) return null;
  const left = benchmarkStats(records, baseline);
  const right = benchmarkStats(records, candidate);
  const scoreDelta = right.score - left.score;
  const successDelta = right.success - left.success;
  const regression = scoreDelta < -.05 || successDelta < -.1;
  return <section className="panel comparison-panel">
    <div className="section-heading"><div><p className="eyebrow">Experiment review</p><h2>Compare benchmark runs</h2><p className="section-help">Quantify whether a new change improved agent quality or introduced a regression.</p></div><span className={`comparison-badge ${regression ? "regression" : "healthy"}`}>{regression ? "⚠ regression detected" : "✓ no regression"}</span></div>
    <div className="comparison-selectors">
      <label>Baseline<select value={baseline} onChange={(event) => setBaseline(event.target.value)}>{available.map((item) => <option key={item.benchmark_run_id} value={item.benchmark_run_id}>{item.benchmark_run_id.slice(0, 10)} · {item.kind}</option>)}</select></label>
      <span className="versus">vs</span>
      <label>Candidate<select value={candidate} onChange={(event) => setCandidate(event.target.value)}>{available.map((item) => <option key={item.benchmark_run_id} value={item.benchmark_run_id}>{item.benchmark_run_id.slice(0, 10)} · {item.kind}</option>)}</select></label>
    </div>
    <div className="comparison-grid">
      <div><span>Average score</span><strong>{formatNumber(left.score)} → {formatNumber(right.score)}</strong><Delta value={scoreDelta} /></div>
      <div><span>Success rate</span><strong>{formatPercent(left.success)} → {formatPercent(right.success)}</strong><Delta value={successDelta} /></div>
      <div><span>Average steps</span><strong>{formatNumber(left.steps, 1)} → {formatNumber(right.steps, 1)}</strong><Delta value={right.steps - left.steps} inverse formatter={(value) => formatNumber(value, 1)} /></div>
      <div><span>Execution time</span><strong>{formatNumber(left.time, 3)}s → {formatNumber(right.time, 3)}s</strong><Delta value={right.time - left.time} inverse formatter={(value) => `${formatNumber(value, 3)}s`} /></div>
    </div>
    <p className="comparison-footnote">{right.total} candidate runs vs {left.total} baseline runs · {right.failures} candidate failure categories recorded</p>
  </section>;
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
            <td>{formatNumber(rows.reduce((sum, row) => sum + scoreFor(row), 0) / rows.length)}</td>
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
          <td>{formatNumber(group.records.reduce((sum, row) => sum + scoreFor(row), 0) / group.records.length)}</td>
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
  const [credentials, setCredentials] = React.useState(null);
  const [login, setLogin] = React.useState({ username: "", password: "" });
  const [query, setQuery] = React.useState("");
  const [darkMode, setDarkMode] = React.useState(() => localStorage.getItem("openenv-theme") === "dark");

  const load = React.useCallback(async () => {
    setStatus("loading");
    try {
      const [runData, benchmarkData] = await Promise.all([getJson("/results/runs?limit=1000", credentials), getJson("/results/benchmarks?limit=100", credentials)]);
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
  }, [credentials]);

  React.useEffect(() => { load(); }, [load]);
  React.useEffect(() => {
    document.documentElement.dataset.theme = darkMode ? "dark" : "light";
    localStorage.setItem("openenv-theme", darkMode ? "dark" : "light");
  }, [darkMode]);
  React.useEffect(() => {
    const onKeyDown = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        document.querySelector("#run-search")?.focus();
      }
      if (event.key === "Escape") document.querySelector("#run-search")?.blur();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
  const updateFilter = (key, value) => setFilters((current) => ({ ...current, [key]: value }));
  const filtered = records.filter((record) =>
    (!filters.agent || record.agent === filters.agent) &&
    (!filters.environment || record.environment === filters.environment) &&
    (!filters.task || record.task_id === filters.task) &&
    (!filters.difficulty || record.difficulty === filters.difficulty) &&
    (!filters.benchmark_run || record.benchmark_run_id === filters.benchmark_run) &&
    (!query || [record.task_id, record.agent, record.environment, record.model_name].filter(Boolean).join(" ").toLowerCase().includes(query.toLowerCase()))
  );

  if (status === "error" && error.includes("(401)")) {
    return <main className="app-shell login-shell"><section className="login-card">
      <p className="eyebrow">OpenEnv Workbench</p><h1>Sign in to evaluations</h1>
      <p className="login-help">Use the API credentials configured for this deployment. They remain in memory only.</p>
      <form onSubmit={(event) => { event.preventDefault(); setCredentials(login); }}>
        <label>Username<input value={login.username} onChange={(event) => setLogin({ ...login, username: event.target.value })} autoComplete="username" /></label>
        <label>Password<input type="password" value={login.password} onChange={(event) => setLogin({ ...login, password: event.target.value })} autoComplete="current-password" /></label>
        <button className="refresh-button" type="submit">Sign in</button>
      </form>
      <p className="error-text">{error}</p>
    </section></main>;
  }
  return <main className="app-shell">
    <header className="topbar"><div className="brand-lockup"><div className="brand-mark">O</div><div><p className="eyebrow">OpenEnv Workbench <span className="live-dot">● live</span></p><h1>Evaluation observatory</h1><p className="subtitle">Make every agent decision measurable.</p></div></div><div className="top-actions"><button className="icon-button" aria-label="Toggle dark mode" onClick={() => setDarkMode((value) => !value)}>{darkMode ? "☀" : "◐"}</button><button className="refresh-button" onClick={load}>↻ Refresh</button></div></header>
    {status === "loading" && <div className="state-panel">Loading evaluation results…</div>}
    {status === "error" && <div className="state-panel error-state">Could not load results: {error}<button onClick={load}>Retry</button></div>}
    {status === "ready" && <>
      <RunEvaluation credentials={credentials} onComplete={load} />
      <QualitySignal records={filtered} />
      <ExperimentComparison benchmarks={benchmarks} records={records} />
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">Overview</p><h2>Benchmark summary</h2></div><span className="run-count">{filtered.length} runs</span></div><Summary records={filtered} /></section>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">Signal over time</p><h2>Quality trend</h2></div><span className="chart-legend"><i /> score</span></div><TrendChart records={filtered} /></section>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">Compare</p><h2>Agent and model performance</h2></div></div><ComparisonTable records={filtered} /></section>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">Coverage</p><h2>Environment and task performance</h2></div></div><PerformanceTable records={filtered} /></section>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">History</p><h2>Benchmark runs</h2></div></div><Filters values={filters} onChange={updateFilter} benchmarks={benchmarks} /><div className="history">{benchmarks.map((benchmark) => <div className="history-row" key={benchmark.benchmark_run_id}><span>{new Date(benchmark.created_at).toLocaleString()}</span><code>{benchmark.benchmark_run_id.slice(0, 12)}</code><span>{benchmark.kind}</span><strong>{benchmark.summary?.success_rate !== undefined ? formatPercent(benchmark.summary.success_rate) : "Comparison"}</strong></div>)}</div></section>
      <section className="panel"><div className="section-heading"><div><p className="eyebrow">Runs</p><h2>Individual evaluations</h2></div><button className="export-button" onClick={() => downloadCsv(filtered)} disabled={!filtered.length}>↓ Export CSV</button></div><div className="search-row"><label className="search-box"><span>⌕</span><input id="run-search" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search task, agent, or environment" /><kbd>Ctrl K</kbd></label><span className="run-count">{filtered.length} matching</span></div>{filtered.length ? <div className="run-list">{filtered.map((record) => <button className="run-row" key={record.result.run_id} onClick={() => setSelected(record)}><span className="run-index">#{String(record.result.run_id).slice(0, 6)}</span><span><strong>{record.task_id || "Untitled task"}</strong><small>{record.environment} · {record.agent}</small></span><span className="run-score">{formatNumber(record.result.grading?.score)}</span><span className={record.result.completed ? "status-success" : "status-failure"}>{record.result.termination_reason}</span><span className="row-arrow">→</span></button>)}</div> : <EmptyState />}</section>
    </>}
    {selected && <RunDetails record={selected} onClose={() => setSelected(null)} />}
  </main>;
}

createRoot(document.getElementById("root")).render(<App />);
