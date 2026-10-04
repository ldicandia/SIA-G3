"use strict";

// Digit Lab: form, polling, canvas charts, sortable table and TOML export.
// All server-provided text is written through textContent / setAttribute on
// elements built with createElement; no HTML is ever assembled from strings.

const OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#999999"];
const POLL_ACTIVE_MS = 1000;
const POLL_IDLE_MS = 4000;

const state = {
  defaults: null,
  runs: [],
  selected: new Set(),
  sort: { key: "id", dir: "desc" },
  metrics: new Map(),
  finished: new Set(),
  colors: new Map(),
  lastGroup: null,
  detailId: null,
  timer: null,
};

function el(tag, attrs, text) {
  const node = document.createElement(tag);
  if (attrs) {
    for (const [name, value] of Object.entries(attrs)) {
      node.setAttribute(name, value);
    }
  }
  if (text !== undefined && text !== null) {
    node.textContent = String(text);
  }
  return node;
}

function $(id) {
  return document.getElementById(id);
}

function showError(message) {
  const box = $("alert");
  box.textContent = message;
  box.hidden = false;
}

function clearError() {
  const box = $("alert");
  box.textContent = "";
  box.hidden = true;
}

async function getJson(url) {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    throw new Error(url + " answered " + response.status);
  }
  return response.json();
}

// ---------------------------------------------------------------- form

function formFields() {
  return Array.from(document.querySelectorAll("[data-field]"));
}

function fillForm(defaults) {
  for (const control of formFields()) {
    const key = control.dataset.field;
    let value;
    if (control.dataset.scope === "study") {
      value = key === "exercise" ? defaults.exercises[0] : defaults[key];
    } else {
      value = defaults.candidate[key];
    }
    control.value = Array.isArray(value) ? value.join(",") : String(value);
  }
}

function readNumber(raw) {
  return raw === "" ? null : Number(raw);
}

function readForm() {
  const candidate = {};
  const request = {};
  for (const control of formFields()) {
    const key = control.dataset.field;
    const raw = control.value.trim();
    let value;
    if (control.tagName === "SELECT") {
      value = raw;
    } else if (control.dataset.kind === "list") {
      value = raw === "" ? [] : raw.split(",").map((part) => readNumber(part.trim()));
    } else if (control.type === "number") {
      value = readNumber(raw);
    } else {
      value = raw;
    }
    if (control.dataset.scope === "study") {
      request[key] = value;
    } else {
      candidate[key] = value;
    }
  }
  request.candidate = candidate;
  const values = $("sweep-values")
    .value.split(/[;\n]+/)
    .map((part) => part.trim())
    .filter((part) => part !== "");
  if (values.length > 0) {
    request.sweep = { field: $("sweep-field").value, values };
    request.parallel = readNumber($("sweep-parallel").value.trim());
  }
  return request;
}

function setupSweep(defaults) {
  const select = $("sweep-field");
  for (const field of defaults.sweep_fields) {
    select.appendChild(el("option", { value: field }, field));
  }
  select.value = "learning_rate";
  const parallel = $("sweep-parallel");
  parallel.setAttribute("max", String(defaults.max_parallel));
  parallel.value = String(defaults.default_parallel);
  const update = () => {
    const field = select.value;
    let hint = "0.1; 0.01; 0.001";
    if (field === "topology") {
      hint = "784,64,10; 784,128,10";
    } else if (field === "optimizer") {
      hint = "adam; sgd; momentum";
    } else if (field === "hidden_activation") {
      hint = "relu; tanh; sigmoid";
    } else if (field === "initialization") {
      hint = "xavier; he";
    } else if (field === "batch_size" || field === "max_epochs" || field === "seed") {
      hint = "32; 64; 128";
    }
    $("sweep-values").setAttribute("placeholder", hint);
  };
  select.addEventListener("change", update);
  update();
}

async function onSubmit(event) {
  event.preventDefault();
  clearError();
  const button = $("run-button");
  button.disabled = true;
  try {
    const response = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readForm()),
    });
    const data = await response.json().catch(() => ({ error: "unreadable server answer" }));
    if (!response.ok) {
      showError(data.error || "request failed (" + response.status + ")");
      return;
    }
    if (data.group) {
      state.lastGroup = data.group;
    }
    for (const run of data.runs) {
      state.selected.add(run.id);
      assignColor(run.id);
    }
    state.detailId = data.runs.length > 0 ? data.runs[0].id : state.detailId;
    schedulePoll(0);
  } catch (error) {
    showError("Could not reach the server: " + error.message);
  } finally {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------- table

function seqOf(run) {
  const parts = run.id.split("-");
  return Number(parts[0]) * 1e6 + Number(parts[1]);
}

function maxEpochs(run) {
  return run.candidate.max_epochs;
}

function progressValue(run) {
  const done = run.result ? run.result.epochs_run : run.last_epoch;
  return done === null || done === undefined ? null : done / maxEpochs(run);
}

function progressText(run) {
  const done = run.result ? run.result.epochs_run : run.last_epoch;
  return (done === null || done === undefined ? 0 : done) + "/" + maxEpochs(run);
}

function formatNumber(value, digits) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "";
  }
  return value.toFixed(digits);
}

function result(run, key) {
  return run.result ? run.result[key] : null;
}

const COLUMNS = [
  { key: "select", label: "Chart", sortable: false },
  { key: "id", label: "id", value: seqOf, text: (r) => r.id },
  { key: "name", label: "name", value: (r) => r.name, text: (r) => r.name },
  { key: "exercise", label: "exercise", value: (r) => r.exercise, text: (r) => r.exercise },
  { key: "status", label: "status", value: (r) => r.status, text: (r) => r.status, cls: (r) => "status-" + r.status },
  { key: "progress", label: "progress", value: progressValue, text: progressText, num: true },
  { key: "val_acc", label: "val acc", value: (r) => result(r, "validation_accuracy"), text: (r) => formatNumber(result(r, "validation_accuracy"), 4), num: true },
  { key: "val_loss", label: "val loss", value: (r) => result(r, "validation_loss"), text: (r) => formatNumber(result(r, "validation_loss"), 4), num: true },
  { key: "train_acc", label: "train acc @best", value: (r) => result(r, "train_accuracy_at_best"), text: (r) => formatNumber(result(r, "train_accuracy_at_best"), 4), num: true },
  { key: "best_epoch", label: "best epoch", value: (r) => result(r, "best_epoch"), text: (r) => formatNumber(result(r, "best_epoch"), 0), num: true },
  { key: "epochs_run", label: "epochs", value: (r) => result(r, "epochs_run"), text: (r) => formatNumber(result(r, "epochs_run"), 0), num: true },
  { key: "seconds", label: "seconds", value: (r) => result(r, "seconds"), text: (r) => formatNumber(result(r, "seconds"), 1), num: true },
  { key: "optimizer", label: "optimizer", value: (r) => r.candidate.optimizer, text: (r) => r.candidate.optimizer },
  { key: "lr", label: "lr", value: (r) => r.candidate.learning_rate, text: (r) => String(r.candidate.learning_rate), num: true },
  { key: "batch", label: "batch", value: (r) => r.candidate.batch_size, text: (r) => String(r.candidate.batch_size), num: true },
  { key: "topology", label: "topology", value: (r) => r.candidate.topology.join("-"), text: (r) => r.candidate.topology.join("-") },
  { key: "config", label: "config", sortable: false },
];

function compareValues(a, b) {
  if (typeof a === "number" && typeof b === "number") {
    return a - b;
  }
  return String(a).localeCompare(String(b));
}

function sortedRuns() {
  const column = COLUMNS.find((c) => c.key === state.sort.key);
  const direction = state.sort.dir === "asc" ? 1 : -1;
  const rows = state.runs.slice();
  if (!column || !column.value) {
    return rows;
  }
  // Array.prototype.sort is stable; null values always go last.
  rows.sort((left, right) => {
    const a = column.value(left);
    const b = column.value(right);
    const aMissing = a === null || a === undefined;
    const bMissing = b === null || b === undefined;
    if (aMissing || bMissing) {
      return aMissing === bMissing ? 0 : aMissing ? 1 : -1;
    }
    return direction * compareValues(a, b);
  });
  return rows;
}

function renderHead() {
  const row = $("head-row");
  row.replaceChildren();
  for (const column of COLUMNS) {
    const th = el("th", { scope: "col" });
    if (column.sortable === false) {
      th.textContent = column.label;
    } else {
      const active = state.sort.key === column.key;
      th.setAttribute("aria-sort", active ? (state.sort.dir === "asc" ? "ascending" : "descending") : "none");
      const button = el("button", { type: "button" }, column.label);
      button.addEventListener("click", () => {
        if (state.sort.key === column.key) {
          state.sort.dir = state.sort.dir === "asc" ? "desc" : "asc";
        } else {
          state.sort = { key: column.key, dir: "asc" };
        }
        renderHead();
        renderTable();
      });
      th.appendChild(button);
    }
    row.appendChild(th);
  }
}

function toggleSelection(id) {
  if (state.selected.has(id)) {
    state.selected.delete(id);
  } else {
    state.selected.add(id);
    assignColor(id);
  }
  state.detailId = id;
  renderTable();
  refreshMetrics().then(drawCharts);
}

function renderTable() {
  const body = $("runs-body");
  body.replaceChildren();
  for (const run of sortedRuns()) {
    const row = el("tr");
    if (state.selected.has(run.id)) {
      row.classList.add("selected");
    }
    row.addEventListener("click", (event) => {
      if (event.target.closest("a, input")) {
        return;
      }
      toggleSelection(run.id);
    });
    for (const column of COLUMNS) {
      const cell = el("td");
      if (column.key === "select") {
        const box = el("input", { type: "checkbox", "aria-label": "Chart run " + run.id });
        box.checked = state.selected.has(run.id);
        box.addEventListener("change", () => toggleSelection(run.id));
        cell.appendChild(box);
      } else if (column.key === "config") {
        const link = el("a", { href: "/api/runs/" + run.id + "/config.toml", download: run.name + ".toml" }, "toml");
        cell.appendChild(link);
      } else {
        cell.textContent = column.text(run);
        if (column.num) {
          cell.classList.add("num");
        }
        if (column.cls) {
          cell.classList.add(column.cls(run));
        }
      }
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
  renderDetail();
  renderStatusLine();
}

function renderDetail() {
  const box = $("detail");
  box.replaceChildren();
  const run = state.runs.find((r) => r.id === state.detailId);
  if (!run) {
    return;
  }
  box.appendChild(el("div", null, "Run " + run.id + " (" + run.name + "): " + run.status));
  if (run.error) {
    box.appendChild(el("pre", null, run.error));
  }
}

function renderStatusLine() {
  const active = state.runs.filter((r) => r.status === "queued" || r.status === "running").length;
  const max = state.defaults ? state.defaults.max_parallel : "?";
  $("status-line").textContent =
    state.runs.length + " runs, " + active + " active, up to " + max + " trainings at the same time";
}

// ---------------------------------------------------------------- charts

function assignColor(id) {
  if (!state.colors.has(id)) {
    state.colors.set(id, state.colors.size % OKABE_ITO.length);
  }
}

async function refreshMetrics() {
  for (const id of state.selected) {
    const run = state.runs.find((r) => r.id === id);
    if (!run) {
      continue;
    }
    const finished = run.status === "done" || run.status === "failed";
    if (finished && state.finished.has(id)) {
      continue;
    }
    try {
      state.metrics.set(id, await getJson("/api/runs/" + id + "/metrics"));
      if (finished) {
        state.finished.add(id);
      }
    } catch (error) {
      // transient: the next tick retries
    }
  }
}

function cssColor(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function niceCeil(value) {
  if (!(value > 0)) {
    return 1;
  }
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  const scaled = value / magnitude;
  const step = scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 5 ? 5 : 10;
  return step * magnitude;
}

function selectedSeries() {
  const series = [];
  for (const id of state.selected) {
    const run = state.runs.find((r) => r.id === id);
    const points = state.metrics.get(id);
    if (run && points && points.length > 0) {
      series.push({ run, points, color: OKABE_ITO[state.colors.get(id) || 0] });
    }
  }
  return series;
}

function drawChart(canvas, series, trainKey, validationKey, kind) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.max(1, Math.round(width * ratio));
  canvas.height = Math.max(1, Math.round(height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  const text = cssColor("--text");
  const muted = cssColor("--muted");
  const line = cssColor("--line");
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";

  if (series.length === 0) {
    ctx.fillStyle = muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Select or launch a run to see its curves", width / 2, height / 2);
    return;
  }

  let maxEpoch = 2;
  let maxValue = 0;
  let minValue = 1;
  for (const item of series) {
    for (const point of item.points) {
      maxEpoch = Math.max(maxEpoch, point.epoch);
      for (const key of [trainKey, validationKey]) {
        const value = point[key];
        if (value !== null && value !== undefined) {
          maxValue = Math.max(maxValue, value);
          minValue = Math.min(minValue, value);
        }
      }
    }
  }
  let yMin = 0;
  let yMax;
  if (kind === "loss") {
    yMax = niceCeil(maxValue);
  } else {
    yMax = 1;
    yMin = Math.max(0, Math.floor(minValue * 20) / 20);
    if (yMin >= yMax) {
      yMin = 0;
    }
  }
  const left = 44;
  const right = 10;
  const top = 8;
  const bottom = 24;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const xOf = (epoch) => left + ((epoch - 1) / (maxEpoch - 1)) * plotWidth;
  const yOf = (value) => top + (1 - (value - yMin) / (yMax - yMin)) * plotHeight;

  ctx.strokeStyle = line;
  ctx.fillStyle = muted;
  ctx.lineWidth = 1;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i += 1) {
    const value = yMin + ((yMax - yMin) * i) / 4;
    const y = yOf(value);
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(left + plotWidth, y);
    ctx.stroke();
    ctx.fillText(value.toFixed(kind === "loss" ? 2 : 3), left - 6, y);
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const tickCount = Math.min(maxEpoch, 8);
  for (let i = 0; i < tickCount; i += 1) {
    const epoch = Math.round(1 + ((maxEpoch - 1) * i) / Math.max(1, tickCount - 1));
    ctx.fillText(String(epoch), xOf(epoch), top + plotHeight + 6);
  }
  ctx.fillStyle = text;
  ctx.textAlign = "right";
  ctx.fillText("epoch", width - right, height - 12);

  const drawLine = (item, key, dashed) => {
    ctx.strokeStyle = item.color;
    ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [6, 4] : []);
    ctx.beginPath();
    let started = false;
    for (const point of item.points) {
      const value = point[key];
      if (value === null || value === undefined) {
        continue;
      }
      const x = xOf(point.epoch);
      const y = yOf(value);
      if (started) {
        ctx.lineTo(x, y);
      } else {
        ctx.moveTo(x, y);
        started = true;
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);
    if (item.points.length === 1) {
      const value = item.points[0][key];
      if (value !== null && value !== undefined) {
        ctx.fillStyle = item.color;
        ctx.beginPath();
        ctx.arc(xOf(item.points[0].epoch), yOf(value), 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  };
  for (const item of series) {
    drawLine(item, trainKey, true);
    drawLine(item, validationKey, false);
  }
}

function describeLatest(series, label, trainKey, validationKey) {
  if (series.length === 0) {
    return label + " chart: no run selected";
  }
  const first = series[0];
  const last = first.points[first.points.length - 1];
  const fmt = (value) => (value === null || value === undefined ? "n/a" : value.toFixed(4));
  return (
    label + " chart. Run " + first.run.name + " at epoch " + last.epoch + ": train " +
    fmt(last[trainKey]) + ", validation " + fmt(last[validationKey])
  );
}

function renderLegend(series) {
  const legend = $("legend");
  legend.replaceChildren();
  if (series.length === 0) {
    return;
  }
  legend.appendChild(el("li", null, "solid = validation, dashed = train"));
  for (const item of series) {
    const entry = el("li");
    const swatch = el("span", { class: "swatch" });
    swatch.style.borderTopColor = item.color;
    entry.appendChild(swatch);
    entry.appendChild(document.createTextNode(item.run.name + " (" + item.run.id + ")"));
    legend.appendChild(entry);
  }
}

function drawCharts() {
  const series = selectedSeries();
  const loss = $("chart-loss");
  const accuracy = $("chart-accuracy");
  drawChart(loss, series, "train_loss", "validation_loss", "loss");
  drawChart(accuracy, series, "train_accuracy", "validation_accuracy", "accuracy");
  loss.setAttribute("aria-label", describeLatest(series, "Loss", "train_loss", "validation_loss"));
  accuracy.setAttribute(
    "aria-label",
    describeLatest(series, "Accuracy", "train_accuracy", "validation_accuracy"),
  );
  renderLegend(series);
}

// ---------------------------------------------------------------- polling

function schedulePoll(delay) {
  if (state.timer !== null) {
    clearTimeout(state.timer);
  }
  state.timer = setTimeout(tick, delay);
}

async function tick() {
  let delay = POLL_IDLE_MS;
  try {
    state.runs = await getJson("/api/runs");
    renderTable();
    await refreshMetrics();
    drawCharts();
    if (state.runs.some((r) => r.status === "queued" || r.status === "running")) {
      delay = POLL_ACTIVE_MS;
    }
  } catch (error) {
    $("status-line").textContent = "Server unreachable: " + error.message;
  }
  schedulePoll(delay);
}

// ---------------------------------------------------------------- export

async function downloadToml(url, fallbackName) {
  clearError();
  try {
    const response = await fetch(url);
    if (!response.ok) {
      const data = await response.json().catch(() => ({ error: "download failed" }));
      showError(data.error || "download failed");
      return;
    }
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = /filename="([^"]+)"/.exec(disposition);
    const link = el("a", { href: URL.createObjectURL(blob), download: match ? match[1] : fallbackName });
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (error) {
    showError("Could not download: " + error.message);
  }
}

function onExportBest() {
  const exercise = document.querySelector('[data-field="exercise"]').value;
  downloadToml("/api/runs/best/config.toml?exercise=" + encodeURIComponent(exercise), "best.toml");
}

function onExportBestSweep() {
  if (!state.lastGroup) {
    showError("Launch a sweep first: there is no last sweep in this session.");
    return;
  }
  downloadToml("/api/runs/best/config.toml?group=" + encodeURIComponent(state.lastGroup), "best-sweep.toml");
}

// ---------------------------------------------------------------- start

async function start() {
  renderHead();
  $("run-form").addEventListener("submit", onSubmit);
  $("export-best").addEventListener("click", onExportBest);
  $("export-best-sweep").addEventListener("click", onExportBestSweep);
  window.addEventListener("resize", drawCharts);
  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawCharts);
  }
  try {
    state.defaults = await getJson("/api/defaults");
    fillForm(state.defaults);
    setupSweep(state.defaults);
  } catch (error) {
    showError("Could not load the defaults: " + error.message);
  }
  drawCharts();
  tick();
}

start();
