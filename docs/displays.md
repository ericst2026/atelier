# The five wall displays

Open `http://<server>/display/1` … `/display/5` in a full-screen browser on each
screen. No sign-in: with `ATELIER_DISPLAYS_PUBLIC=true` (the default) a display
needs no credentials, which means no session to expire in the middle of a lesson.
Set it to `false` and append `?token=<teacher token>` if your room is less private.

Each display holds a websocket open. The **server** decides what it shows and
pushes a complete snapshot — every five seconds, and immediately when a run starts,
finishes, or a teacher changes a screen. Nothing on the display polls, and a
reconnect happens by itself after a network blip.

## Defaults

| # | Name | Mode | Shows |
|---|---|---|---|
| 1 | Hardware | `grafana` | the provisioned Grafana dashboard in kiosk mode: per-GPU utilization, memory, temperature, power, CPU, RAM, disk, network, queue depth |
| 2 | Live runs | `live` | the eight GPUs with the student on each, plus running and queued jobs with live progress |
| 3 | Class progress | `progress` | for each experiment, how many students are at step 0–4 and how many have submitted |
| 4 | Leaderboard | `leaderboard` | published submissions ranked by the experiment's metric |
| 5 | Spotlight | `message` | whatever the teacher casts: a run's charts, or a message |

## Changing a screen

Teacher → Displays. Pick a mode, fill its payload, press **Push to display N** —
the screen changes immediately.

- `run` takes `{"run_id": 123}` and mirrors that run: live charts while it trains,
  the full metrics and charts when it finishes. Good for walking the room through
  one student's loss curve.
- `leaderboard` takes an optional `{"experiment": "tokenizer"}`.
- `message` takes `{"title": "…", "text": "…"}` — the lesson plan, a break, a hint.
- `grafana` takes an optional `{"url": "…"}` if you build your own dashboard.

Only submissions marked **published** appear on the leaderboard, so the screen
stays a celebration rather than a ranking nobody agreed to.

## Grafana

The dashboard is provisioned from `deploy/grafana/dashboards/hardware.json` (uid
`atelier-hw`) against the Prometheus datasource, with anonymous viewing and
embedding enabled and Grafana served under `/grafana/`. Metrics come from two
places: the Atelier API exports NVML readings at `/metrics`
(`atelier_gpu_utilization_percent`, `atelier_gpu_memory_used_bytes`,
`atelier_gpu_temperature_celsius`, `atelier_gpu_power_watts`, `atelier_runs`,
`atelier_gpus_allocated`, `atelier_users_active`), and node-exporter covers the
host. Edit the dashboard in Grafana, export the JSON, and drop it back in that
folder to keep it.
