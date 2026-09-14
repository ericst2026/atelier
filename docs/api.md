# API

Base path `/api`. Bearer token in `Authorization`, or `?token=` for links the
browser opens directly (downloads, iframes, websockets). Interactive docs live at
`/api/docs`.

## Auth

| Method | Path | Who | What |
|---|---|---|---|
| POST | `/auth/login` | anyone | `{username,password}` → `{token,user}` |
| GET | `/auth/me` | signed in | the current user |
| POST | `/auth/password` | signed in | `{current,new}` |

## Users (teachers)

| Method | Path | What |
|---|---|---|
| GET | `/users` | list |
| POST | `/users` | `{username,name,role,password}` |
| POST | `/users/import` | CSV upload: `username,name,role,password` |
| PATCH | `/users/{id}` | rename, change role, reset password, deactivate |

## Experiments

| Method | Path | What |
|---|---|---|
| GET | `/experiments` | cards + this user's progress (+ registry errors for teachers) |
| GET | `/experiments/{slug}` | full spec: steps, params, materials with availability, README |
| GET | `/experiments/{slug}/materials-tree?path=` | browse a material folder |
| GET | `/experiments/{slug}/materials/{path}` | download one material file |
| GET | `/experiments/{slug}/sample/tree` | sample project file list |
| GET | `/experiments/{slug}/sample/file?path=` | one sample file |
| GET | `/experiments/{slug}/sample/download` | sample project as a zip |

## Runs

| Method | Path | What |
|---|---|---|
| POST | `/experiments/{slug}/steps/{n}/runs` | start a step: `{params,parent_run_id,gpus,label}` |
| GET | `/runs` | filters: `experiment, kind, step, status, user_id, mine, limit` |
| GET | `/runs/{id}` | one run |
| GET | `/runs/{id}/logs?offset=` | incremental log read |
| GET | `/runs/{id}/result` | `result.json` |
| GET | `/runs/{id}/live` | live series so far |
| GET | `/runs/{id}/artifacts` | artifact tree |
| GET | `/runs/{id}/artifacts/{path}` | download an artifact |
| POST | `/runs/{id}/cancel` | stop a queued or running job |
| DELETE | `/runs/{id}` | teachers: delete a finished run and its files |

Run kinds: `step` (guided), `workspace` (free command in the project), `test`
(grader on the live project), `grade` (grader on a frozen submission).

## Workspace — the student's project

| Method | Path | What |
|---|---|---|
| GET | `/workspaces/{slug}` | tree, size, default command (`?user_id=` for teachers) |
| GET/PUT/DELETE | `/workspaces/{slug}/file` | read, write, delete one file |
| POST | `/workspaces/{slug}/reset` | copy the sample project over it |
| POST | `/workspaces/{slug}/upload?replace=` | upload a zip |
| GET | `/workspaces/{slug}/download` | download as a zip |
| POST | `/workspaces/{slug}/run` | `{command,gpus,label}` |
| POST | `/workspaces/{slug}/test` | run the grader against the live project |
| GET | `/workspaces/{slug}/runs` | this project's recent runs |

## Submissions

| Method | Path | What |
|---|---|---|
| POST | `/experiments/{slug}/submissions` | freeze the workspace, auto-grade |
| GET | `/submissions` | own, or all with `?experiment=&user_id=` for teachers |
| GET | `/submissions/{id}` · `/tree` · `/file?path=` · `/download` · `/runs` | inspect |
| POST | `/submissions/{id}/test` | teachers: re-run the grader |
| POST | `/submissions/{id}/grade` | teachers: `{score,feedback,published}` |

## Displays and boards

| Method | Path | What |
|---|---|---|
| GET | `/displays` · `/displays/{n}` | current state (public by default) |
| PUT | `/displays/{n}` | teachers: `{mode,payload,name}`, modes `grafana, live, progress, leaderboard, run, message` |
| GET | `/board/live` | GPUs, running, queued, recent |
| GET | `/board/progress` | how far each student is in each experiment |
| GET | `/board/leaderboard?experiment=&metric=` | published submissions ranked |
| GET | `/board/run/{id}` | one run's metrics, charts and log tail |

## System and websockets

`GET /system/health` · `/system/gpus` · `/system/info` · `/system/queue` (teachers) ·
`GET /metrics` (Prometheus, unprefixed).

| Socket | Who | Sends |
|---|---|---|
| `/ws/runs/{id}?token=` | owner or teacher | `snapshot`, `log`, `progress`, `status` |
| `/ws/displays/{n}` | public by default | `display` — a full board snapshot, pushed by the server |
| `/ws/teacher?token=` | teachers | `live` every few seconds plus every run and submission event |
