#!/usr/bin/env python3
"""Rebuild fotn-sketch1.canvas.tsx from metrics.jsonl + live tqdm."""
from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

RUN = Path("/root/autodl-fs/fotn/runs/curriculum_sketch1")
METRICS = RUN / "metrics.jsonl"
CANVAS = Path("/root/.cursor/projects/root-autodl-fs/canvases/fotn-sketch1.canvas.tsx")
TERM = Path("/root/.cursor/projects/root-autodl-fs/terminals/645704.txt")
CST = timezone(timedelta(hours=8))


def load_rows() -> list[dict]:
    if not METRICS.exists():
        return []
    out = []
    for line in METRICS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def parse_live() -> dict:
    info = {"desc": "", "pct": None, "step": None, "total": None, "it_s": None, "finished": False}
    if not TERM.exists():
        return info
    text = TERM.read_bytes()[-200000:].decode("utf-8", errors="replace").replace("\r", "\n")
    if "done. best va_rp=" in text:
        info["finished"] = True
    last_done = None
    done_pat = re.compile(r"^e(\d+) (flow|conv|res)/(\d+)\s")
    tq_pat = re.compile(
        r"(flow|conv|res) e(\d+):\s+(\d+)%\|[^\n]*?\|\s*(\d+)/(\d+)\s+\[[^\]]*?(?:,\s*([\d.]+)it/s)?"
    )
    for line in text.splitlines():
        m = done_pat.match(line.strip())
        if m:
            last_done = (m.group(2), int(m.group(3)))
    found = []
    for m in tq_pat.finditer(text):
        st, loc = m.group(1), int(m.group(2))
        if last_done is None or (st == last_done[0] and loc > last_done[1]) or (
            st != last_done[0] and st == "res"
        ):
            found.append(m)
    if found:
        m = found[-1]
        info.update(
            desc=f"{m.group(1)} e{m.group(2)}",
            pct=int(m.group(3)),
            step=int(m.group(4)),
            total=int(m.group(5)),
            it_s=float(m.group(6)) if m.group(6) else None,
        )
    return info


def train_alive() -> bool:
    r = subprocess.run(["pgrep", "-af", "train.py"], capture_output=True, text=True)
    return any("fotn-sketch1" in ln or ln.endswith("configs/config.yaml") or "train.py --config configs/config.yaml" in ln
               for ln in r.stdout.splitlines())


def js_num(x, nd=4):
    if x is None:
        return "null"
    return f"{float(x):.{nd}f}"


def arr(rows, key, nd=6):
    return "[" + ", ".join(js_num(r.get(key), nd) for r in rows) + "]"


def cell(x, nd=4):
    if x is None:
        return '"—"'
    return f'"{float(x):.{nd}f}"'


def load_eval(split: str) -> dict | None:
    p = RUN / f"eval_metrics_{split}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def eval_block() -> str:
    te, va = load_eval("test"), load_eval("val")
    if not te:
        return ""

    def trio(d, k):
        return f'{d[k]["1h"]:.3f}', f'{d[k]["2h"]:.3f}', f'{d[k]["3h"]:.3f}'

    def csi_row(raw, lab):
        t = te["csi"][raw]
        v = f'{va["csi"][raw]:.3f}' if va else "—"
        return f'            ["{lab}", "{t:.3f}", "{v}"]'

    rows = [
        csi_row("8.0", "CSI@8"),
        csi_row("16.0", "CSI@16"),
        csi_row("32.0", "CSI@32"),
        csi_row("51.2", "CSI@20dBz"),
        csi_row("76.8", "CSI@30dBz"),
        csi_row("102.4", "CSI@40dBz"),
    ]
    t20, t30, t40 = trio(te["csi_dbz_hour"], "20"), trio(te["csi_dbz_hour"], "30"), trio(te["csi_dbz_hour"], "40")
    if va:
        v20, v30, v40 = trio(va["csi_dbz_hour"], "20"), trio(va["csi_dbz_hour"], "30"), trio(va["csi_dbz_hour"], "40")
    else:
        v20 = v30 = v40 = ("—", "—", "—")
    hour_rows = [
        f'            ["20 dBz", "{t20[0]}", "{t20[1]}", "{t20[2]}", "{v20[0]}", "{v20[1]}", "{v20[2]}"]',
        f'            ["30 dBz", "{t30[0]}", "{t30[1]}", "{t30[2]}", "{v30[0]}", "{v30[1]}", "{v30[2]}"]',
        f'            ["40 dBz", "{t40[0]}", "{t40[1]}", "{t40[2]}", "{v40[0]}", "{v40[1]}", "{v40[2]}"]',
    ]
    vmae = f'{va["mae"]:.3f}' if va else "—"
    return f"""
      <Stack gap={{8}}>
        <H2>best.pt 雷达评估（e{te.get("ckpt_epoch")} {te.get("ckpt_stage")}）</H2>
        <Grid columns={{4}} gap={{12}}>
          <Stat value="{te['mae']:.3f}" label="202508 测试 MAE [0,128]" />
          <Stat value="{te['csi']['8.0']:.3f}" label="测试 CSI@8" />
          <Stat value="{te['csi']['51.2']:.3f}" label="测试 CSI@20dBz" />
          <Stat value="{vmae}" label="202505 验证 MAE [0,128]" />
        </Grid>
        <Table
          headers={{["阈值", "测试 CSI", "验证 CSI"]}}
          rows={{[
{(", " + chr(10)).join(rows)}
          ]}}
          columnAlign={{["left", "right", "right"]}}
          striped
        />
        <Table
          headers={{["阈值", "测 1h", "测 2h", "测 3h", "验 1h", "验 2h", "验 3h"]}}
          rows={{[
{(", " + chr(10)).join(hour_rows)}
          ]}}
          columnAlign={{["left", "right", "right", "right", "right", "right", "right"]}}
          striped
        />
        <Text tone="tertiary" size="small">
          来源 eval_metrics_test.json / eval_metrics_val.json · 20/30/40 dBz = 显示尺度 51.2/76.8/102.4 ·
          对照 l7tricks 测试 20 dBz 1h/2h/3h = 0.593/0.462/0.386
        </Text>
      </Stack>
"""


def stage_how(name, n, cap, current, alive, finished):
    if n == 0:
        return "未开始"
    if name == current and alive and not finished:
        return "进行中"
    if 0 < n < cap:
        return "早停"
    if n >= cap:
        return "满额"
    return "已停"


def build(rows: list[dict], live: dict, alive: bool) -> str:
    now = datetime.now(CST).strftime("%Y-%m-%d %H:%M CST")
    last = rows[-1]
    best = min(rows, key=lambda r: r["va_rp"])
    counts = {s: sum(1 for r in rows if r["stage"] == s) for s in ("flow", "conv", "res")}
    elapsed_min = sum(r.get("sec", 0.0) for r in rows) / 60.0
    last_sec = float(last.get("sec") or 160)

    if last["stage"] == "res":
        remain = max(0, 15 - last["local"]) * last_sec / 60.0
        remain_txt = f"res 最多还剩约 {remain:.0f} min（patience 5）。"
    elif last["stage"] == "conv":
        remain_txt = "conv 结束后还有 res ≤15 ep。"
    else:
        remain_txt = "后面还有 conv 与 res。"

    finished = bool(live["finished"])
    if finished:
        status_tone, status_title = "success", "课程已结束"
        live_line = "三段课程跑完，待测 202508。"
    elif not alive:
        status_tone, status_title = "warning", "训练进程已停"
        live_line = "进程不在，请核对 last.pt。"
    else:
        status_tone, status_title = "info", f"正在训练 {last['stage']}"
        if live.get("desc") and live.get("step"):
            rate = f"  {live['it_s']:.1f} it/s" if live.get("it_s") else ""
            live_line = f"实时 {live['desc']}  {live['step']}/{live['total']} ({live['pct']}%){rate}。"
        else:
            live_line = f"已完成全局 e{last['epoch']} {last['stage']}/{last['local']}，下一 epoch 或验证中。"

    cur = last["stage"]
    flow_pill = "success" if counts["flow"] else "neutral"
    conv_pill = "success" if counts["conv"] and cur == "res" else ("info" if cur == "conv" else "neutral")
    res_pill = "success" if finished else ("info" if cur == "res" and alive else "neutral")
    run_pill = "success" if alive else "warning"
    run_lab = "训练中" if alive else "进程停"
    flow_act = "true" if cur == "flow" else "false"
    conv_act = "true" if cur == "conv" else "false"
    res_act = "true" if cur == "res" else "false"

    evo_rows = [r for r in rows if r.get("tr_evo_r") is not None]
    evo_block = ""
    if evo_rows:
        evo_cats = "[" + ", ".join(f'"{r["epoch"]}"' for r in evo_rows) + "]"
        evo_block = f"""
      <Stack gap={{8}}>
        <H2>演化 / 门控训练项（conv 起）</H2>
        <LineChart
          categories={{{evo_cats}}}
          series={{[
            {{ name: "evo radar", data: {arr(evo_rows, "tr_evo_r")}, tone: "info" }},
            {{ name: "evo pwv", data: {arr(evo_rows, "tr_evo_v")} }},
            {{ name: "evo precip", data: {arr(evo_rows, "tr_evo_p")}, tone: "neutral" }},
            {{ name: "gate", data: {arr(evo_rows, "tr_gate")}, tone: "warning" }},
          ]}}
          beginAtZero={{false}}
          height={{220}}
        />
        <Text tone="tertiary" size="small">
          横轴全局 epoch · 纵轴训练均值 · 仅 conv/res；precip 在总损失里权 0.4，gate 权 0.15
        </Text>
      </Stack>"""

    table_lines = []
    tones = []
    for r in rows:
        table_lines.append(
            f'            ["{r["epoch"]}", "{r["stage"]}", "{r["local"]}", '
            f'"{r["va_rp"]:.4f}", "{r.get("a", 0):.3f}", "{r.get("tr_loss", 0):.4f}", '
            f'{cell(r.get("tr_evo_r"))}, {cell(r.get("tr_evo_v"))}, {cell(r.get("tr_gate"), 3)}, '
            f'"{r.get("sec", 0):.0f}s"]'
        )
        if r["epoch"] == best["epoch"]:
            tones.append('"success"')
        elif r["epoch"] == last["epoch"]:
            tones.append('"info"')
        else:
            tones.append("undefined")

    flow_best = min(r["va_rp"] for r in rows if r["stage"] == "flow")
    conv_best = min((r["va_rp"] for r in rows if r["stage"] == "conv"), default=None)
    res_best = min((r["va_rp"] for r in rows if r["stage"] == "res"), default=None)

    cats = "[" + ", ".join(f'"{r["epoch"]}"' for r in rows) + "]"
    data_lines = []
    for r in rows:
        data_lines.append(
            "  {ep:%d, st:%s, loc:%d, va:%s, a:%s, sec:%s, tr:%s, "
            "advR:%s, advV:%s, sm:%s, evoR:%s, evoV:%s, evoP:%s, gate:%s},"
            % (
                r["epoch"], json.dumps(r["stage"]), r["local"],
                js_num(r["va_rp"]), js_num(r.get("a")), js_num(r.get("sec"), 1),
                js_num(r.get("tr_loss")), js_num(r.get("tr_adv_r")), js_num(r.get("tr_adv_v")),
                js_num(r.get("tr_smooth")), js_num(r.get("tr_evo_r")), js_num(r.get("tr_evo_v")),
                js_num(r.get("tr_evo_p")), js_num(r.get("tr_gate")),
            )
        )

    return f"""import {{
  Callout,
  Grid,
  H1,
  H2,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
}} from "cursor/canvas";

// Auto-synced {now} from curriculum_sketch1
type RowT = {{
  ep: number; st: string; loc: number; va: number; a: number; sec: number;
  tr: number; advR: number; advV: number; sm: number;
  evoR: number | null; evoV: number | null; evoP: number | null; gate: number | null;
}};

const DATA: RowT[] = [
{chr(10).join(data_lines)}
];

export default function Sketch1Board() {{
  return (
    <Stack gap={{20}}>
      <Stack gap={{6}}>
        <H1>sketch1 三段课程</H1>
        <Text tone="secondary" size="small">
          来源 curriculum_sketch1 · 2023 训 / 202505 验 · 10→30 帧 · 同步 {now}
        </Text>
        <Row gap={{8}} wrap>
          <Pill tone="{flow_pill}" active={{{flow_act}}}>flow {counts['flow']}/20</Pill>
          <Pill tone="{conv_pill}" active={{{conv_act}}}>conv {counts['conv']}/20</Pill>
          <Pill tone="{res_pill}" active={{{res_act}}}>res {counts['res']}/15</Pill>
          <Pill tone="{run_pill}">{run_lab}</Pill>
        </Row>
      </Stack>

      <Grid columns={{6}} gap={{12}}>
        <Stat value="{cur}" label="当前阶段" />
        <Stat value="{last['epoch']} / {last['local']}" label="全局 / 阶段内 epoch" />
        <Stat value="{last['va_rp']:.4f}" label="最新 va_rp（radar+pwv evo）" />
        <Stat value="{best['va_rp']:.4f}" label="最优 va_rp · e{best['epoch']} {best['stage']}/{best['local']}" tone="success" />
        <Stat value="{last['a']:.3f}" label="转换效率 a" />
        <Stat value="{last.get('tr_loss', 0):.4f}" label="最新训练总损失" />
      </Grid>

      <Callout tone="{status_tone}" title="{status_title}">
        {live_line} 已跑 {elapsed_min:.0f} min。{remain_txt}
        选择指标是验证集 evo_loss(radar)+evo_loss(pwv)。flow 只训平流，va_rp 几乎不动是预期；
        conv 打开 K/η 后降到 0.0420 并早停；res 打开残差后首轮回升也常见。
      </Callout>
{eval_block()}
      <Stack gap={{8}}>
        <H2>验证选择指标 va_rp</H2>
        <LineChart
          categories={{{cats}}}
          series={{[{{ name: "va_rp = evo(radar)+evo(pwv)", data: {arr(rows, 'va_rp')}, tone: "info" }}]}}
          beginAtZero={{false}}
          height={{220}}
        />
        <Text tone="tertiary" size="small">
          横轴全局 epoch · 纵轴验证 evo 之和（越小越好）· flow e1–e{counts['flow']} · conv 随后 · res 最后
        </Text>
      </Stack>

      <Grid columns={{2}} gap={{20}}>
        <Stack gap={{8}}>
          <H2>训练总损失与平流项</H2>
          <LineChart
            categories={{{cats}}}
            series={{[
              {{ name: "train total", data: {arr(rows, 'tr_loss')}, tone: "info" }},
              {{ name: "advect radar", data: {arr(rows, 'tr_adv_r')} }},
              {{ name: "advect pwv", data: {arr(rows, 'tr_adv_v')} }},
              {{ name: "smooth", data: {arr(rows, 'tr_smooth')}, tone: "neutral" }},
            ]}}
            beginAtZero={{false}}
            height={{220}}
          />
          <Text tone="tertiary" size="small">
            横轴全局 epoch · 纵轴训练均值 · pwv 平流权 4×，radar 1×，smooth 0.05
          </Text>
        </Stack>
        <Stack gap={{8}}>
          <H2>转换效率 a</H2>
          <LineChart
            categories={{{cats}}}
            series={{[{{ name: "a", data: {arr(rows, 'a')}, tone: "warning" }}]}}
            beginAtZero={{false}}
            height={{220}}
          />
          <Text tone="tertiary" size="small">
            横轴全局 epoch · 纵轴无量纲 a · flow 冻结为 1 · conv 起可学
          </Text>
        </Stack>
      </Grid>
{evo_block}

      <Stack gap={{8}}>
        <H2>分阶段摘要</H2>
        <Table
          headers={{["阶段", "已跑 / 上限", "patience", "最好 va_rp", "结束方式"]}}
          rows={{[
            ["flow", "{counts['flow']} / 20", "6", "{flow_best:.4f}", "{stage_how('flow', counts['flow'], 20, cur, alive, finished)}"],
            ["conv", "{counts['conv']} / 20", "6", "{js_num(conv_best) if conv_best is not None else '—'}", "{stage_how('conv', counts['conv'], 20, cur, alive, finished)}"],
            ["res", "{counts['res']} / 15", "5", "{js_num(res_best) if res_best is not None else '—'}", "{stage_how('res', counts['res'], 15, cur, alive, finished)}"],
          ]}}
          columnAlign={{["left", "right", "right", "right", "left"]}}
          striped
        />
        <Text tone="tertiary" size="small">
          已用时 {elapsed_min:.1f} min · 每 epoch 约 {last_sec:.0f}s · 训练窗 17658 / 验证窗 7271
        </Text>
      </Stack>

      <Stack gap={{8}}>
        <H2>逐 epoch 记录</H2>
        <Table
          headers={{["e", "阶段", "局", "va_rp", "a", "train", "evo R", "evo V", "gate", "秒"]}}
          rows={{[
{("," + chr(10)).join(table_lines)}
          ]}}
          columnAlign={{["right", "left", "right", "right", "right", "right", "right", "right", "right", "right"]}}
          rowTone={{[{", ".join(tones)}]}}
          striped
          stickyHeader
        />
        <Text tone="tertiary" size="small">
          绿点是目前全局最优 va_rp，蓝点是最新一轮。evo / gate 在 flow 为空白。共 {{DATA.length}} 行。
        </Text>
      </Stack>
    </Stack>
  );
}}
"""


def main() -> None:
    rows = load_rows()
    if not rows:
        raise SystemExit("no metrics yet")
    live = parse_live()
    alive = train_alive()
    CANVAS.write_text(build(rows, live, alive), encoding="utf-8")
    last = rows[-1]
    print(
        f"synced {CANVAS.name} n={len(rows)} "
        f"stage={last.get('stage')} e={last.get('epoch')} "
        f"va_rp={last.get('va_rp')} alive={alive} live={live.get('desc')}"
    )


if __name__ == "__main__":
    main()
