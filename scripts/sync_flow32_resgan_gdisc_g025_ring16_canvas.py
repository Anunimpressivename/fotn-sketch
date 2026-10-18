#!/usr/bin/env python3
"""Canvas for g025 ablation: ring-softmax pool k=5 stride=1 τ=16."""
from __future__ import annotations

import json
import math
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

RUN = Path("/root/autodl-fs/fotn-sketch1/runs/ring16")
METRICS = RUN / "metrics.jsonl"
LOG = RUN / "train.log"
CANVAS = Path(
    "/root/.cursor/projects/root-autodl-fs/canvases/fotn-sketch1-rp-flow32-resgan-gdisc-g025-ring16.canvas.tsx"
)
CST = timezone(timedelta(hours=8))
GLOBAL_W = 0.25
POOL_W = 0.01
MAE_W = 0.15
ADV_W = 6.0
LOG25 = math.log(25)
CFG_NAME = "configs/config_rp_flow32_resgan_gdisc_g025_ring16.yaml"
STEPS_PER_EP = 1472
GAN_EPOCHS = 12


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
    info = {
        "phase": "",
        "epoch": None,
        "pct": None,
        "step": None,
        "total": None,
        "it_s": None,
        "finished": False,
    }
    if not LOG.exists():
        return info
    text = LOG.read_bytes()[-500000:].decode("utf-8", errors="replace").replace("\r", "\n")
    if "done. best val CSI40=" in text:
        info["finished"] = True
        info["phase"] = "已结束"
        return info
    ep = list(
        re.finditer(
            r"gan e(\d+):\s+(\d+)%\|.*?(\d+)/(\d+)\s+\[[^\]]*?([\d.]+)it/s",
            text,
        )
    )
    pre = list(
        re.finditer(
            r"pretrain D:\s+(\d+)%\|.*?(\d+)/(\d+)\s+\[[^\]]*?([\d.]+)it/s",
            text,
        )
    )
    if ep:
        m = ep[-1]
        info.update(
            phase="gan",
            epoch=int(m.group(1)),
            pct=int(m.group(2)),
            step=int(m.group(3)),
            total=int(m.group(4)),
            it_s=float(m.group(5)),
        )
    pre_tail = text.rfind("pretrain D:")
    gan_tail = text.rfind("gan e")
    if pre and pre_tail > gan_tail and "pretrain stop" not in text[pre_tail:]:
        m = pre[-1]
        info.update(
            phase="预训练",
            epoch=None,
            pct=int(m.group(1)),
            step=int(m.group(2)),
            total=int(m.group(3)),
            it_s=float(m.group(4)),
        )
    return info


def train_alive() -> bool:
    r = subprocess.run(["pgrep", "-af", "train_resgan.py"], capture_output=True, text=True)
    for ln in r.stdout.splitlines():
        if (
            "python" in ln
            and CFG_NAME in ln
            and "pgrep" not in ln
            and "AGENT_LOOP" not in ln
        ):
            return True
    return False


def num(x, nd: int) -> str:
    return f"{float(x):.{nd}f}"


def downsample(rows: list[dict], cap: int = 80) -> list[dict]:
    if len(rows) <= cap:
        return rows
    stride = max(len(rows) // cap, 1)
    shown = rows[::stride]
    if shown[-1] is not rows[-1]:
        shown.append(rows[-1])
    return shown


def main() -> None:
    rows = load_rows()
    gan_ep = [r for r in rows if r.get("stage") == "gan" and "val_csi40" in r]
    steps = [r for r in rows if r.get("split") == "train_step"]
    live = parse_live()
    alive = train_alive()
    now = datetime.now(CST).strftime("%H:%M")
    if live["finished"]:
        status = "已结束"
        callout_tone = "success"
    elif alive:
        status = "训练中"
        callout_tone = "info"
    else:
        status = "进程不在"
        callout_tone = "warning"
    step_label = f"{live['step']}/{live['total']}" if live["step"] is not None and live["total"] else "—"
    pct = f"{live['pct']}%" if live["pct"] is not None else "—"
    speed = f"{live['it_s']:.2f}" if live["it_s"] is not None else "—"
    local_ep = str(live["epoch"]) if live["epoch"] is not None else "—"
    last_step = steps[-1] if steps else None
    sup_now = f"{last_step['sup']:.3f}" if last_step and "sup" in last_step else "—"
    gadv_w = (
        f"{GLOBAL_W * last_step['gadv']:.2f}"
        if last_step and "gadv" in last_step
        else "—"
    )
    pool_w = (
        f"{POOL_W * last_step['pool_mse']:.2f}"
        if last_step and "pool_mse" in last_step
        else "—"
    )
    ent_now = f"{last_step['pool_ent']:.3f}" if last_step and "pool_ent" in last_step else "—"
    remain = "—"
    if live["it_s"] and live["step"] and live["total"] and live["epoch"]:
        left_this = max(live["total"] - live["step"], 0) / live["it_s"]
        left_ep = (GAN_EPOCHS - live["epoch"]) * (live["total"] / live["it_s"])
        mins = (left_this + left_ep) / 60.0
        remain = f"{mins:.0f} min"
    done_steps = 0
    if live["epoch"] and live["step"]:
        done_steps = (live["epoch"] - 1) * STEPS_PER_EP + live["step"]
    best = None
    if gan_ep:
        best = max(gan_ep, key=lambda r: float(r.get("val_csi40", -1.0)))
    if live["finished"] and best is not None:
        callout_body = (
            f"12 个 epoch 完成。验证最优 e{int(best['local'])} CSI40 {float(best['val_csi40']):.4f}。"
        )
        test_early = RUN / "eval_metrics_test_epoch04.json"
        if test_early.exists():
            te = json.loads(test_early.read_text(encoding="utf-8"))
            callout_body += (
                f" 测试 202508：MAE {float(te['mae']):.3f}，"
                f"CSI@8 {float(te['csi']['8.0']):.3f}，CSI40 {float(te['csi']['102.4']):.3f}。"
            )
    else:
        callout_body = (
            f"阶段 {live['phase'] or '—'} · gan epoch {local_ep}/12 · {step_label}（{pct}）· "
            f"{speed} step/秒 · 权重熵 {ent_now}（均匀上界 {LOG25:.2f}）· 约剩 {remain}"
        )

    used = ["Callout", "H1"]
    if len(steps) >= 2 or gan_ep:
        used += ["H2", "LineChart"]
    used += ["Pill", "Row", "Stack", "Stat", "Text"]
    if gan_ep:
        used.append("Table")
    if done_steps > 0 and (alive or live["finished"]):
        used.append("UsageBar")
    import_line = "import { " + ", ".join(used) + ' } from "cursor/canvas";'

    parts = [
        import_line,
        "",
        "export default function Flow32ResganGdiscG025Ring16() {",
        "  return (",
        "    <Stack gap={20}>",
        "      <Stack gap={6}>",
        "        <H1>残差 GAN · g025 ring-softmax τ=16</H1>",
        "        <Text tone=\"secondary\">",
        "          相对 g025：损失权重不变，把 MaxPool 换成 5×5 ring-softmax（切比雪夫环 1 / 0.5 / 0.25 只乘 logit，加权用原值），stride=1，τ=16。局部对抗 6，全局 0.25，池化 0.01，像素 MAE 0.15。从 best_stage_res 起训 12 个 epoch。",
        "        </Text>",
        "      </Stack>",
        "      <Row gap={8} align=\"center\">",
        f"        <Pill active>{status}</Pill>",
        f"        <Text tone=\"tertiary\" size=\"small\">同步 {now}</Text>",
        "      </Row>",
        f"      <Callout tone=\"{callout_tone}\" title=\"{status}\">",
        f"        {callout_body}",
        "      </Callout>",
        "      <Row gap={20} align=\"end\" wrap>",
        f"        <Stat value=\"{live['phase'] or '—'}\" label=\"阶段\" />",
        f"        <Stat value=\"{local_ep}\" label=\"gan epoch\" />",
        f"        <Stat value=\"{step_label}\" label=\"step\" />",
        f"        <Stat value=\"{speed}\" label=\"step/秒\" />",
        f"        <Stat value=\"{ent_now}\" label=\"最近 pool_ent\" />",
        f"        <Stat value=\"{pool_w}\" label=\"最近 0.01×池化\" />",
        f"        <Stat value=\"{gadv_w}\" label=\"最近 0.25×全局\" />",
        f"        <Stat value=\"{sup_now}\" label=\"最近监督损失\" />",
        "      </Row>",
    ]

    if done_steps > 0 and (alive or live["finished"]):
        parts += [
            "      <UsageBar",
            f"        total={{{GAN_EPOCHS * STEPS_PER_EP}}}",
            f"        topLeftLabel=\"GAN 进度 {done_steps}/{GAN_EPOCHS * STEPS_PER_EP} step\"",
            f"        topRightLabel=\"约剩 {remain}\"",
            "        segments={[",
            f"          {{ id: \"done\", value: {done_steps} }},",
            "        ]}",
            "      />",
        ]

    if best is not None:
        parts += [
            "      <Row gap={20} align=\"end\" wrap>",
            f"        <Stat value=\"{int(best['local'])}\" label=\"验证最优 epoch\" />",
            f"        <Stat value=\"{num(best['val_csi40'], 4)}\" label=\"最优 CSI40\" />",
            f"        <Stat value=\"{num(best['val_radar_mae128'], 3)}\" label=\"该 epoch MAE\" />",
            f"        <Stat value=\"{num(best.get('val_csi8', float('nan')), 3)}\" label=\"该 epoch CSI@8\" />",
            "      </Row>",
        ]

    if len(steps) >= 2:
        shown = downsample(steps)
        cats = ", ".join(f"\"e{r.get('local', r.get('epoch'))}·{r['step']}\"" for r in shown)
        adv = ", ".join(num(ADV_W * float(r["adv"]), 2) for r in shown)
        gadv = ", ".join(num(GLOBAL_W * float(r.get("gadv", 0.0)), 2) for r in shown)
        pool = ", ".join(num(POOL_W * float(r["pool_mse"]), 2) for r in shown)
        pix = ", ".join(num(MAE_W * float(r.get("pixel_mae", 0.0)), 2) for r in shown)
        ents = ", ".join(num(r.get("pool_ent", float("nan")), 3) for r in shown)
        parts += [
            "      <H2>写进生成器的四项加权损失</H2>",
            "      <LineChart",
            f"        categories={{[{cats}]}}",
            "        series={[",
            "          { name: \"6×局部对抗\", data: [" + adv + "] },",
            "          { name: \"0.25×全局对抗\", data: [" + gadv + "] },",
            "          { name: \"0.01×池化 MSE\", data: [" + pool + "] },",
            "          { name: \"0.15×逐像素 MAE\", data: [" + pix + "] },",
            "        ]}",
            "        height={240}",
            "        beginAtZero",
            "        fill={false}",
            "      />",
            "      <Text size=\"small\" tone=\"tertiary\">",
            f"        横轴是训练步（e·step）。纵轴是写入生成器总损失的加权值。来源 metrics.jsonl · 同步 {now}",
            "      </Text>",
            "      <H2>ring-softmax 权重熵</H2>",
            "      <LineChart",
            f"        categories={{[{cats}]}}",
            "        series={[{ name: \"pool_ent\", data: [" + ents + "], tone: \"info\" }]}",
            "        height={200}",
            "        beginAtZero={false}",
            "        fill={false}",
            f"        referenceLines={{[{{ value: {LOG25:.4f}, label: \"均匀上界 log(25)\" }}]}}",
            "      />",
            "      <Text size=\"small\" tone=\"tertiary\">",
            f"        5×5 窗口权重熵。{LOG25:.2f} 是均匀分布上界；越低越接近硬 max。来源 metrics.jsonl · 同步 {now}",
            "      </Text>",
        ]

    if gan_ep:
        cats = ", ".join(f"\"{r['local']}\"" for r in gan_ep)
        parts += [
            "      <H2>验证集 40 dBZ CSI 与雷达 MAE</H2>",
            "      <LineChart",
            f"        categories={{[{cats}]}}",
            "        series={[",
            "          { name: \"40 dBZ CSI\", data: [" + ", ".join(num(r['val_csi40'], 4) for r in gan_ep) + "] },",
            "          { name: \"雷达 MAE / 100\", data: [" + ", ".join(num(r['val_radar_mae128'] / 100.0, 4) for r in gan_ep) + "] },",
            "        ]}",
            "        height={240}",
            "        fill={false}",
            "      />",
            "      <Text size=\"small\" tone=\"tertiary\">",
            f"        横轴是 gan epoch。40 dBZ CSI 越高越好。雷达 MAE 除以 100 后画在同一张图上。来源 metrics.jsonl · 同步 {now}",
            "      </Text>",
            "      <Table",
            "        headers={[\"epoch\", \"雷达 MAE\", \"CSI@8\", \"20 dBZ\", \"30 dBZ\", \"40 dBZ\", \"池化MSE\", \"权重熵\", \"全局对抗\", \"监督损失\"]}",
            "        columnAlign={[\"right\", \"right\", \"right\", \"right\", \"right\", \"right\", \"right\", \"right\", \"right\", \"right\"]}",
            "        rows={["
            + ", ".join(
                "[" + ", ".join([
                    str(int(r["local"])),
                    num(r["val_radar_mae128"], 3),
                    num(r["val_csi8"], 3),
                    num(r["val_csi20"], 3),
                    num(r["val_csi30"], 3),
                    num(r["val_csi40"], 3),
                    num(r.get("pool_mse", float("nan")), 1),
                    num(r.get("pool_ent", float("nan")), 3),
                    num(r.get("gadv", float("nan")), 3),
                    num(r.get("sup", float("nan")), 3),
                ]) + "]"
                for r in gan_ep
            )
            + "]}",
            "        striped",
            "      />",
            "      <Text size=\"small\" tone=\"tertiary\">",
            "        对照（测试 202508）：g025 MaxPool s=2 最好 ep3，MAE 7.370，CSI@8 0.494，CSI40 0.109；pools1 MaxPool s=1 最好 ep6，MAE 7.358，CSI@8 0.482，CSI40 0.110。",
            "      </Text>",
        ]

    test_path = RUN / "eval_metrics_test_epoch04.json"
    if test_path.exists():
        test = json.loads(test_path.read_text(encoding="utf-8"))
        h40 = test["csi_dbz_hour"]["40"]
        parts += [
            "      <H2>测试 202508 · 验证最优 e4</H2>",
            "      <Row gap={20} align=\"end\" wrap>",
            f"        <Stat value=\"{num(test['mae'], 3)}\" label=\"测试 MAE\" />",
            f"        <Stat value=\"{num(test['csi']['8.0'], 3)}\" label=\"测试 CSI@8\" />",
            f"        <Stat value=\"{num(test['csi']['51.2'], 3)}\" label=\"测试 20 dBZ\" />",
            f"        <Stat value=\"{num(test['csi']['76.8'], 3)}\" label=\"测试 30 dBZ\" />",
            f"        <Stat value=\"{num(test['csi']['102.4'], 3)}\" label=\"测试 40 dBZ\" />",
            f"        <Stat value=\"{num(h40['1h'], 3)}/{num(h40['2h'], 3)}/{num(h40['3h'], 3)}\" label=\"40 dBZ 1h/2h/3h\" />",
            "      </Row>",
            "      <Table",
            "        headers={[\"模型\", \"epoch\", \"MAE\", \"CSI@8\", \"20 dBZ\", \"30 dBZ\", \"40 dBZ\", \"40dBZ 1h\", \"40dBZ 2h\", \"40dBZ 3h\"]}",
            "        columnAlign={[\"left\", \"right\", \"right\", \"right\", \"right\", \"right\", \"right\", \"right\", \"right\", \"right\"]}",
            "        rows={[",
            f"          [\"ring16 τ=16\", \"4\", \"{num(test['mae'], 3)}\", \"{num(test['csi']['8.0'], 3)}\", \"{num(test['csi']['51.2'], 3)}\", \"{num(test['csi']['76.8'], 3)}\", \"{num(test['csi']['102.4'], 3)}\", \"{num(h40['1h'], 3)}\", \"{num(h40['2h'], 3)}\", \"{num(h40['3h'], 3)}\"],",
            "          [\"g025 MaxPool s=2\", \"3\", \"7.370\", \"0.494\", \"0.346\", \"0.230\", \"0.109\", \"0.231\", \"0.140\", \"0.099\"],",
            "          [\"pools1 MaxPool s=1\", \"6\", \"7.358\", \"0.482\", \"0.344\", \"0.226\", \"0.110\", \"0.232\", \"0.136\", \"0.096\"],",
            "        ]}",
            "        striped",
            "      />",
            "      <Text size=\"small\" tone=\"tertiary\">",
            f"        测试月份 202508，7317 样本。ring16 用 best_csi40.pt（验证 e4）。来源 eval_metrics_test_epoch04.json · 同步 {now}",
            "      </Text>",
        ]

    parts += [
        "      <Text size=\"small\" tone=\"tertiary\">",
        f"        日志 {RUN}/train.log · 初始化 best_stage_res.pt · 不覆盖 curriculum_dem_flow / curriculum_dem_flow_gamma",
        "      </Text>",
        "    </Stack>",
        "  );",
        "}",
        "",
    ]
    CANVAS.write_text("\n".join(parts), encoding="utf-8")
    last = gan_ep[-1] if gan_ep else None
    print(json.dumps({
        "status": status,
        "alive": alive,
        "live": live,
        "remain": remain,
        "n_gan": len(gan_ep),
        "n_step": len(steps),
        "best_local": None if best is None else int(best["local"]),
        "best_csi40": None if best is None else best.get("val_csi40"),
        "last_mae": None if last is None else last.get("val_radar_mae128"),
        "last_csi40": None if last is None else last.get("val_csi40"),
        "last_pool_w": None if last_step is None else POOL_W * float(last_step.get("pool_mse", 0.0)),
        "last_gadv_w": None if last_step is None else GLOBAL_W * float(last_step.get("gadv", 0.0)),
        "last_adv_w": None if last_step is None else ADV_W * float(last_step.get("adv", 0.0)),
        "last_mae_w": None if last_step is None else MAE_W * float(last_step.get("pixel_mae", 0.0)),
        "last_pool_ent": None if last_step is None else last_step.get("pool_ent"),
        "last_sup": None if last_step is None else last_step.get("sup"),
        "canvas": str(CANVAS),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
