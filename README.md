# sketch1

按 `fotn-mag3h/l7tricks-repro-sketch.md` 从零搭的雷达+PWV 外推：平流 → 转化分流 → 残差。数据划分与原先相同（2023 夏训 / 202505 验 / 202508 测，10→30 帧）。

```bash
# 原版：转化头看降水，有 evo_precip
python -u train.py --config configs/config.yaml

# l7w8 风格：无降水输入 / GT / 损失
python -u train.py --config configs/config_rp.yaml

# ring16 ResGAN
python -u train_resgan.py --config configs/config_rp_flow32_resgan_gdisc_g025_ring16.yaml
```
