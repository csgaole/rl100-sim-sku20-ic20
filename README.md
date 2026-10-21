# rl100-sim-sku20-ic20 (staging 2026-10-21 CST)

Fork of rl100-sim-sku20 with IC p20_y10 = +/-20mm / +/-10deg yaw (was p5_y2). Collect: 20 SKU x 100 demos, 80/10/10. GitHub create pending confirmation.

See docs/collect_config_ic20.json, docs/COLLECT_COMMANDS_IC20.md, tools/collect_all_ic20.sh. Do NOT start collection yet.

---

# rl100-sim-sku20 (created 2026-10-20 CST)

Fork of the 5-SKU multi-SKU setup (rl100-sim-repro + sku5-flip-demonstrations + sku5-carton-probe) extended to 20 SKUs. SKU5A..E data are HARDLINKS (read-only) of /data02/kemove/sku5-flip-demonstrations/data/v1; SKU5F..T are new (scene_sku15..29). No human correction data; scripted privileged-state expert only.

Layout: vendor/ (env, = env/), expert/ (collect_sku.py etc.), tools/ (strategy_opt CEM, bake, install_search_sku20.sh, render_sku20.sh, check_sku20.py), assets/ (per-SKU baked refs), strategy/ (CEM runs), data/v1/sku{A..T}, train/bc_sku20_v1 (BC config, NOT launched), docs/.

| SKU | L x W x H (m) | mass (kg) | flip method | demos | expert per-try strict | per-try flip_up_no_floor_v3 |
|---|---|---|---|---|---|---|
| SKU5A | 0.25 x 0.20 x 0.15 | 0.5893 | support_scale (CEM) | 100 | 0.885 | 0.947 |
| SKU5B | 0.27 x 0.22 x 0.12 | 0.5600 | face_offset (CEM r2s2) | 100 | 0.971 | 0.99 |
| SKU5C | 0.30 x 0.20 x 0.10 | 0.4714 | raw SKU09 reference (no CEM) | 100 | 0.535 | 0.759 |
| SKU5D | 0.30 x 0.25 x 0.07 | 0.4125 | face_offset (CEM) | 100 | 0.98 | 1.0 |
| SKU5E | 0.25 x 0.16 x 0.14 | 0.4400 | face_offset (CEM) | 100 | 0.862 | 0.914 |
| SKU5F | 0.22 x 0.18 x 0.12 | 0.3734 | face_offset (CEM r1; val 20/21, heldout 19/20) | 100 | 0.98 | 0.98 |
| SKU5G | 0.24 x 0.20 x 0.10 | 0.4320 | face_offset (CEM r1; val 20/21, heldout 17/20) | 100 | 0.758 | 0.97 |
| SKU5H | 0.26 x 0.18 x 0.09 | 0.2948 | support_scale (CEM r1; val 21/21, heldout 20/20) | 100 | 0.787 | 0.929 |
| SKU5I | 0.28 x 0.21 x 0.13 | 0.6115 | face_offset (CEM r1; val 21/21, heldout 19/20) | 100 | 0.847 | 1.0 |
| SKU5J | 0.28 x 0.24 x 0.10 | 0.5712 | face_offset (CEM r1; val 20/21, heldout 20/20) | 100 | 0.98 | 1.0 |
| SKU5K | 0.26 x 0.22 x 0.14 | 0.5606 | face_offset (CEM r2s2; val 21/21, heldout 19/20) | 100 | 0.855 | 0.991 |
| SKU5L | 0.30 x 0.22 x 0.08 | 0.4752 | face_offset (CEM r1; val 20/21, heldout 20/20) | 100 | 0.552 | 0.994 |
| SKU5M | 0.23 x 0.17 x 0.13 | 0.4829 | raw (CEM r1; val 21/21, heldout 20/20) | 100 | 0.84 | 0.857 |
| SKU5N | 0.29 x 0.23 x 0.11 | 0.5503 | raw (CEM r1; val 21/21, heldout 20/20) | 100 | 0.99 | 1.0 |
| SKU5O | 0.25 x 0.24 x 0.08 | 0.3840 | support_scale (CEM r1; val 21/21, heldout 20/20) | 100 | 0.855 | 0.991 |
| SKU5P | 0.27 x 0.19 x 0.15 | 0.5387 | raw (CEM r1; val 21/21, heldout 20/20) | 100 | 0.909 | 0.936 |
| SKU5Q | 0.30 x 0.18 x 0.12 | 0.4212 | face_offset (CEM r2s1; val 21/21, heldout 19/20) | 100 | 0.578 | 0.965 |
| SKU5R | 0.24 x 0.22 x 0.11 | 0.5808 | face_offset (CEM r1; val 20/21, heldout 18/20) | 100 | 0.847 | 1.0 |
| SKU5S | 0.28 x 0.17 x 0.10 | 0.4046 | face_offset (CEM r2s1; val 21/21, heldout 20/20) | 100 | 0.813 | 0.821 |
| SKU5T | 0.26 x 0.25 x 0.12 | 0.5850 | face_offset (CEM r1; val 21/21, heldout 19/20) | 100 | 0.877 | 1.0 |
