"""Create scene_sku15..29.xml (SKU5F..SKU5T) in the sku20 repo's vendored env, as copies of scene_sku04.xml
exactly like make_scenes.py did for SKU5A..E, but with per-SKU density (varied mass). Never overwrites."""
import re, sys, json, time
from pathlib import Path
D=Path('/data02/kemove/rl100-sim-sku20/vendor/logistics-sorting-sim/src/logistics_sorting_sim')
SKUS=json.loads(Path('/data02/kemove/rl100-sim-sku20/docs/sku_table_new15.json').read_text())
stamp=time.strftime('%Y%m%d_%H%M%S'); src=(D/'scene_sku04.xml').read_text()
assert src.count('size="0.125 0.08 0.035" mass="0.22"')==1 and src.count('pos="0.008 -0.2 0.8360000000000001"')==1
for key,v in SKUS.items():
    idx=v['scene'];L,W,H=v['dims'];rho=v['rho'];out=D/f'scene_sku{idx:02d}.xml'
    if out.exists(): sys.exit(f'REFUSE overwrite {out}')
    m=rho*L*W*H
    x=src.replace('size="0.125 0.08 0.035" mass="0.22"',f'size="{L/2:.4f} {W/2:.4f} {H/2:.4f}" mass="{m:.4f}"')
    x=x.replace('pos="0.008 -0.2 0.8360000000000001"',f'pos="0.008 -0.2 {0.801+H/2:.4f}"')
    x=x.replace('pos="0.0 0.0 0.035300000000000005"',f'pos="0.0 0.0 {H/2+0.0003:.4f}"')
    x=re.sub(r'(<geom name="barcode\d" type="box" pos="[-0-9.]+ 0\.0 )0\.0357"',lambda mm:mm.group(1)+f'{H/2+0.0007:.4f}"',x)
    x=x.replace('<mujoco model="piper_dual_sorting">',f'<!-- {key}: carton {L}x{W}x{H} m, {m:.4f} kg (rho {rho}); generated from scene_sku04.xml {stamp} (sku20) -->\n<mujoco model="piper_dual_sorting">',1)
    assert x.count(f'{H/2+0.0007:.4f}"')==9
    out.write_text(x); print('created',out,round(m,4))
t=D/'task.py'; s=t.read_text(); anchor='SKU_INDEX.update({"SKU5A": 10, "SKU5B": 11, "SKU5C": 12, "SKU5D": 13, "SKU5E": 14})'
assert s.count(anchor)==1
t.write_text(s.replace(anchor,anchor+'\n# Additive (2026-10-20, sku20 repo): 15 new carton SKUs SKU5F..SKU5T; scenes scene_sku15..29.xml\nSKU_INDEX.update({'+', '.join(f'"{k}": {v["scene"]}' for k,v in SKUS.items())+'})',1))
print('task.py updated')
