"""Create NEW scene_sku10..14.xml (copies of scene_sku04.xml) + additive SKU_INDEX. Never overwrites."""
import re, shutil, time, sys
from pathlib import Path
DIRS=[Path('/data02/kemove/rl100-sim-repro/env/logistics-sorting-sim/src/logistics_sorting_sim'),
      Path('/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/vendor/logistics-sorting-sim/src/logistics_sorting_sim')]
RHO=78.57
SKUS={'SKU5A':(10,[.25,.20,.15]),'SKU5B':(11,[.27,.22,.12]),'SKU5C':(12,[.30,.20,.10]),'SKU5D':(13,[.30,.25,.07]),'SKU5E':(14,[.25,.16,.14])}
stamp=time.strftime('%Y%m%d_%H%M%S')
for D in DIRS:
    src=(D/'scene_sku04.xml').read_text()
    assert src.count('size="0.125 0.08 0.035" mass="0.22"')==1
    assert src.count('pos="0.008 -0.2 0.8360000000000001"')==1
    for key,(idx,(L,W,H)) in SKUS.items():
        out=D/f'scene_sku{idx:02d}.xml'
        if out.exists(): sys.exit(f'REFUSE overwrite {out}')
        m=RHO*L*W*H
        x=src.replace('size="0.125 0.08 0.035" mass="0.22"',f'size="{L/2:.4f} {W/2:.4f} {H/2:.4f}" mass="{m:.4f}"')
        x=x.replace('pos="0.008 -0.2 0.8360000000000001"',f'pos="0.008 -0.2 {0.801+H/2:.4f}"')
        x=x.replace('pos="0.0 0.0 0.035300000000000005"',f'pos="0.0 0.0 {H/2+0.0003:.4f}"')
        x=re.sub(r'(<geom name="barcode\d" type="box" pos="[-0-9.]+ 0\.0 )0\.0357"',lambda mm:mm.group(1)+f'{H/2+0.0007:.4f}"',x)
        x=x.replace('<mujoco model="piper_dual_sorting">',f'<!-- {key}: carton {L}x{W}x{H} m, {m:.4f} kg (rho {RHO}); generated from scene_sku04.xml {stamp} -->\n<mujoco model="piper_dual_sorting">',1)
        assert x.count(f'{H/2+0.0007:.4f}"')==9
        out.write_text(x); print('created',out)
    t=D/'task.py'; s=t.read_text()
    old='SKU_INDEX = {"SKU02": 1, "SKU09": 4}'
    assert s.count(old)==1, 'unexpected SKU_INDEX line'
    bak=D/f'task.py.bak_{stamp}'; shutil.copy2(t,bak); print('backup',bak)
    new=old+'\n# Additive (2026-10-18): 5 new carton SKUs for multi-SKU BC; scenes scene_sku10..14.xml\nSKU_INDEX.update({'+', '.join(f'"{k}": {v[0]}' for k,v in SKUS.items())+'})'
    t.write_text(s.replace(old,new,1)); print('edited',t)
