"""Per-SKU demo/expert stats for all 20 SKUs. Expert success per try = strict collect contract (success_2s & F<=350);
v3 per try = flip_up_no_floor_v3 analogue on the expert attempt (complete 2 s, never fell below .74 m, final label_up > .98)."""
import json,glob,sys,os
import numpy as np
R='/data02/kemove/rl100-sim-sku20/data/v1'; out={}
for sid in 'ABCDEFGHIJKLMNOPQRST':
    O=f'{R}/sku{sid}'
    if not os.path.isdir(O+'/candidates'): continue
    C=[json.load(open(f)) for f in sorted(glob.glob(O+'/candidates/*.json'))]
    allr=[a['result'] for c in C for a in c['attempts']]
    first=[c['attempts'][0]['result'] for c in C]
    strict=lambda r: r['success'] and r['peak_gripper_force_N']<=350
    v3=lambda r: r['reason']!='drop' and r['final_label_up']>.98
    F=np.array([c['selected']['result']['peak_gripper_force_N'] for c in C if c['selected']] or [0])
    eps=len(glob.glob(O+'/episodes/*.h5')); man=sum(1 for _ in open(O+'/manifest.jsonl')) if os.path.exists(O+'/manifest.jsonl') else 0
    out[sid]=dict(candidates=len(C),selected=sum(c['selected'] is not None for c in C),episodes=eps,manifest=man,tries=len(allr),
        per_try_strict=round(float(np.mean([strict(r) for r in allr])),3),first_try_strict=round(float(np.mean([strict(r) for r in first])),3),
        per_try_v3=round(float(np.mean([v3(r) for r in allr])),3),first_try_v3=round(float(np.mean([v3(r) for r in first])),3),
        selected_all_v3=all(v3(c['selected']['result']) for c in C if c['selected']),
        F_p50=round(float(np.median(F)),1),F_max=round(float(F.max()),1))
    print(sid,json.dumps(out[sid]))
json.dump(out,open('/data02/kemove/rl100-sim-sku20/docs/collection_check_sku20.json','w'),indent=1)
