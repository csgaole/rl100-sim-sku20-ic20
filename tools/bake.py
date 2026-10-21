"""Bake a CEM strategy (base ref + residual + timing) into a stand-alone 480x14 reference for collect_sku.py,
then validate: replay on 21 standard + 20 held-out ICs (contacts) and a collect_sku-style search on 30 indices."""
import os,sys,json,subprocess
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np
sys.path.insert(0,'/data02/kemove/rl100-sim-sku20/tools')
import strategy_opt as SO
P=SO.P;collect=SO.collect
d=Path(sys.argv[1]);sku=json.loads((d/'summary.json').read_text())['summary']['sku']
ref=np.load(d/'base_ref.npy');maps=np.load(d/'base_maps.npy');x=np.load(d/'best_x.npy')
SO.init_worker(sku,ref,maps)
r=SO.sim(x,(0.,0.,0.,0),record=True);baked=r['effective'];assert baked.shape==(480,14),baked.shape
bm=SO.maps_for(baked,sku)
collect.setup_env=P.make_setup(sku);collect.A=baked
res,logs,_,_,_=collect.simulate(P.nominal_cfg(),record=True);ang=np.arccos(-np.clip(logs['label_up'],-1,1))
for n,v in [('baked_ref.npy',baked),('baked_maps.npy',bm),('baked_angles.npy',ang)]:np.save(d/n,v)
collect.ANGLES=ang
held=[(*np.random.default_rng(990000+s).uniform(-.005,.005,2),0,990000+s) for s in range(20)]
held=[]
for s in range(20):
    g=np.random.default_rng(990000+s);xy=g.uniform(-.005,.005,2);held.append((xy[0],xy[1],g.uniform(-2,2),990000+s))
rows=[P.run(sku,P.nominal_cfg(*ic[:3],seed=ic[3]),baked) for ic in P.ics(20)+held]
modes={};[modes.__setitem__(q['mode'],modes.get(q['mode'],0)+1) for q in rows]
res_knots,tm=SO.unpack(x)
out=dict(sku=sku,dir=str(d),nominal_baked=res['reason'],replay41_modes=modes,
  peak_max=max(q['peak_gripper_force_N'] for q in rows),armlink_ticks_median=float(np.median([q['arm_link_contact_ticks'] for q in rows])),
  max_pen_mm=max(q['max_penetration_mm'] for q in rows),timing={k:round(float(v),3) for k,v in tm.items()},
  residual_mm_arm0=(np.clip(res_knots[:,0],-.12,.12)*1000).round(0).tolist(),residual_mm_arm1=(np.clip(res_knots[:,1],-.12,.12)*1000).round(0).tolist(),
  knots_phase=SO.KNOTS.round(0).tolist())
(d/'bake_check.json').write_text(json.dumps(out,indent=1));print(json.dumps(out))
