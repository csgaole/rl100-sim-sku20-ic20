"""Per-SKU copy of collect.py: state-conditioned scripted demonstrations, strict 2 s validation.

Same physics/IC/success contract as collect.py (p20_y10 nominal +-20mm/+-10deg yaw, label up + settled
200 ms at 2 s, no drop, peak gripper force <= 350 N, no object forces, no human correction), but
parameterised by --sku/--ref/--maps/--angles and writing ONLY under --out (refuses the SKU09 data dir).
  search:  python src/collect_sku.py search --sku SKU5C --ref assets/ref_SKU5C.npy --out <dir> --start 0 --stop 100
  render:  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=7 CUDA_VISIBLE_DEVICES=7 python src/collect_sku.py render ...
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, time, hashlib, argparse
from pathlib import Path
import numpy as np
import mujoco
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor/logistics-sorting-sim/src'))
from logistics_sorting_sim import LogisticsSortingEnv
from logistics_sorting_sim.task import task_state
CAMS = ['global','left_wrist','right_wrist']
# Filled by configure() (also inside each worker via initializer)
SKU = None; A = None; MAPS = None; ANGLES = None; OUT = None; SEED_BASE = None; REF_NAME = None

def configure(sku, ref, maps, angles, out, seed_base):
    global SKU, A, MAPS, ANGLES, OUT, SEED_BASE, REF_NAME
    SKU = sku; A = np.load(ref); MAPS = np.load(maps); REF_NAME = str(ref)
    ANGLES = np.load(angles) if angles else np.interp(np.arange(480), np.arange(7,480,8),
        np.arccos(-np.clip([s['label_up'] for s in json.loads((ROOT/'assets/reference_states.json').read_text())],-1,1)))
    assert A.shape == (480,14) and MAPS.shape == (480,2,6,3) and ANGLES.shape == (480,)
    OUT = Path(out); SEED_BASE = int(seed_base)
    # IC±20 repo owns data/v1/skuA-T; only refuse the ±5mm line's hardlinked SKU5A-E dirs.
    _forbid = Path('/data02/kemove/rl100-sim-sku20/data/v1')
    assert all(not str(OUT.resolve()).startswith(str((_forbid/f'sku{k}').resolve())) for k in 'ABCDE'), 'refuse original ±5 SKU5A-E data dirs (hardlinked, read-only)'
    for sub in ['candidates','episodes','rejected']: (OUT/sub).mkdir(parents=True, exist_ok=True)

def setup_env(cfg):
    env=LogisticsSortingEnv(sku=SKU,render=False);env.reset(seed=0)
    e=env.physics;m,d=e.model,e.data
    adr=m.jnt_qposadr[m.joint('parcel_free').id]
    jitter=np.random.default_rng(0).uniform(-.005,.005,2)
    d.qpos[adr:adr+2]-=jitter
    d.qpos[adr:adr+2]+=np.array([cfg['dx'],.05+cfg['dy']])
    yaw=np.pi/2+np.deg2rad(cfg['yaw_deg'])
    d.qpos[adr+3:adr+7]=[0,np.cos(yaw/2),np.sin(yaw/2),0]
    m.actuator_gainprm[e.aidx.ravel(),0]=1200
    m.actuator_biasprm[e.aidx.ravel(),1]=-1200
    mujoco.mj_forward(m,d)
    return env

def simulate(cfg, capture=False, record=True):
    maps=MAPS;env=setup_env(cfg);e=env.physics;m,d=e.model,e.data
    renderer=mujoco.Renderer(m,height=240,width=320) if capture else None
    phase=0.;stable=[];peak=0.;dropped=False
    logs={k:[] for k in ['requested','effective','proprio','qpos','qvel','ctrl','object_pose','object_velocity','contact_force','bottom','label_up','phase','stable']}
    rgb={c:[] for c in CAMS};obs_proprio=[]
    config=dict(cfg, sku=SKU, reference=REF_NAME, initial_qpos=d.qpos.tolist(), initial_qvel=d.qvel.tolist(),
        initial_ctrl=d.ctrl.tolist(), parcel_mass=float(m.body_mass[m.body('parcel').id]),
        parcel_inertia=m.body_inertia[m.body('parcel').id].tolist(),
        parcel_half_extents=m.geom_size[m.geom('parcel_box').id].tolist(),
        geom_friction=m.geom_friction.tolist(), geom_names=[m.geom(i).name for i in range(m.ngeom)],
        gain=1200,physics_hz=240,action_hz=240,camera_hz=30,image_shape=[240,320,3],
        action_semantics='absolute_joint_position_targets_rad; columns 6,13 fixed_width_m=0.012',
        requested_to_effective='joint limit clipping then 3 rad/s target slew limit',
        task=f'{SKU}_long_edge_flip_2s_not_full_sorting',expert_uses_privileged_state=True)
    def snapshot():
        obs_proprio.append(e.proprio().copy())
        if renderer:
            for c in CAMS:
                renderer.update_scene(d,camera=c);rgb[c].append(renderer.render().copy())
    snapshot()
    try:
        for tick in range(480):
            state=task_state(e);angle=float(np.arccos(-np.clip(state['label_up'],-1,1)))
            i=min(478,int(phase));f=min(1,phase-i);a=A[i]*(1-f)+A[i+1]*f
            sp=np.clip(phase+cfg['support_phase_ticks']*np.sin(np.pi*min(phase,288)/288),0,479)
            si=min(478,int(sp));sf=min(1,sp-si);a[7:13]=A[si,7:13]*(1-sf)+A[si+1,7:13]*sf
            for arm in range(2):
                delta=np.array([cfg['tracking_fraction']*cfg['dx'],cfg['tracking_fraction']*cfg['dy'],cfg['height_m']])*min(1,tick/48)
                a[7*arm:7*arm+6]+=maps[i,arm]@delta
            if record:
                logs['proprio'].append(e.proprio().copy());logs['qpos'].append(d.qpos.copy());logs['qvel'].append(d.qvel.copy())
            e.step(a);s=task_state(e);v=np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('parcel').id,v,0)
            ok=s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15
            stable.append(bool(ok));peak=max(peak,max(s['forces']));dropped|=s['bottom']<.74
            if np.any(d.xfrc_applied) or np.any(d.qfrc_applied):raise AssertionError('External applied force')
            if record:
                vals=dict(requested=a.copy(),effective=e.history[-1][2],ctrl=d.ctrl.copy(),object_pose=np.r_[d.xpos[m.body('parcel').id],d.xquat[m.body('parcel').id]],object_velocity=v,contact_force=s['forces'],bottom=s['bottom'],label_up=s['label_up'],phase=phase,stable=ok)
                for k,val in vals.items():logs[k].append(val)
            if (tick+1)%8==0:snapshot()
            rate=cfg['approach_rate'] if phase<65 else cfg['turn_rate'] if phase<235 else cfg['exit_rate']
            if 65<phase<270:
                rate*=np.clip(1+cfg['angle_feedback']*(angle-ANGLES[i]),.7,1.3)
            phase=min(479,phase+rate)
        times=[(j+48)/240 for j in range(433) if all(stable[j:j+48])]
        success=all(stable[-48:]) and not dropped
        summary=dict(success=bool(success),reason='success_2s' if success else 'drop' if dropped else 'wrong_face_or_unsettled_at_2s',
            peak_gripper_force_N=float(peak),first_stable_200ms_end_s=times[0] if times else None,
            final_label_up=float(s['label_up']),final_bottom_m=float(s['bottom']),
            final_linear_speed=float(np.linalg.norm(v[3:])),final_angular_speed=float(np.linalg.norm(v[:3])),
            external_object_wrench=False)
        if record:
            logs={k:np.asarray(v) for k,v in logs.items()};logs['final_qpos']=d.qpos.copy();logs['final_qvel']=d.qvel.copy()
        return summary,logs,rgb,np.asarray(obs_proprio),config
    finally:
        if renderer:renderer.close()
        env.close()

def proposal(rng, index, attempt):
    # All 'regular' nominal p20_y10 starts. IC stream independent from control search.
    initial_rng=np.random.default_rng(SEED_BASE+index)
    xy=initial_rng.uniform(-.020,.020,2);yaw=initial_rng.uniform(-10,10)
    return dict(category='regular',index=index,attempt=attempt,dx=float(xy[0]),dy=float(xy[1]),yaw_deg=float(yaw),
        approach_rate=float(rng.uniform(.96,1.05)),turn_rate=float(rng.uniform(.94,1.08)),exit_rate=float(rng.uniform(.97,1.05)),
        support_phase_ticks=float(rng.uniform(-3,3)),tracking_fraction=float(rng.uniform(0,.2)),height_m=float(rng.uniform(-.0005,.0005)),
        angle_feedback=float(rng.uniform(.05,.35)),seed=SEED_BASE+index)

def find_episode(index,max_attempts=100):
    rng=np.random.default_rng(10_000_000+SEED_BASE+index);attempts=[];best=None
    for attempt in range(max_attempts):
        cfg=proposal(rng,index,attempt)
        result,_,_,_,_=simulate(cfg,record=False)
        row=dict(config=cfg,result=result);attempts.append(row)
        if not result['success'] or result['peak_gripper_force_N']>350:continue
        best=row;break
    (OUT/'candidates'/f'{index:04d}.json').write_text(json.dumps(dict(index=index,sku=SKU,category='regular',attempts=attempts,selected=best),indent=2))
    return dict(index=index,category='regular',attempts=len(attempts),selected=best is not None,result=best['result'] if best else None)

def save_episode(index):
    import h5py
    row=json.loads((OUT/'candidates'/f'{index:04d}.json').read_text())['selected']
    if not row:raise RuntimeError(f'No qualifying episode {index}')
    dest=OUT/'episodes'/f'episode_{index:04d}.h5'
    if dest.exists():raise RuntimeError(f'refuse overwrite {dest}')
    result,logs,rgb,proprio,cfg=simulate(row['config'],capture=True)
    assert result['success'] and abs(result['peak_gripper_force_N']-row['result']['peak_gripper_force_N'])<1e-7
    with h5py.File(str(dest)+'.tmp','w') as f:
        f.attrs['schema_version']='1.0';f.attrs['config_json']=json.dumps(cfg);f.attrs['result_json']=json.dumps(result)
        f.attrs['provenance']=f'privileged_state_conditioned_script_search; {SKU} geometry-retargeted SKU09 reference family; not human teleoperation'
        f.attrs['category']=cfg['category'];f.attrs['episode_id']=index;f.attrs['sku']=SKU
        f.create_dataset('observations/timestamp',data=np.arange(61)/30)
        f.create_dataset('observations/proprio',data=proprio)
        for camera in CAMS:
            f.create_dataset('observations/rgb/'+camera,data=np.asarray(rgb[camera]),compression='gzip',compression_opts=1,chunks=(1,240,320,3))
            f.create_dataset('observations/rgb_timestamp/'+camera,data=np.arange(61)/30)
        f.create_dataset('actions/timestamp',data=np.arange(480)/240)
        f.create_dataset('actions/requested',data=logs.pop('requested'))
        f.create_dataset('actions/effective',data=logs.pop('effective'))
        f.create_dataset('actions/actuator_ctrl',data=logs.pop('ctrl'))
        f.create_dataset('proprio/timestamp',data=np.arange(480)/240)
        f.create_dataset('proprio/state',data=logs.pop('proprio'))
        reward=np.zeros(480,np.float32);reward[-1]=1
        done=np.zeros(480,np.bool_);done[-1]=True
        f.create_dataset('outcome/reward',data=reward);f.create_dataset('outcome/terminated',data=done)
        f.create_dataset('diagnostics/post_step_timestamp',data=np.arange(1,481)/240)
        f.create_dataset('diagnostics/pre_step_timestamp',data=np.arange(480)/240)
        for k,v in logs.items():f.create_dataset('diagnostics/'+k,data=v,compression='gzip' if np.asarray(v).ndim else None)
    os.replace(str(dest)+'.tmp',dest)
    return dict(index=index,sku=SKU,path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),bytes=dest.stat().st_size,category=cfg['category'],result=result,config=row['config'])

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor,as_completed
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['search','render'])
    p.add_argument('--sku',required=True);p.add_argument('--ref',required=True);p.add_argument('--maps',required=True)
    p.add_argument('--angles',default='');p.add_argument('--out',required=True);p.add_argument('--seed_base',type=int,required=True)
    p.add_argument('--start',type=int,default=0);p.add_argument('--stop',type=int,default=100);p.add_argument('--workers',type=int,default=24)
    args=p.parse_args()
    cargs=(args.sku,args.ref,args.maps,args.angles,args.out,args.seed_base);configure(*cargs)
    fn=find_episode if args.mode=='search' else save_episode
    with ProcessPoolExecutor(max_workers=args.workers,initializer=configure,initargs=cargs) as pool:
        futures=[pool.submit(fn,i) for i in range(args.start,args.stop)]
        for fut in as_completed(futures):
            row=fut.result();print(json.dumps(row),flush=True)
            name='search_log.jsonl' if args.mode=='search' else 'manifest.jsonl'
            with open(OUT/name,'a') as f:f.write(json.dumps(row)+'\n')
