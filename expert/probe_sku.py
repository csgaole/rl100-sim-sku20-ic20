"""Feasibility probe: replay the SKU09 reference trajectory on another SKU scene (no search).

Copied/adapted from collect.py (setup_env/simulate). Read-only w.r.t. existing assets/data:
writes only to --out (default /data02/kemove/sku5-carton-probe/results). CPU only, no render.
  python src/probe_sku.py --sku SKU5A --seeds 5 [--retarget support_scale --alpha 1.0]
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl'); os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys, json, time, argparse
from pathlib import Path
import numpy as np, mujoco
sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect
from collect import ROOT
from logistics_sorting_sim import LogisticsSortingEnv
ORIG_A = collect.A.copy()
_MON = {}

def nominal_cfg(dx=0., dy=0., yaw=0., seed=0):
    return dict(category='regular', index=-1, attempt=0, dx=float(dx), dy=float(dy), yaw_deg=float(yaw),
                approach_rate=1., turn_rate=1., exit_rate=1., support_phase_ticks=0., tracking_fraction=.1,
                height_m=0., angle_feedback=.2, disturbance_rate=.7, trigger_deficit_rad=.35,
                recovery_multiplier=1.6, seed=int(seed))

def make_setup(sku):
    def setup_env(cfg):
        env = LogisticsSortingEnv(sku=sku, render=False); env.reset(seed=0)
        e = env.physics; m, d = e.model, e.data
        adr = m.jnt_qposadr[m.joint('parcel_free').id]
        d.qpos[adr:adr+2] -= np.random.default_rng(0).uniform(-.005, .005, 2)
        d.qpos[adr:adr+2] += np.array([cfg['dx'], .05+cfg['dy']])
        yaw = np.pi/2+np.deg2rad(cfg['yaw_deg'])
        d.qpos[adr+3:adr+7] = [0, np.cos(yaw/2), np.sin(yaw/2), 0]
        m.actuator_gainprm[e.aidx.ravel(), 0] = 1200; m.actuator_biasprm[e.aidx.ravel(), 1] = -1200
        mujoco.mj_forward(m, d)
        pbox = m.geom('parcel_box').id
        mon = dict(arm_link_ticks=0, finger_ticks=0, min_dist=0., init_min_dist=0., tick=0, first_finger_tick=None)
        _MON['m'] = mon
        def scan():
            arm_link = finger = False
            for ci in range(d.ncon):
                c = d.contact[ci]
                if pbox not in (c.geom1, c.geom2): continue
                other = c.geom2 if c.geom1 == pbox else c.geom1
                n = m.geom(int(other)).name or ''
                if n.startswith('arm'):
                    mon['min_dist'] = min(mon['min_dist'], float(c.dist))
                    if mon['tick'] < 3: mon['init_min_dist'] = min(mon['init_min_dist'], float(c.dist))
                    if 'gripper_link' in n: finger = True
                    else: arm_link = True
            mon['arm_link_ticks'] += arm_link; mon['finger_ticks'] += finger
            if finger and mon['first_finger_tick'] is None: mon['first_finger_tick'] = mon['tick']
        scan()
        orig = e.step
        def step(a):
            r = orig(a); mon['tick'] += 1; scan(); return r
        e.step = step
        return env
    return setup_env

def run(sku, cfg, reference=None):
    collect.setup_env = make_setup(sku); collect.A = ORIG_A if reference is None else reference
    try:
        res, logs, _, _, _ = collect.simulate(cfg, record=True)
    finally:
        collect.A = ORIG_A
    mon = _MON['m']; lu = logs['label_up']; pose = logs['object_pose'][-1]
    if res['success']:
        mode = 'success' if res['peak_gripper_force_N'] <= 350 else 'force_limit'
    elif 'drop' in res['reason']: mode = 'drop'
    elif res['final_label_up'] < .98: mode = 'wrong_face'
    else: mode = 'unsettled'
    collision = bool(mon['arm_link_ticks'] > 0 or mon['init_min_dist'] < -.003 or mon['min_dist'] < -.01
                     or res['peak_gripper_force_N'] > 350)
    return dict(sku=sku, cfg={k: cfg[k] for k in ('dx', 'dy', 'yaw_deg', 'seed')}, mode=mode,
                success=bool(mode == 'success'), collision_flag=collision,
                peak_gripper_force_N=round(res['peak_gripper_force_N'], 2),
                final_label_up=round(res['final_label_up'], 4), max_label_up=round(float(np.max(lu)), 4),
                max_turn_deg=round(float(np.degrees(np.arccos(-np.clip(np.max(lu), -1, 1)))), 1),
                final_bottom_m=round(res['final_bottom_m'], 4), final_pos=np.round(pose[:3], 4).tolist(),
                final_quat=np.round(pose[3:], 4).tolist(), first_stable_s=res['first_stable_200ms_end_s'],
                arm_link_contact_ticks=mon['arm_link_ticks'], finger_contact_ticks=mon['finger_ticks'],
                first_finger_contact_tick=mon['first_finger_tick'],
                max_penetration_mm=round(-1000*mon['min_dist'], 2), initial_penetration_mm=round(-1000*mon['init_min_dist'], 2))

# ---------------- geometry retarget (support-edge scaling in the box frame) ----------------
def size_of(sku):
    env = LogisticsSortingEnv(sku=sku, render=False); m = env.physics.model
    s = 2*m.geom_size[m.geom('parcel_box').id].copy(); env.close(); return s   # body x=L, y=W, z=H

def reference_object_traj(out):
    f = out/'sku09_nominal_object_pose.npy'
    if f.exists(): return np.load(f)
    collect.setup_env = make_setup('SKU09'); res, logs, _, _, _ = collect.simulate(nominal_cfg(), record=True)
    np.save(f, logs['object_pose']); return logs['object_pose']

def quat_mat(q):
    R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); return R.reshape(3, 3)

def retarget(sku, out, alpha=1., scale_L=False, z_extra=0., rotation_weight=.15, mode='support_scale', soft=.02):
    s0 = size_of('SKU09'); s1 = size_of(sku); obj = reference_object_traj(out)
    S = np.diag([1+alpha*(s1[0]/s0[0]-1) if scale_L else 1., 1+alpha*(s1[1]/s0[1]-1), 1+alpha*(s1[2]/s0[2]-1)])
    env = make_setup('SKU09')(nominal_cfg()); e = env.physics; m, d = e.model, e.data
    corners = np.array([[0, sy*s0[1]/2, sz*s0[2]/2] for sy in (-1, 1) for sz in (-1, 1)])
    idx = np.unique(np.r_[np.arange(0, 480, 4), 479]); out_q = []; delta = np.zeros((2, 6)); maxerr = 0.
    try:
        for k in idx:
            nominal = ORIG_A[k].copy(); d.qpos[e.qidx] = np.stack([nominal[:6], nominal[7:13]]); mujoco.mj_forward(m, d)
            c0 = obj[k, :3]; R = quat_mat(obj[k, 3:])
            wc = c0 + corners@R.T; s = wc[np.argmin(wc[:, 2])]          # support edge point on table
            A = R@S@R.T
            targets = []
            for arm in range(2):
                p = d.site_xpos[m.site(f'arm{arm}_tcp').id].copy(); rot = d.site_xmat[m.site(f'arm{arm}_tcp').id].reshape(3, 3).copy()
                if mode == 'support_scale':
                    tgt = s+A@(p-s)
                else:   # face_offset: keep distance to nearest faces; centre moves as if scaled about support edge
                    c1 = s+A@(c0-s); qb = R.T@(p-c0); half = alpha*(s1-s0)/2
                    off = np.array([0., np.tanh(qb[1]/soft)*half[1], np.tanh(qb[2]/soft)*half[2]])
                    if scale_L: off[0] = np.tanh(qb[0]/soft)*half[0]
                    tgt = c1+R@(qb+off)
                targets.append((tgt+np.array([0, 0, z_extra]), rot))
            q = np.stack([nominal[:6], nominal[7:13]])+delta
            for arm, (pos, rot) in enumerate(targets):
                ids = e.qidx[arm]; vid = e.vidx[arm]; site = m.site(f'arm{arm}_tcp').id; nq = nominal[7*arm:7*arm+6]
                lo = m.jnt_range[np.array(e.joint_ids[arm]), 0]+.005; hi = m.jnt_range[np.array(e.joint_ids[arm]), 1]-.005
                q[arm] = np.clip(q[arm], lo, hi)
                def resid(qv):
                    d.qpos[ids] = qv; mujoco.mj_forward(m, d); cur = d.site_xmat[site].reshape(3, 3)
                    return np.r_[pos-d.site_xpos[site], rotation_weight*.5*sum(np.cross(cur[:, j], rot[:, j]) for j in range(3))]
                for _ in range(30):
                    er = resid(q[arm]); reg = 1e-4
                    if np.linalg.norm(er) < 1e-6: break
                    jp = np.zeros((3, m.nv)); jr = np.zeros_like(jp); mujoco.mj_jacSite(m, d, jp, jr, site)
                    J = np.vstack([jp[:, vid], rotation_weight*jr[:, vid]])
                    dq = np.clip(np.linalg.solve(J.T@J+reg*np.eye(6), J.T@er-reg*(q[arm]-nq)), -.1, .1)
                    cost = er@er+reg*np.sum((q[arm]-nq)**2); ok = False
                    for sc in [1, .5, .25, .125, .0625]:
                        pr = np.clip(q[arm]+sc*dq, lo, hi); re = resid(pr)
                        if re@re+reg*np.sum((pr-nq)**2) < cost: q[arm] = pr; ok = True; break
                    if not ok: break
                maxerr = max(maxerr, float(np.linalg.norm(resid(q[arm])[:3])))
            delta = q-np.stack([nominal[:6], nominal[7:13]]); out_q.append(np.r_[q[0], .012, q[1], .012])
    finally:
        env.close()
    out_q = np.asarray(out_q)
    full = np.stack([np.interp(np.arange(480), idx, out_q[:, j]) for j in range(14)], 1)
    return full, dict(mode=mode, alpha=alpha, scale_L=scale_L, z_extra=z_extra, S=np.diag(S).round(4).tolist(), max_ik_pos_err_m=round(maxerr, 5))

def ics(n):
    rows = [(0., 0., 0., 0)]
    for sd in range(n):
        r = np.random.default_rng(880000+sd); xy = r.uniform(-.020, .020, 2); rows.append((xy[0], xy[1], r.uniform(-10, 10), 880000+sd))
    return rows

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    p = argparse.ArgumentParser()
    p.add_argument('--sku', required=True); p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--retarget', choices=['none', 'support_scale', 'face_offset'], default='none')
    p.add_argument('--alpha', type=float, default=1.); p.add_argument('--scale_L', action='store_true')
    p.add_argument('--z_extra', type=float, default=0.); p.add_argument('--workers', type=int, default=6)
    p.add_argument('--tag', default=''); p.add_argument('--out', default='/data02/kemove/sku5-carton-probe/results')
    a = p.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    assert not str(out.resolve()).startswith(str((ROOT/'data').resolve())), 'refuse to write into existing data dir'
    t0 = time.time(); ref = None; meta = dict(retarget='none')
    if a.retarget != 'none':
        ref, meta = retarget(a.sku, out, a.alpha, a.scale_L, a.z_extra, mode=a.retarget); meta['retarget'] = a.retarget
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futs = [pool.submit(run, a.sku, nominal_cfg(dx, dy, yw, sd), ref) for dx, dy, yw, sd in ics(a.seeds)]
        rows = [f.result() for f in futs]
    summ = dict(sku=a.sku, meta=meta, n=len(rows), n_success=sum(r['success'] for r in rows),
                modes={mm: sum(r['mode'] == mm for r in rows) for mm in set(r['mode'] for r in rows)},
                peak_force_max_N=max(r['peak_gripper_force_N'] for r in rows), wall_s=round(time.time()-t0, 1), rows=rows)
    name = f"probe_{a.sku}_{meta['retarget']}{a.tag}.json"
    (out/name).write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != 'rows'}))
    for r in rows:
        print(' ', r['cfg']['seed'], r['mode'], 'coll' if r['collision_flag'] else '-', 'F', r['peak_gripper_force_N'],
              'lu', r['final_label_up'], 'maxturn', r['max_turn_deg'], 'armlink', r['arm_link_contact_ticks'],
              'pen', r['max_penetration_mm'], 'ipen', r['initial_penetration_mm'], 'pos', r['final_pos'])
