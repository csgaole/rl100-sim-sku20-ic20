"""Phase-2 strategy search for SKU5A/B/D/E: base trajectory + learned per-arm Cartesian residual
keyframes (via TCP Jacobian maps) + phase timing, optimised by CEM in parallel CPU sims.
No object forces; same success contract as collect.py (2 s, label up, settled 200 ms, no drop, F<=350 N).
Writes only under /data02/kemove/rl100-sim-sku20/strategy/ (sku20 copy)."""
import os, sys, json, time, argparse
os.environ['CUDA_VISIBLE_DEVICES']=''; os.environ.setdefault('OMP_NUM_THREADS','1'); os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np, mujoco
S=Path('/data02/kemove/rl100-sim-sku20'); sys.path.insert(0,str(S/'expert'))
import probe_sku as P, collect
from logistics_sorting_sim import LogisticsSortingEnv
from logistics_sorting_sim.task import task_state
OUTROOT=Path('/data02/kemove/rl100-sim-sku20/strategy')
K=8; KNOTS=np.linspace(40,330,K)            # residual keyframes over phase ticks
G={}

def maps_for(ref, sku):
    maps=np.zeros((480,2,6,3))
    with LogisticsSortingEnv(sku=sku,render=False) as env:
        e=env.physics;m=e.model;d=mujoco.MjData(m)
        for k,q in enumerate(ref):
            for arm in range(2):
                d.qpos[e.qidx[arm]]=q[7*arm:7*arm+6];mujoco.mj_forward(m,d)
                jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(m,d,jp,jr,m.site(f'arm{arm}_tcp').id)
                J=np.vstack([jp[:,e.vidx[arm]],jr[:,e.vidx[arm]]]);maps[k,arm]=np.linalg.solve(J.T@J+1e-6*np.eye(6),J.T)[:,:3]
    return maps

def init_worker(sku, ref, maps):
    G.update(sku=sku, A=ref, maps=maps, setup=P.make_setup(sku))
    G['angles']=collect.ANGLES

NT=6  # timing params
def unpack(x):
    res=x[:K*6].reshape(K,2,3); t=x[K*6:]
    return res, dict(approach_rate=1+t[0], turn_rate=1+t[1], exit_rate=1+t[2], support_phase_ticks=20*t[3], angle_feedback=.2+t[4], grip_dz=t[5])

def residual_at(res, phase):
    out=np.zeros((2,3))
    for arm in range(2):
        for j in range(3): out[arm,j]=np.interp(phase,KNOTS,res[:,arm,j],left=0.,right=res[-1,arm,j]) * min(1,phase/KNOTS[0])
    return out

def sim(x, ic, record=False):
    A=G['A'];maps=G['maps'];res,tm=unpack(np.asarray(x));res=np.clip(res,-.12,.12)
    cfg=P.nominal_cfg(*ic[:3],seed=ic[3]);env=G['setup'](cfg);e=env.physics;m,d=e.model,e.data
    phase=0.;stable=[];peak=0.;dropped=False;traj=[]
    try:
        for tick in range(480):
            s=task_state(e);angle=float(np.arccos(-np.clip(s['label_up'],-1,1)))
            i=min(478,int(phase));f=min(1,phase-i);a=A[i]*(1-f)+A[i+1]*f
            sp=np.clip(phase+tm['support_phase_ticks']*np.sin(np.pi*min(phase,288)/288),0,479)
            si=min(478,int(sp));sf=min(1,sp-si);a[7:13]=A[si,7:13]*(1-sf)+A[si+1,7:13]*sf
            r=residual_at(res,phase)
            for arm in range(2):
                delta=np.array([.1*cfg['dx'],.1*cfg['dy'],0.])*min(1,tick/48)+r[arm]
                a[7*arm:7*arm+6]+=maps[i,arm]@delta
            e.step(a)
            if record: traj.append(e.history[-1][2].copy())
            s=task_state(e);v=np.zeros(6)
            mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('parcel').id,v,0)
            ok=s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15
            stable.append(bool(ok));peak=max(peak,max(s['forces']));dropped|=s['bottom']<.74
            rate=tm['approach_rate'] if phase<65 else tm['turn_rate'] if phase<235 else tm['exit_rate']
            if 65<phase<270: rate*=np.clip(1+tm['angle_feedback']*(angle-G['angles'][i]),.7,1.3)
            phase=min(479,phase+max(.3,rate))
        succ=all(stable[-48:]) and not dropped
        mode='success' if succ and peak<=350 else 'force_limit' if succ else 'drop' if dropped else 'wrong_face' if s['label_up']<.98 else 'unsettled'
        score=(2. if mode=='success' else 0.)+(0. if dropped else .5*(s['label_up']+1)/2)+.3*np.mean(stable[-48:])-(1. if dropped else 0.)-.002*max(0,peak-350)
        out=dict(mode=mode,score=float(score),peak=float(peak),label_up=float(s['label_up']),pos=np.round(d.xpos[m.body('parcel').id],3).tolist())
        if record: out['effective']=np.asarray(traj)
        return out
    finally: env.close()

def eval_x(args):
    x,ics=args; rs=[sim(x,ic) for ic in ics]
    return float(np.mean([r['score'] for r in rs])), [r['mode'] for r in rs], max(r['peak'] for r in rs)

if __name__=='__main__':
    from concurrent.futures import ProcessPoolExecutor
    p=argparse.ArgumentParser();p.add_argument('--sku',required=True);p.add_argument('--base',default='face_offset')
    p.add_argument('--alpha',type=float,default=1.);p.add_argument('--iters',type=int,default=40);p.add_argument('--pop',type=int,default=40)
    p.add_argument('--workers',type=int,default=10);p.add_argument('--sigma',type=float,default=.03);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--ntrain',type=int,default=7);p.add_argument('--tag',default='');a=p.parse_args()
    out=OUTROOT/f'{a.sku}_{a.base}_a{a.alpha}{a.tag}';out.mkdir(parents=True,exist_ok=True)
    if a.base=='raw': ref=P.ORIG_A.copy()
    else: ref,_=P.retarget(a.sku,OUTROOT,a.alpha,False,0.,mode=a.base)
    maps=maps_for(ref,a.sku); np.save(out/'base_ref.npy',ref); np.save(out/'base_maps.npy',maps)
    train_ics=P.ics(a.ntrain)     # nominal + ntrain seeds (880000..) for optimisation
    def held(n):
        out=[]
        for sd in range(n):
            r=np.random.default_rng(990000+sd); xy=r.uniform(-.005,.005,2); out.append((xy[0],xy[1],r.uniform(-2,2),990000+sd))
        return out
    val_ics=P.ics(20)             # nominal + 20 seeds (same 21 as the earlier probe) for validation
    D=K*6+NT; rng=np.random.default_rng(a.seed); mu=np.zeros(D); sd=np.r_[np.full(K*6,a.sigma),[.08,.1,.05,.3,.1,0.]]
    best=(-1e9,None,None); hist=[]; t0=time.time()
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init_worker,initargs=(a.sku,ref,maps)) as pool:
        for it in range(a.iters):
            X=mu+sd*rng.standard_normal((a.pop,D)); X[0]=mu
            R=list(pool.map(eval_x,[(x,train_ics) for x in X]))
            sc=np.array([r[0] for r in R]); el=np.argsort(-sc)[:max(4,a.pop//6)]
            if sc.max()>best[0]: best=(float(sc.max()),X[sc.argmax()].copy(),R[sc.argmax()])
            mu=.3*mu+.7*X[el].mean(0); sd=np.maximum(.3*sd+.7*X[el].std(0), np.r_[np.full(K*6,.004),[.01,.01,.01,.03,.01,0.]])
            hist.append(dict(it=it,best=best[0],it_max=float(sc.max()),it_mean=float(sc.mean()),best_modes=best[2][1]))
            print(json.dumps(hist[-1]),flush=True)
            if best[0]>=2.79 and it>=15: break   # all train ICs succeed (2+.5+.3 ~ 2.8 max)
        V=list(pool.map(sim,[best[1]]*len(val_ics),val_ics)); HO=list(pool.map(sim,[best[1]]*20,held(20)))
    modes={mm:sum(v['mode']==mm for v in V) for mm in set(v['mode'] for v in V)}
    summ=dict(sku=a.sku,base=a.base,alpha=a.alpha,train_best=best[0],val_success=modes.get('success',0),val_n=len(V),val_modes=modes,
              val_peak_max=max(v['peak'] for v in V),val_peaks=[round(v['peak'],1) for v in V],heldout_success=sum(v['mode']=='success' for v in HO),heldout_n=20,heldout_modes={mm:sum(v['mode']==mm for v in HO) for mm in set(v['mode'] for v in HO)},wall_s=round(time.time()-t0,1),iters=len(hist))
    np.save(out/'best_x.npy',best[1]); (out/'summary.json').write_text(json.dumps(dict(summary=summ,hist=hist,val=V),indent=1))
    print('SUMMARY',json.dumps(summ))
