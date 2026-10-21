"""External IC±20 gate runner for bc_s43_ic20 (built-in nominal_gate is imported at trainer startup and asserts p5_y2, so it cannot be switched live).
Watches development_v2/candidate_EEEE.pt for EPOCHS, evaluates each with p20_y10 IC (|dx|,|dy|<=20mm, |yaw|<=10deg = expert/collect_sku.py),
same seeds/eps/sampler/success as s43 gate (880000-880019, 20/SKU x 20 SKUs, upstream, flip_up_no_floor_v3). Report-only: never stops/alters training.
GPU: only a fully idle card among 7,4,5 (no compute apps, <1GiB); otherwise waits. Never shares a GPU with training or other users."""
import json,os,subprocess,sys,time,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from nominal_gate_multisku import SKUS,SEED0,PER_SKU
POS,YAW=.020,10.
PROTOCOL=dict(version='p20y10_multisku_ext_v1',skus=SKUS,seeds=[SEED0,SEED0+PER_SKU-1],episodes_per_sku=PER_SKU,episodes=PER_SKU*len(SKUS),position_half_range_m=POS,yaw_half_range_deg=YAW,sampler='upstream_eta1_final_sigma0',success='flip_up_no_floor_v3')
def smi(q):return subprocess.run(['nvidia-smi',f'--query-{q}','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
def idle_gpu():
    apps=smi('compute-apps=gpu_uuid');rows=dict((i,(u,float(m))) for i,u,m in [l.split(', ') for l in smi('gpu=index,uuid,memory.used').strip().splitlines()])
    for g in ('7','4','5'):
        if g in rows and rows[g][1]<1024 and rows[g][0] not in apps:return int(g)
    return None
def run(out,epoch,dry=False):
    cand=out/'development_v2'/f'candidate_{epoch:04d}.pt';folder=out/'development_p20y10';folder.mkdir(exist_ok=True)
    while not cand.exists():time.sleep(60)
    time.sleep(30)
    while (gpu:=idle_gpu()) is None:
        print(time.strftime('%F %T'),f'waiting for idle GPU (7/4/5) epoch={epoch}',flush=True);time.sleep(300)
    n=1 if dry else PER_SKU;skus=SKUS[:1] if dry else SKUS
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4')
    print(time.strftime('%F %T'),f'P20Y10_GATE_START epoch={epoch} gpu={gpu} cand={cand}',flush=True);sha=hashlib.sha256(cand.read_bytes()).hexdigest();ps=[];parts=[]
    for sku in skus:
        res=folder/f'eval_{epoch:04d}_{sku}.json';parts.append((sku,res))
        if res.exists():continue
        ps.append(subprocess.Popen([sys.executable,str(HERE/'evaluate_bc_multisku.py'),'--checkpoint',str(cand),'--output',str(res),'--seed-start',str(SEED0),'--episodes',str(n),'--sampler','upstream','--sku',sku,'--position-half-range',str(POS),'--yaw-half-range',str(YAW)],stdout=open(folder/f'eval_{epoch:04d}_{sku}.log','a'),stderr=subprocess.STDOUT,env=env))
    for p in ps:p.wait()
    R=[(s,json.loads(r.read_text())) for s,r in parts if r.exists()]
    assert all(r['position_half_range_m']==POS and r['yaw_half_range_deg']==YAW and r['checkpoint_sha256']==sha for s,r in R)
    per={s:dict(successes=r['successes'],episodes=r['episodes'],strict=r['strict_successes']) for s,r in R}
    rep=dict(epoch=epoch,gpu=gpu,candidate=str(cand),sha256=sha,protocol=PROTOCOL,successes=sum(v['successes'] for v in per.values()),episodes=sum(v['episodes'] for v in per.values()),per_sku=per,missing=[s for s,r in parts if not r.exists()],time=time.strftime('%F %T'))
    (folder/f'summary_{epoch:04d}.json').write_text(json.dumps(rep,indent=1));print('P20Y10_GATE_DONE',json.dumps({k:rep[k] for k in ['epoch','successes','episodes','missing']}),flush=True)
if __name__=='__main__':
    out=Path(sys.argv[1]);eps=[int(e) for e in sys.argv[2].split(',')];dry='--dry' in sys.argv
    for e in eps:run(out,e,dry)
