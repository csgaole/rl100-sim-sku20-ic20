import os,sys,json,glob,subprocess
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np,h5py,mujoco
from concurrent.futures import ProcessPoolExecutor
R='/data02/kemove/sku5-flip-demonstrations/data/v1';SK09='/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/data/episodes'
PK='/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/vendor/logistics-sorting-sim/src/logistics_sorting_sim'
SCENE={'A':10,'B':11,'C':12,'D':13,'E':14,'SKU09':4}
VID='/data02/kemove/sku5-carton-probe/videos';os.makedirs(VID,exist_ok=True)
DT=1/240
def stats(args):
    sid,f=args
    m=mujoco.MjModel.from_xml_path(f'{PK}/scene_sku{SCENE[sid]:02d}.xml');d=mujoco.MjData(m);pb=m.geom('parcel_box').id
    h=h5py.File(f,'r');q=h['diagnostics/qpos'][:];a=h['actions/effective'][:];res=json.loads(h.attrs['result_json'])
    pen=0.;armlink=0
    for k in range(0,len(q),2):
        d.qpos[:]=q[k];mujoco.mj_forward(m,d);al=False
        for ci in range(d.ncon):
            c=d.contact[ci]
            if pb not in (c.geom1,c.geom2):continue
            o=c.geom2 if c.geom1==pb else c.geom1;n=m.geom(int(o)).name or ''
            if n.startswith('arm'):
                if 'gripper_link' in n: pen=max(pen,-c.dist)
                else: al=True; pen=max(pen,-c.dist)
        armlink+=al
    j=a[:,[0,1,2,3,4,5,7,8,9,10,11,12]];jerk=np.abs(np.diff(j,3,axis=0))/DT**3;acc=np.abs(np.diff(j,2,axis=0))/DT**2
    return dict(sid=sid,file=f,pen_mm=1000*pen,armlink_ticks=2*armlink,jerk_max=float(jerk.max()),jerk_mean=float(jerk.mean()),acc_max=float(acc.max()),
                jerk_p99=float(np.percentile(jerk,99)),F=res['peak_gripper_force_N'])
jobs=[(s,f) for s in 'ABCDE' for f in sorted(glob.glob(f'{R}/sku{s}/episodes/*.h5'))]+[('SKU09',f) for f in sorted(glob.glob(SK09+'/*.h5'))]
with ProcessPoolExecutor(48) as ex: rows=list(ex.map(stats,jobs,chunksize=4))
summ={}
for s in ['SKU09','A','B','C','D','E']:
    r=[x for x in rows if x['sid']==s];g=lambda k:np.array([x[k] for x in r])
    summ[s]=dict(n=len(r),pen_mm_max=round(g('pen_mm').max(),2),pen_mm_p50=round(float(np.median(g('pen_mm'))),2),armlink_ticks_p50=float(np.median(g('armlink_ticks'))),
      jerk_max=round(g('jerk_max').max(),0),jerk_max_p50=round(float(np.median(g('jerk_max'))),0),jerk_mean_p50=round(float(np.median(g('jerk_mean'))),1),
      jerk_p99_p50=round(float(np.median(g('jerk_p99'))),0),acc_max=round(g('acc_max').max(),1),F_max=round(g('F').max(),1))
    print(s,json.dumps(summ[s]))
json.dump(dict(summary=summ,rows=rows),open('/data02/kemove/sku5-carton-probe/results/review_stats.json','w'),indent=0)
# videos
import imageio.v2 as iio
for s in 'ABCDE':
    C={int(os.path.basename(f)[:4]):json.load(open(f)) for f in glob.glob(f'{R}/sku{s}/candidates/*.json')}
    one=min(i for i,c in C.items() if len(c['attempts'])==1)
    hard=max(C,key=lambda i:(C[i]['selected']['result']['peak_gripper_force_N'],len(C[i]['attempts'])))
    for tag,i in [('1try',one),('hardest',hard)]:
        h=h5py.File(f'{R}/sku{s}/episodes/episode_{i:04d}.h5','r')
        fr=np.concatenate([h['observations/rgb/'+c][:] for c in ['global','left_wrist','right_wrist']],axis=2)
        out=f'{VID}/sku{s}_{tag}_ep{i:04d}_F{C[i]["selected"]["result"]["peak_gripper_force_N"]:.0f}N_tries{len(C[i]["attempts"])}.mp4'
        iio.mimwrite(out,fr,fps=30,codec='libx264',quality=8,macro_block_size=16);print(out,os.path.getsize(out))
