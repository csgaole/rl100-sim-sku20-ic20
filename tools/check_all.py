import json,glob,h5py,numpy as np,sys
R='/data02/kemove/sku5-flip-demonstrations/data/v1'
out={}
for sid in 'ABCDE':
    O=f'{R}/sku{sid}'
    C=[json.load(open(f)) for f in sorted(glob.glob(O+'/candidates/*.json'))]
    att=np.array([len(c['attempts']) for c in C]);allr=[a['result'] for c in C for a in c['attempts']]
    F=np.array([c['selected']['result']['peak_gripper_force_N'] for c in C if c['selected']])
    bad=0;dark=0;n=0
    for f in sorted(glob.glob(O+'/episodes/*.h5')):
        h=h5py.File(f,'r');n+=1
        ok=h['actions/effective'].shape==(480,14) and all(h['observations/rgb/'+c].shape==(61,240,320,3) for c in ['global','left_wrist','right_wrist']) and json.loads(h.attrs['result_json'])['success'] and json.loads(h.attrs['result_json'])['peak_gripper_force_N']<=350
        bad+=not ok
        dark+=any(h['observations/rgb/'+c][k].mean()<5 for c in ['global','left_wrist','right_wrist'] for k in (0,30,60))
    man=sum(1 for _ in open(O+'/manifest.jsonl'))
    out[sid]=dict(selected=int(sum(c['selected'] is not None for c in C)),candidates=len(C),episodes=n,h5_bad=bad,dark_frames=dark,manifest=man,
      tries_mean=round(float(att.mean()),2),tries_max=int(att.max()),tries_hist=np.bincount(att).tolist(),per_try_success=round(float(np.mean([r['success'] for r in allr])),3),
      F=dict(min=round(F.min(),1),p50=round(float(np.median(F)),1),p90=round(float(np.percentile(F,90)),1),max=round(F.max(),1),over300=int((F>300).sum())))
    print(sid,json.dumps(out[sid]))
json.dump(out,open('/data02/kemove/sku5-carton-probe/results/collection_check.json','w'),indent=1)
