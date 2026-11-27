import json,glob,os,statistics as st,math,random
roots=glob.glob('/tmp/np_probe/leaderboard/*')+glob.glob('/tmp/np_pr20/leaderboard/Ridge*')
tasks=['onset','speech','volume','delta_volume','pitch','word_index','word_gap','gpt2_surprisal','word_head_pos','word_part_speech','word_length','global_flow','local_flow','frame_brightness','face_num']
want={'MAPA':'MAPA','DIVER-1(0.1s,tiny,frozen)':'DIVERfz','PopulationTransformer_Lap':'PopT','CNN_Lap':'CNN','Ridge_6':'Ridge6','Linear_raw':'LinRaw','Linear_spectrogram':'LinSpec','Linear_Laplacian':'LinLap'}
def load(reg):
    res={}
    for r in roots:
        b=os.path.basename(r); k=next((v for p,v in want.items() if b.startswith(p)),None)
        if not k or not os.path.isdir(f'{r}/{reg}'): continue
        d={}
        for t in tasks:
            f=f'{r}/{reg}/population_{t}.json'
            if os.path.exists(f):
                j=json.load(open(f))['evaluation_results']
                d[t]={s:[x['test_roc_auc'] for x in v['population']['one_second_after_onset']['folds']] for s,v in j.items()}
        res[k]=d
    return res
for reg in ['Within-Session','Cross-Session','Cross-Subject']:
    R=load(reg); print('\n##',reg, list(R))
    sess=sorted(R['MAPA']['onset'])
    def smac(k): return {s:st.mean(st.mean(R[k][t][s]) for t in tasks) for s in sess}
    m=smac('MAPA')
    for k in R:
        if k=='MAPA' or len(R[k])<15 or set(R[k]['onset'])!=set(sess): continue
        o=smac(k); d=[m[s]-o[s] for s in sess]
        md=st.mean(d); sd=st.stdev(d); se=sd/math.sqrt(len(d))
        # sign-flip permutation
        cnt=0;N=20000
        for _ in range(N):
            if abs(st.mean([x*random.choice([1,-1]) for x in d]))>=abs(md)-1e-12: cnt+=1
        print(f'MAPA-{k}: mean {md:+.4f} sd_sess {sd:.4f} SE_paired {se:.4f} t {md/se:.2f} perm_p {cnt/N:.4f} wins {sum(x>0 for x in d)}/{len(d)}')
    # SE of single-model macro across sessions
    for k in ['MAPA','Ridge6']:
        v=list(smac(k).values()); print(k,'unpaired SE across sessions',round(st.stdev(v)/math.sqrt(len(v)),4))
    # within-session fold noise
    if reg=='Within-Session':
        diffs=[]
        for t in tasks:
            for s in sess:
                f=R['MAPA'][t][s]
                if len(f)==2: diffs.append(f[0]-f[1])
        print('MAPA fold0-fold1 per task-session sd',round(st.stdev(diffs),4),'n',len(diffs))
    # per-task paired MAPA - Ridge noise: sd over sessions of per-task diffs
    sds=[st.stdev([st.mean(R['MAPA'][t][s])-st.mean(R['Ridge6'][t][s]) for s in sess]) for t in tasks]
    print('per-task MAPA-Ridge sd across sessions (median)',round(st.median(sds),4),'-> per-task SE',round(st.median(sds)/math.sqrt(len(sess)),4))
    # MDE at 80% power alpha .05 two-sided paired: (1.96+0.84)*SE, using MAPA-Ridge sd as proxy
    o=smac('Ridge6'); d=[m[s]-o[s] for s in sess]; se=st.stdev(d)/math.sqrt(len(d))
    print('MDE80 (paired, sd from MAPA-Ridge)',round(2.8*se,4), ' using t crit',round((2.2+0.88)*se,4))
    # LinRaw vs LinSpec word_index
    for k in ['LinRaw','LinSpec']:
        if k in R and 'word_index' in R[k]: print(k,'word_index',round(st.mean(st.mean(v) for v in R[k]['word_index'].values()),3),'macro',round(st.mean(st.mean(st.mean(v) for v in R[k][t].values()) for t in R[k]),3))

print('\n\n==== subject-level ====')
import itertools
for reg in ['Within-Session','Cross-Session','Cross-Subject']:
    R=load(reg); sess=sorted(R['MAPA']['onset'])
    subj=sorted(set(s.split('_')[0] for s in sess))
    def subm(k): 
        return {u:st.mean(st.mean(st.mean(R[k][t][s]) for t in tasks) for s in sess if s.split('_')[0]==u) for u in subj}
    m=subm('MAPA')
    for k in ['Ridge6','DIVERfz','CNN','PopT']:
        if k not in R or len(R[k])<15 or set(R[k]['onset'])!=set(sess): continue
        o=subm(k); d=[m[u]-o[u] for u in subj]; n=len(d)
        # exact one-sided sign flip
        md=st.mean(d); cnt=sum(1 for signs in itertools.product([1,-1],repeat=n) if st.mean([x*s for x,s in zip(d,signs)])>=md-1e-12)
        print(reg,f'MAPA-{k}: subj n={n} mean {md:+.4f} sd {st.stdev(d):.4f} SE {st.stdev(d)/math.sqrt(n):.4f} exact one-sided p {cnt/2**n:.4f}')
    # session-pair correlation of MAPA-Ridge diff within subject
    d={s:st.mean(st.mean(R['MAPA'][t][s]) for t in tasks)-st.mean(st.mean(R['Ridge6'][t][s]) for t in tasks) for s in sess}
    pairs=[(d[a],d[b]) for a in sess for b in sess if a<b and a.split('_')[0]==b.split('_')[0]]
    xs,ys=zip(*pairs)
    if len(pairs)>2:
        mx,my=st.mean(xs),st.mean(ys); r=sum((x-mx)*(y-my) for x,y in pairs)/math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
        print(reg,'within-subject session-pair corr of MAPA-Ridge diff',round(r,2))
# per-task: where MAPA is not best
for reg in ['Within-Session','Cross-Session','Cross-Subject']:
    R=load(reg)
    out=[]
    for t in tasks:
        sc={k:st.mean(st.mean(v) for v in R[k][t].values()) for k in R if t in R[k]}
        b=max(sc,key=sc.get)
        if b!='MAPA': out.append(f'{t}:{b}+{sc[b]-sc["MAPA"]:.3f}')
    print(reg,'tasks where MAPA not best:',out)
