import json,glob,os,statistics as st
roots=glob.glob('/tmp/np_probe/leaderboard/*')+glob.glob('/tmp/np_pr20/leaderboard/Ridge*')
tasks=['onset','speech','volume','delta_volume','pitch','word_index','word_gap','gpt2_surprisal','word_head_pos','word_part_speech','word_length','global_flow','local_flow','frame_brightness','face_num']
want={'MAPA':'MAPA','DIVER-1(0.1s,tiny,frozen)':'DIVERfz','DIVER-1(0.1s,tiny)':'DIVERft','PopulationTransformer_Lap':'PopT','CNN_Lap':'CNN','Ridge_6':'Ridge6','Linear_Laplacian':'LinLap','MLP_Lap':'MLPLap','BrainBERT_frozen_global':'BBglob'}
def load(reg):
    res={}
    for r in roots:
        b=os.path.basename(r)
        k=next((v for p,v in want.items() if b.startswith(p)),None)
        if not k or not os.path.isdir(f'{r}/{reg}'): continue
        d={}
        for t in tasks:
            f=f'{r}/{reg}/population_{t}.json'
            if not os.path.exists(f): continue
            j=json.load(open(f))['evaluation_results']
            d[t]={s:st.mean(x['test_roc_auc'] for x in v['population']['one_second_after_onset']['folds']) for s,v in j.items()}
        res[k]=d
    return res
import math
for reg in ['Within-Session','Cross-Session','Cross-Subject']:
    R=load(reg)
    print('\n##',reg, sorted(R))
    pairs=[('MAPA','Ridge6'),('MAPA','PopT'),('MAPA','DIVERfz'),('Ridge6','CNN'),('PopT','CNN'),('DIVERfz','PopT'),('DIVERfz','DIVERft'),('Ridge6','LinLap'),('CNN','LinLap')]
    for a,b in pairs:
        if a not in R or b not in R: continue
        line=[]
        for t in tasks:
            ss=sorted(set(R[a][t])&set(R[b][t]))
            d=[R[a][t][s]-R[b][t][s] for s in ss]
            m=st.mean(d); se=st.stdev(d)/math.sqrt(len(d))
            line.append(f'{t[:9]}:{m:+.3f}({se:.3f})')
        ss=sorted(R[a]['onset'])
        dm=[st.mean(R[a][t][s] for t in tasks)-st.mean(R[b][t][s] for t in tasks) for s in ss]
        print(f'{a}-{b} macro {st.mean(dm):+.3f} se {st.stdev(dm)/math.sqrt(len(dm)):.3f}')
        print('   '+' '.join(line))
    # subject 7 per task
    for s in ['btbank7_0','btbank7_1']:
        print(s, ' '.join(f"{t[:9]}:"+'/'.join(f"{R[k][t][s]:.2f}" for k in ['MAPA','Ridge6','CNN','PopT'] if k in R and s in R[k][t]) for t in tasks))
