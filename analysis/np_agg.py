import json,glob,os,collections,statistics as st
roots=glob.glob('/tmp/np_probe/leaderboard/*')+glob.glob('/tmp/np_pr20/leaderboard/Ridge*')
tasks=['onset','speech','volume','delta_volume','pitch','word_index','word_gap','gpt2_surprisal','word_head_pos','word_part_speech','word_length','global_flow','local_flow','frame_brightness','face_num']
want={'MAPA':'MAPA','DIVER-1(0.1s,tiny,frozen)':'DIVERfz','PopulationTransformer_Lap':'PopT','CNN_Lap':'CNN','Ridge_6':'Ridge6','Linear_Laplacian':'LinLap','Linear_laplacian':'LinLap'}
for reg in ['Within-Session','Cross-Session','Cross-Subject']:
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
    ks=sorted(res,key=lambda k:-st.mean(st.mean(res[k][t].values()) for t in res[k]))
    print(f'\n## {reg}\n'+'task'.ljust(18)+''.join(k.rjust(9) for k in ks)+'   best-of'.rjust(9))
    orc=[]
    for t in tasks:
        row=[st.mean(res[k][t].values()) if t in res[k] else float('nan') for k in ks]
        orc.append(max(row)); print(t.ljust(18)+''.join(f'{x:9.3f}' for x in row)+f'{max(row):9.3f}')
    print('MACRO'.ljust(18)+''.join(f'{st.mean(st.mean(res[k][t].values()) for t in res[k]):9.3f}' for k in ks)+f'{st.mean(orc):9.3f}')
    # per-subject macro
    sess=sorted(res[ks[0]][tasks[0]])
    print('per-session macro:')
    for s in sess:
        print(s.ljust(18)+''.join((f'{st.mean(res[k][t][s] for t in res[k]):9.3f}' if s in res[k][tasks[0]] else '      nan') for k in ks))
