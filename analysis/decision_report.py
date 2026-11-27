"""Print completed decision evidence and explicitly pending phases, without opening trial data.

From the server code directory: ../.venv/bin/python -m analysis.decision_report
"""
import json
from collections import Counter

from common import ledger, lock


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def report(root):
    c, e, o = root/'classical', root/'ensemble', root/'overnight'
    comparisons = [row for row in ledger.read(root) if row['kind'] == 'comparison']

    def paired(phase, regime, arm, baseline, wave=None):
        for row in reversed(comparisons):
            if (row['phase'],row['regime'],row['arm'],row['baseline']) != (phase,regime,arm,baseline):
                continue
            extra = json.loads(row['extra'] or '{}')
            if wave is not None and (extra.get('wave') != wave or row['decision'] == 'Descriptive'
                    and extra.get('question') != 'threshold table against the regime baseline'):
                continue
            return {**{k:row[k] or None for k in ('score','baseline_score','delta','subjects_up','n_subjects',
                    'p_signflip','ci_lo','ci_hi','decision')}, 'per_subject':extra.get('per_subject'),
                    'session_5_0':extra.get('per_session', {}).get('btbank5_0')}
        return None

    results = {}
    for r in ('within','cross_subject','cross_session'):
        parity = read(c/f'c0_{r}.json')
        out = {'C0':{k:parity[k] for k in ('macro','published','gap','decision')} if parity else 'pending'}
        for name in ('c1','c2'):
            v=read(c/f'{name}_{r}.json')
            if not v:
                out[name.upper()]='pending'; continue
            keep=('kept_blocks','block_weights','block_gains','block_controls','kept_weight','copied_from_within',
                  'member','classical_member','weight','baseline','decision','delta','controls','side')
            detail={k:v[k] for k in keep if k in v}
            arm=v.get('classical_member') if name=='c1' else v['member']
            detail['paired']=paired(name,r,arm,'ridge6' if name=='c1' else v['baseline']) if arm else None
            out[name.upper()]=detail
        out['C3']=read(c/f'controls_{r}.json') or 'pending'
        for w in ((1,2) if r=='within' else (1,)):
            screen=read(e/f'e1_{r}_w{w}.json')
            if not screen:
                out[f'E1_wave{w}']='pending'; continue
            best=screen['best']; out[f'E1_wave{w}']={k:screen[k] for k in (
                'pool','passed','best','leave_one_member_out','self_ensemble','mapa_pretraining_split')}
            out[f'E1_wave{w}']['paired']=paired('E1',r,best['name'],screen['baseline'],w)
        if r=='within':
            out['E2']=read(e/'e2_within_w2.json') or 'pending'
            out['E3']=read(e/'e3_within_w2.json') or 'pending'
        out['selection']=read(o/f'selection_{r}.json') or 'pending; not frozen'
        out['V']=read(o/f'confirmation_V_{r}.json') or 'not verified'
        out['T']=read(o/f'confirmation_T_{r}.json') or 'not verified'
        out['blocked']=read(o/f'confirmation_blocked_{r}.json')
        for tier in ('V','T'):
            if isinstance(out[tier],dict):
                extra=json.loads(out[tier].pop('extra') or '{}'); out[tier]['per_subject']=extra.get('per_subject')
        results[r]=out
    return {'regimes':results,'trained_descriptive':[read(p) for p in sorted(e.glob('quick_*.json'))],
            'test_access_log_entries_by_purpose':dict(Counter(x['purpose'] for x in lock.access_log(root))),
            'disclosures':['Within DIVER was admitted after a 17-subset Tier P look; E1/E3 are not independent replication.',
                           'Only regenerated float32 CNN is reported. CNN is outside the restricted decision.',
                           'DIVER cross-subject has no registered protocol: not applicable, not Stop.']}


if __name__=='__main__':
    print(json.dumps(report(ledger.results_root()),indent=2))
