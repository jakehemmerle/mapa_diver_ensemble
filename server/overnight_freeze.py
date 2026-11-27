"""Advance the registered November 25 candidates when both Tier P tracks are terminal.

Server-only, event-driven. Never reads trial labels. Missing evidence means wait, not Stop.
The V/T launcher separately requires the exact frozen JSON in the clean synced code tree.
"""
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

from common import ledger, lock

REGIMES = ('within', 'cross_subject', 'cross_session')
RESTRICTED = {'mapa12', 'ridge6', 'diver'}
root = ledger.results_root()
states = root / 'overnight'


def get(folder, name):
    try:
        return json.loads((folder / (name + '.json')).read_text())
    except FileNotFoundError:
        return None


def save(name, result):
    tmp = states / (name + '.tmp')
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, states / (name + '.json'))


def ensemble_done(r):
    e = root / 'ensemble'
    first = get(e, f'e1_{r}_w1')
    if not first:
        return False
    if set(first['pool']) != {'mapa12', 'mapa0', 'ridge6'} or not first['self_ensemble']['ran']:
        raise ValueError('wave-1 pool or self-control does not match the registered screen')
    if r == 'cross_session':
        return True  # descriptive; never select on its two subjects
    if r == 'cross_subject':
        if first['passed'] or first['best']['decision'] != 'Stop' or first['leave_one_member_out'].get('ridge6', {}).get('keep'):
            raise ValueError('cross-subject ensemble is not terminal at the verified wave-1 Stop')
        return True
    second = get(e, 'e1_within_w2')
    if not second:
        return False
    if set(second['pool']) != RESTRICTED or set(second.get('pool_restricted_to') or []) != RESTRICTED:
        raise ValueError('refusing an unrestricted or incomplete DIVER decision')
    checks = get(e, 'e0_P_within') or {}
    if not all(checks.get(m, {}).get('passed') for m in RESTRICTED):
        raise ValueError('restricted pool lacks an E0 pass')
    if not second['passed']:
        return True
    ladder, controls = get(e, 'e2_within_w2'), get(e, 'e3_within_w2')
    if not ladder or not controls:
        return False
    if set(ladder['members']) != set(second['best']['members']) or not controls['e1_passed']:
        raise ValueError('E2/E3 do not correspond to the official E1 winner')
    if controls.get('candidate'):
        proposal = next((p for p in lock.proposals('within', root) if p['track'] == 'ensemble'), None)
        if not proposal or (proposal['member'], proposal['decision'], proposal['controls_pass']) != (
                controls['candidate'], controls['decision'], controls['controls_pass']):
            raise ValueError('E3 proposal does not match completed controls')
    return True


def advance(r):
    c = root / 'classical'
    c1, c2, c3 = get(c, 'c1_' + r), get(c, 'c2_' + r), get(c, 'controls_' + r)
    if not (c1 and c2 and c3 and ensemble_done(r)):
        return
    if get(Path(__file__).resolve().parents[1], 'COMMIT').get('dirty') is not False:
        raise ValueError('selection requires clean deployed code')
    proposals = lock.proposals(r, root)
    if r != 'cross_session':
        ensemble = next((p for p in proposals if p['track'] == 'ensemble'), None)
        final = get(root / 'ensemble', 'e3_within_w2') if r == 'within' else None
        if ensemble and (not final or final.get('candidate') != ensemble['member']):
            raise ValueError('stale ensemble proposal without its completed E3 decision')
    classical = next((p for p in proposals if p['track'] == 'classical'), None)
    if not classical or (classical['member'], classical['decision'], classical['controls_pass']) != (
            c2['member'], c2['decision'], c3['c2']['controls_pass']):
        raise ValueError('C2 proposal does not match completed C3 controls')
    if r == 'cross_session':
        within = get(states, 'selection_within')
        if not within:
            return
        if within['decision'] == 'blocked':
            raise ValueError('within-session selection is blocked; no design to inherit')
        eligible = within['decision'] == 'frozen'
        if eligible:
            source = lock.frozen('within', root)
            if not any(p['track'] == source['track'] and p['config_hash'] == source['config_hash'] for p in proposals):
                raise ValueError('cross-session did not inherit the identical within-session design')
    else:
        eligible = any(lock.eligible(p) for p in proposals)
    frozen = get(root / 'frozen', r)
    if eligible:
        if not frozen:
            subprocess.run([sys.executable, '-m', 'ensemble.run', 'freeze', '--regime', r], check=True)
            frozen = lock.frozen(r, root)
        result = {'decision': 'frozen', 'candidate': lock.candidate_id(frozen), 'member': frozen['member']}
    else:
        if frozen:
            raise ValueError('ineligible regime already has a freeze file')
        result = {'decision': 'no eligible candidate', 'proposals': [
            {k:p[k] for k in ('track','member','decision','controls_pass','tier_p_gain')} for p in proposals]}
        ledger.append(dict(tier='P', regime=r, phase='FREEZE', kind='info', arm='none',
                           decision='no eligible candidate', extra=result), root)
    save('selection_' + r, result)
    print(r, result, flush=True)


def main():
    states.mkdir(exist_ok=True)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    for folder in [root / 'classical', root / 'ensemble', states]:
        if libc.inotify_add_watch(fd, os.fsencode(folder), 0x80) < 0:  # IN_MOVED_TO: atomic JSON writes
            raise OSError(ctypes.get_errno(), str(folder))
    while True:
        for r in REGIMES:
            if get(states, 'selection_' + r):
                continue
            try:
                advance(r)
            except Exception as error:
                save('selection_' + r, {'decision': 'blocked', 'reason': str(error)})
                print(r, 'blocked:', error, flush=True)
        if all(get(states, 'selection_' + r) for r in REGIMES):
            return
        os.read(fd, 8192)


if __name__ == '__main__':
    main()
