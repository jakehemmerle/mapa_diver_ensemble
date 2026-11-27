"""One regime, one committed freeze, one V veto, and T only on V Go. No design selection."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from common import ledger, lock, mapa_features, predictions, ridge, splits
from classical import features
from ensemble.run import missing_predictions, quick_confirmation
from server.overnight_freeze import REGIMES, get, root, save, states

code = Path(__file__).resolve().parents[1]
# Pending V/T DIVER exports: three workers on each of GPUs 0 and 1, two of the 12 evaluation
# sessions each (a worker is bound by one Python thread at ~13% GPU activity and ~16 GB RSS).
# Each shard is the one writer for its session pairs.
_GPU0, _GPU1 = 'GPU-013cce7e-9628-49c2-f849-251f3304d038', 'GPU-ad6718d6-1cf2-f3e7-2f2b-515f9c00859f'
DIVER_CARDS = (_GPU0, _GPU1) * 3


def run(*command, **env):
    print('exec', *command, flush=True)
    subprocess.run(command, check=True, cwd=code, env={**os.environ, **env})


def module(name, *args):
    run(sys.executable, '-u', '-m', name, *args, CUDA_VISIBLE_DEVICES='')


def attested(r):
    """Require the frozen bytes themselves in Git, not merely the commit of the freezer."""
    marker = get(code, 'COMMIT') or {}
    digest = marker.get('frozen_design_sha256', {}).get(r)
    if marker.get('dirty') is not False or not digest:
        return None
    frozen = (root / 'frozen' / (r + '.json')).read_bytes()
    if hashlib.sha256(frozen).hexdigest() != digest or frozen != (code / 'frozen-designs' / (r + '.json')).read_bytes():
        raise ValueError('committed and live frozen designs differ')
    return marker['commit']


def await_design(r):
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'inotify_init1')
    try:
        for folder in (states, code):
            if libc.inotify_add_watch(fd, os.fsencode(folder), 0x80 | 0x08) < 0:
                raise OSError(ctypes.get_errno(), str(folder))
        while True:
            selection = get(states, 'selection_' + r)
            if selection:
                if selection['decision'] != 'frozen':
                    return None
                if commit := attested(r):
                    return commit
            os.read(fd, 8192)
    finally:
        os.close(fd)


def data(r, tier, design):
    classical = design['track'] == 'classical'
    spec = design['spec']
    members = set() if classical else set(spec['members'])
    mapa = sorted({'mapa12'} | (members & {'mapa0'}))
    required = {'ridge6', *mapa, *members}
    if classical:
        required.add(design['member'])
    for cell in splits.list_cells(tier, r):
        for tap in (0, 12) if classical or 'mapa0' in members else (12,):
            path = mapa_features.kernel_file('E', r, cell.key(), tap)
            if not path.is_file():
                raise FileNotFoundError(path)
    module('ensemble.run', 'export', '--tier', tier, '--regime', r, '--members', *mapa)
    phase = 'c4' if tier == 'V' else 'c5'
    module('classical.run', phase, '--regimes', r, '--workers', '2', '--fit-workers', '2', '--stages', 'extract', 'fit')
    if 'diver' in members:
        mode = '--quick' if quick_confirmation(design, root) else '--oof-only'
        shards = [subprocess.Popen(('bash', 'ensemble/launch/e0.sh', card, '--wave', '2', '--member', 'diver',
                                    '--tier', tier, '--regime', r, mode, '--export-only',
                                    '--gpu-shard', f'{i}/{len(DIVER_CARDS)}'),
                                   cwd=code, env={**os.environ, 'NPEXP_WAIT': '1'})
                  for i, card in enumerate(DIVER_CARDS)]
        failed = [shard.args for shard in shards if shard.wait() != 0]   # wait for every shard
        if failed:
            raise RuntimeError(f'DIVER export shards failed: {failed}')
    if not classical and 'cnn' in members:
        raise ValueError('CNN is not in the authorized wave-2 decision')
    for m in required:
        absent = missing_predictions(m, tier, r, root)
        if absent:
            raise ValueError(f'{m}: {len(absent)} prediction files missing')
    # The C0 Ridge6 baseline on T predates fit-row centering. Only if the frozen
    # stacker actually uses it, repair the OOF part without changing its full fit.
    if tier == 'T' and spec.get('rung') == 'b' and 'ridge6' in members:
        for c in splits.list_cells(tier, r):
            p = predictions.read(c, 'ridge6', root); s = splits.get_cell(c)
            k, _ = features.kernels_and_dim(c, 'base6', 'window', 'bipolar')
            gram, _ = ridge.combine([k]); del k, _
            p.rows['train'] = s.rows['train']
            p.scores['train'] = ridge._out_of_fold(gram, s.labels('train').astype('float64'), s.oof_blocks(), p.meta['lam']); del gram
            p.meta['oof_refresh_commit'] = ledger.commit()
            predictions.write(c, 'ridge6', p.rows, p.scores, p.meta, root)


def confirm(r, commit):
    if attested(r) is None:
        raise ValueError('missing clean freeze receipt')
    design = lock.frozen(r, root); candidate = lock.candidate_id(design)
    amendment = {}
    if r == 'within':
        from server.confirm_amendment import attest
        audit = attest(root, commit)
        amendment = {'protocol': audit['protocol'], 'amended_release': commit,
                     'prior_blocked_sha256': audit['original_blocked_sha256']}
    classical = design['track'] == 'classical'
    for tier in ('V', 'T'):
        comparison = [row for row in ledger.read(root) if row['tier'] == tier and row['regime'] == r
                      and row['candidate'] == candidate and row['kind'] == 'comparison']
        if not comparison:
            ledger.guard_candidate(root, tier, r, candidate)
            attempt = states / f'attempt_{tier}_{r}.json'
            if attempt.exists():
                raise RuntimeError(f'{attempt} exists without a completed comparison; no repeated score')
            data(r, tier, design)
            if not attested(r):
                raise ValueError('frozen design lost its commit attestation before scoring')
            with attempt.open('x') as f:
                json.dump({'candidate': candidate, 'freeze_receipt_commit': commit, **amendment}, f)
            if classical:
                module('classical.run', 'c4' if tier == 'V' else 'c5', '--regimes', r, '--stages', 'score')
            else:
                module('ensemble.run', 'e4' if tier == 'V' else 'e5', '--regime', r)
            comparison = [row for row in ledger.read(root) if row['tier'] == tier and row['regime'] == r
                          and row['candidate'] == candidate and row['kind'] == 'comparison']
        if len(comparison) != 1:
            raise RuntimeError(f'expected one completed {tier} comparison')
        row = comparison[0]
        save('confirmation_' + tier + '_' + r, {**amendment, **{k:row[k] for k in (
            'tier','regime','candidate','arm','baseline','score','baseline_score','delta','subjects_up',
            'n_subjects','p_signflip','ci_lo','ci_hi','decision','extra')}})
        if tier == 'V' and row['decision'] != 'Go':
            print('V veto; T stays closed', r, flush=True)
            return


def main():
    p = argparse.ArgumentParser(); p.add_argument('regime', choices=REGIMES); r = p.parse_args().regime
    states.mkdir(exist_ok=True)
    try:
        commit = await_design(r)
        if not commit:
            return
        with (states / 'confirmation.lock').open('a') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            confirm(r, commit)
    except Exception as error:
        dest = 'confirmation_amended_blocked_' if r == 'within' else 'confirmation_blocked_'
        save(dest + r, {'decision':'blocked','reason':str(error)})
        raise


if __name__ == '__main__':
    main()
