"""Verify public evidence and links without training, inference or third-party packages."""
from pathlib import Path
from html.parser import HTMLParser
from math import comb, sqrt
from urllib.parse import urlsplit, unquote
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
EVIDENCE = DOCS / 'evidence'


def read(path):
    return json.loads(path.read_text())


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    r = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return [c-r, c+r]


def paired_check(folder, expected):
    a = {r['seed']: r for r in read(folder / 'upstream_episodes.json')}
    b = {r['seed']: r for r in read(folder / 'source_visual_episodes.json')}
    assert set(a) == set(b) == set(range(35000, 35050))
    assert all(r['valid'] for r in [*a.values(), *b.values()])
    counts = [sum(r['strict_bin_success'] for r in rows.values()) for rows in (a, b)]
    assert counts == expected, (counts, expected)
    cells = dict(both_success=0, only_upstream=0, only_source_visual=0, both_failure=0)
    for seed in a:
        x, y = a[seed]['strict_bin_success'], b[seed]['strict_bin_success']
        key = 'both_success' if x and y else 'only_upstream' if x else 'only_source_visual' if y else 'both_failure'
        cells[key] += 1
    discordant = cells['only_upstream'] + cells['only_source_visual']
    small = min(cells['only_upstream'], cells['only_source_visual'])
    p = min(1., 2*sum(comb(discordant, k) for k in range(small+1))/2**discordant) if discordant else 1.
    summary = read(folder / 'paired_statistics.json')
    assert summary['paired_counts'] == cells
    assert abs(summary['mcnemar_exact_two_sided_p'] - p) < 1e-12
    return a, b, p


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        self.ids.update([attrs['id']] if 'id' in attrs else [])
        self.links.extend(attrs[key] for key in ('href', 'src', 'poster') if key in attrs)


def main():
    a, b, p4 = paired_check(EVIDENCE, [17, 21])
    _, _, p0 = paired_check(EVIDENCE / 'render0', [6, 14])
    metrics = read(EVIDENCE / 'metrics.json')
    assert metrics['total_policy_rollouts_in_render_material_study'] == 200
    for profile, rows in [('upstream', a), ('source_visual', b)]:
        for stage, expected in metrics['stage_counts'][profile].items():
            assert sum(r[stage] for r in rows.values()) == expected
        k = sum(r['strict_bin_success'] for r in rows.values())
        assert all(abs(x-y) < 1e-12 for x, y in zip(wilson(k, 50), metrics['profiles'][profile]['wilson_95_interval']))
        traces = list((EVIDENCE / 'traces' / profile).glob('seed_*_trace.npz'))
        assert len(traces) == 50
        assert all((EVIDENCE / 'traces' / profile / f'seed_{seed}_trace.npz').exists() for seed in rows)
    offline = read(EVIDENCE / 'offline_checkpoint_metrics.json')
    assert [r['step'] for r in offline] == list(range(0, 40001, 5000))
    assert all(r['anchors'] == r['distinct_trajectory_groups'] == 128 for r in offline)
    reduction = 1 - offline[-1]['action_mae_radians'] / offline[0]['action_mae_radians']
    assert abs(reduction - metrics['archived_training']['relative_action_mae_reduction']) < 1e-12
    for clip in read(DOCS / 'assets/proof/manifest.json')['clips']:
        trace = DOCS / clip['portable_trace']
        assert hashlib.sha256(trace.read_bytes()).hexdigest() == clip['source_trace_sha256']
    parser = Links()
    parser.feed((DOCS / 'index.html').read_text())
    missing = []
    for url in parser.links:
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc:
            continue
        path = DOCS / unquote(parsed.path)
        if not parsed.path:
            if parsed.fragment and parsed.fragment not in parser.ids:
                missing.append(url)
        elif not path.exists():
            missing.append(url)
    for doc in [ROOT / 'README.md', *DOCS.glob('*.md'), ROOT / 'env_rebuild/README.md', ROOT / 'training/README.md']:
        for url in re.findall(r'\]\(([^)]+)\)', doc.read_text()):
            parsed = urlsplit(url)
            if not parsed.scheme and not parsed.netloc and parsed.path and not (doc.parent/unquote(parsed.path)).exists():
                missing.append(f'{doc.relative_to(ROOT)}: {url}')
    assert not missing, missing
    manifest = read(EVIDENCE / 'file_manifest.json')
    for name, record in manifest['files'].items():
        path = DOCS / name
        assert path.stat().st_size == record['bytes'], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], name
    print(json.dumps({'verified': True, 'policy_outcomes': 200, 'main_physical_traces': 100,
                      'main_material_p': p4, 'msaa0_material_p': p0,
                      'offline_action_mae_reduction': reduction,
                      'hashed_artifacts': len(manifest['files']), 'missing_local_links': 0}, indent=2))


if __name__ == '__main__':
    main()
