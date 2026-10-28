# SmolVLA SO-101 write-up

[Read the article](https://ndsmn.github.io/smolVLA/) or serve this directory locally:

```bash
python3 -m http.server 8000 --directory docs
```

## Contents

- `index.html`, `style.css`, `article.js`: article and interactive results.
- `demo.mp4`: captioned simulation demo.
- `assets/proof/`: full sampled success/failure clips and trace manifest.
- `output/pdf/so101_smolvla_report.pdf`: training and evaluation report.
- `evidence/`: 200 outcome records, 100 main physical traces, training logs and audit results.
- [Architecture](architecture.md): policy interface, control loop and metrics.
- [Credits](CREDITS.md): component attribution and licenses.

Run `python3 scripts/verify_evidence.py` from the repository root to check the reported metrics, local links and artifact hashes.
