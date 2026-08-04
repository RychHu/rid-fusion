# RID Fusion Studio v0.4.0

RID Fusion Studio is an explainable research prototype for normalising, associating and fusing multi-source drone Remote ID observations. It includes complete Python source code, a native Windows WPF desktop client, deterministic scenarios, file import/replay, anomaly checks, algorithm comparison, report export and tests.

This is a research and patent-demonstration prototype. It is not an RF receiver, an airworthiness product, an airspace approval service or an enforcement system.

## Quick start

```powershell
git clone https://github.com/RychHu/rid-fusion.git
cd rid-fusion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m rid_fusion.desktop_api selftest
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\desktop\Run-RIDFusion.ps1
```

See [README.md](README.md) for the complete Chinese documentation and project boundaries.
