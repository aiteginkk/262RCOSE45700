"""Step 0: find what profiling support the installed furiosa-torch offers.

Run on the NPU server from the repository root:

    python experiments/profiling/find_profiler.py

Read-only. It does not run a model. It lists:
- furiosa.* modules and attributes whose names mention profiling or tracing
- whether torch.profiler exposes a PrivateUse1 (custom device) activity
- FURIOSA_* environment variable names mentioned inside the installed package
"""
import importlib
import pkgutil
import re
from pathlib import Path

import torch
import furiosa
import furiosa.torch

KEYWORDS = re.compile(r"profil|trace|tracing|perfetto|timeline", re.IGNORECASE)

print("== torch.profiler activities ==")
from torch.profiler import ProfilerActivity

activities = [a.name for a in ProfilerActivity]
print("Available:", activities)
print("PrivateUse1 activity available:", "PrivateUse1" in activities)
print("PrivateUse1 backend name:", torch._C._get_privateuse1_backend_name())

print("\n== furiosa modules mentioning profiling/tracing ==")
found_modules = []
for root in furiosa.__path__:
    for info in pkgutil.walk_packages([root], prefix="furiosa."):
        if KEYWORDS.search(info.name):
            found_modules.append(info.name)
for name in sorted(set(found_modules)):
    print(" ", name)
if not found_modules:
    print("  (none)")

print("\n== attributes mentioning profiling/tracing ==")
for module_name in ["furiosa.torch", "furiosa.torch.config", *found_modules]:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        print(f"  {module_name}: import failed ({error})")
        continue
    hits = [attr for attr in dir(module) if KEYWORDS.search(attr)]
    if hits:
        print(f"  {module_name}: {hits}")

print("\n== FURIOSA_* environment variables mentioned in the package ==")
env_names = set()
for root in furiosa.__path__:
    for path in Path(root).rglob("*.py"):
        try:
            env_names.update(re.findall(r"FURIOSA_[A-Z0-9_]+", path.read_text(errors="ignore")))
        except OSError:
            pass
for name in sorted(env_names):
    marker = "  <-- profiling related" if KEYWORDS.search(name) else ""
    print(f"  {name}{marker}")
if not env_names:
    print("  (none found in .py files; they may live in compiled extensions)")
