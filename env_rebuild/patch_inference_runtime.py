"""Apply only import-registry compatibility patches to isolated LeRobot 0.5.1."""
from pathlib import Path
import importlib.metadata
import sysconfig

if importlib.metadata.version("lerobot")!="0.5.1":
    raise RuntimeError("This import-only patch is specific to the packaged LeRobot 0.5.1 runtime")
root=Path(sysconfig.get_paths()["purelib"])/"lerobot/policies"
header='''# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed to in writing, software is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
'''
(root/"__init__.py").write_text(header+'''"""Only register SmolVLA in this isolated inference environment.

Avoid eager imports of unrelated policies that fail on Python 3.12.
The SmolVLA forward pass and checkpoint weights are unchanged.
"""
from .smolvla.configuration_smolvla import SmolVLAConfig as SmolVLAConfig
__all__=["SmolVLAConfig"]
''')
(root/"groot/__init__.py").write_text(header+'''"""Expose config to the registry without eagerly importing the unrelated model."""
from .configuration_groot import GrootConfig
__all__=["GrootConfig"]
''')
print("Applied import-only compatibility patches to",root)
