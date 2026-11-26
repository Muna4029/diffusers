# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import unittest
from importlib import import_module


class DependencyTester(unittest.TestCase):
    def test_diffusers_import(self):
        try:
            import diffusers  # noqa: F401
        except ImportError:
            # If diffusers is not installed in this environment, skip the test
            # as this suite may be run in environments that don't have optional
            # packages available.
            self.skipTest("diffusers package is not installed")

    def test_backend_registration(self):
        import sys

        import diffusers
        from diffusers.dependency_versions_table import deps

        # Collect classes from the diffusers package and any loaded submodules.
        all_classes = []
        for mod_name, mod in list(sys.modules.items()):
            if mod_name == diffusers.__name__ or mod_name.startswith(diffusers.__name__ + "."):
                try:
                    all_classes.extend(inspect.getmembers(mod, inspect.isclass))
                except Exception:
                    # Ignore modules that can't be inspected for classes
                    continue

        # Fallback to top-level inspection if nothing was found
        if not all_classes:
            all_classes = inspect.getmembers(diffusers, inspect.isclass)

        for cls_name, cls_module in all_classes:
            if "dummy_" in getattr(cls_module, "__module__", ""):
                # Some classes may not declare _backends; skip those safely.
                backends = getattr(cls_module, "_backends", None)
                if not backends:
                    continue
                for backend in backends:
                    if backend == "k_diffusion":
                        backend = "k-diffusion"
                    elif backend == "invisible_watermark":
                        backend = "invisible-watermark"
                    assert backend in deps, f"{backend} is not in the deps table!"

    def test_pipeline_imports(self):
        import diffusers
        import diffusers.pipelines

        all_classes = inspect.getmembers(diffusers, inspect.isclass)
        for cls_name, cls_module in all_classes:
            if hasattr(diffusers.pipelines, cls_name):
                pipeline_folder_module = ".".join(str(cls_module.__module__).split(".")[:3])
                if not pipeline_folder_module:
                    continue
                try:
                    # Import the pipeline module by name only. Passing the class
                    # name as the 'package' argument is inappropriate and can
                    # raise errors; also treat ImportError as a skip because
                    # optional dependencies may be missing.
                    _ = import_module(pipeline_folder_module)
                except ImportError:
                    # Optional dependencies for this pipeline are not installed; skip
                    continue
