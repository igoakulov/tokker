import os
import importlib
import sys
import types
import unittest
from typing import cast


class TestRuntimeEnvironment(unittest.TestCase):
    def setUp(self):
        # Ensure a clean import of tokker.runtime each test
        self.module_name = "tokker.runtime"
        if self.module_name in sys.modules:
            del sys.modules[self.module_name]

        # Backup current environment we might modify and restore later
        self.env_backup = dict(os.environ)

    def tearDown(self):
        # Restore environment to the previous state
        os.environ.clear()
        os.environ.update(self.env_backup)
        # Clean import cache for next tests
        if self.module_name in sys.modules:
            del sys.modules[self.module_name]

    def test_env_defaults_are_set_when_missing(self):
        # Unset relevant env vars to test defaults
        os.environ.pop("TRANSFORMERS_NO_TF_WARNING", None)
        os.environ.pop("TRANSFORMERS_NO_ADVISORY_WARNINGS", None)
        os.environ.pop("GOOGLE_CLOUD_LOCATION", None)

        # Provide a stub for transformers.utils.logging so import doesn't fail
        # Build a fake transformers.utils.logging module structure
        transformers_pkg = cast(types.ModuleType, types.ModuleType("transformers"))
        utils_pkg = cast(types.ModuleType, types.ModuleType("transformers.utils"))
        logging_mod = cast(types.ModuleType, types.ModuleType("transformers.utils.logging"))

        def set_verbosity_error():
            # no-op; just to simulate that it exists
            return None

        logging_mod.set_verbosity_error = set_verbosity_error  # type: ignore[attr-defined]
        utils_pkg.logging = logging_mod  # type: ignore[attr-defined]
        transformers_pkg.utils = utils_pkg  # type: ignore[attr-defined]

        # Inject fakes into sys.modules so importlib can resolve it
        sys.modules["transformers"] = transformers_pkg
        sys.modules["transformers.utils"] = utils_pkg
        sys.modules["transformers.utils.logging"] = logging_mod

        # Import the runtime module to trigger env setup
        runtime = importlib.import_module(self.module_name)
        self.assertIsNotNone(runtime)

        # Assert defaults were set
        self.assertEqual(os.environ.get("TRANSFORMERS_NO_TF_WARNING"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS"), "1")
        self.assertEqual(os.environ.get("GOOGLE_CLOUD_LOCATION"), "us-central1")

    def test_env_defaults_do_not_override_existing(self):
        # Pre-set env vars to verify they are not overridden
        os.environ["TRANSFORMERS_NO_TF_WARNING"] = "0"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "0"
        os.environ["GOOGLE_CLOUD_LOCATION"] = "europe-west3"

        # Provide minimal fake transformers logging module
        transformers_pkg = cast(types.ModuleType, types.ModuleType("transformers"))
        utils_pkg = cast(types.ModuleType, types.ModuleType("transformers.utils"))
        logging_mod = cast(types.ModuleType, types.ModuleType("transformers.utils.logging"))

        def set_verbosity_error():
            return None

        logging_mod.set_verbosity_error = set_verbosity_error  # type: ignore[attr-defined]
        utils_pkg.logging = logging_mod  # type: ignore[attr-defined]
        transformers_pkg.utils = utils_pkg  # type: ignore[attr-defined]

        sys.modules["transformers"] = transformers_pkg
        sys.modules["transformers.utils"] = utils_pkg
        sys.modules["transformers.utils.logging"] = logging_mod

        # Import the runtime module to trigger env setup
        runtime = importlib.import_module(self.module_name)
        self.assertIsNotNone(runtime)

        # Ensure pre-set values preserved
        self.assertEqual(os.environ.get("TRANSFORMERS_NO_TF_WARNING"), "0")
        self.assertEqual(os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS"), "0")
        self.assertEqual(os.environ.get("GOOGLE_CLOUD_LOCATION"), "europe-west3")

    def test_handles_missing_transformers_gracefully(self):
        # Ensure any previous fake transformers modules are cleared
        for name in list(sys.modules.keys()):
            if name == "transformers" or name.startswith("transformers."):
                del sys.modules[name]

        # Import should not raise even if transformers isn't available
        runtime = importlib.import_module(self.module_name)
        self.assertIsNotNone(runtime)

        # Defaults still set
        self.assertEqual(os.environ.get("TRANSFORMERS_NO_TF_WARNING"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS"), "1")
        self.assertEqual(os.environ.get("GOOGLE_CLOUD_LOCATION"), "us-central1")


if __name__ == "__main__":
    unittest.main()
