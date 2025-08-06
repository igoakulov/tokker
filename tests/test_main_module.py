import unittest
from unittest.mock import patch, Mock
import sys


class TestMainModule(unittest.TestCase):
    def test_main_successful_dispatch_returns_zero(self):
        # Patch tokenize.main where it's imported into __main__, and ensure argv triggers dispatch
        with patch("tokker.cli.tokenize.main", return_value=0) as mock_cli_main, \
             patch("sys.argv", ["tok", "--models"]):
            import importlib
            if "tokker.__main__" in sys.modules:
                del sys.modules["tokker.__main__"]
            main_module = importlib.import_module("tokker.__main__")

            rc = main_module.main()
            self.assertEqual(rc, 0)
            mock_cli_main.assert_called_once()

    def test_main_handles_exceptions_and_returns_nonzero(self):
        # Simulate cli_main raising an unexpected exception and verify stderr message
        stderr_mock = Mock()
        with patch("tokker.cli.tokenize.main", side_effect=RuntimeError("boom")) as mock_cli_main, \
             patch("sys.argv", ["tok", "--models"]), \
             patch("sys.stderr", stderr_mock):
            import importlib
            if "tokker.__main__" in sys.modules:
                del sys.modules["tokker.__main__"]
            main_module = importlib.import_module("tokker.__main__")

            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            mock_cli_main.assert_called_once()
            wrote = any(
                (args and isinstance(args[0], str) and "Unexpected error" in args[0])
                for args, _ in getattr(stderr_mock, "write").call_args_list
            )
            self.assertTrue(wrote, "Expected an 'Unexpected error' message on stderr")

    def test_module_entrypoint_exits_with_return_code(self):
        # Ensure main returns the code from tokenize.main and that we can pass it to sys.exit
        with patch("tokker.cli.tokenize.main", return_value=3) as mock_cli_main, \
             patch("sys.argv", ["tok", "--models"]), \
             patch("sys.exit") as mock_exit:
            import importlib
            if "tokker.__main__" in sys.modules:
                del sys.modules["tokker.__main__"]
            main_module = importlib.import_module("tokker.__main__")

            rc = main_module.main()
            self.assertEqual(rc, 3)
            mock_cli_main.assert_called_once()
            sys.exit(rc)
            mock_exit.assert_called_with(3)


if __name__ == "__main__":
    unittest.main()
