import unittest
from unittest.mock import patch
import sys
from io import StringIO
from tokker import messages as msg


def _expected_model_not_found_with_hints(model: str, providers_str: str = "none") -> str:
    return (
        msg.MSG_DEFAULT_MODEL_UNSUPPORTED_FMT.format(model=model, providers=providers_str)
        + "\n"
        + msg.MSG_DEP_HINT_HEADING
        + "\n"
        + msg.MSG_DEP_HINT_ALL
        + "\n"
        + msg.MSG_DEP_HINT_TIKTOKEN
        + "\n"
        + msg.MSG_DEP_HINT_GOOGLE
        + "\n"
        + msg.MSG_DEP_HINT_TRANSFORMERS
        + "\n"
    )






class TestMainModule(unittest.TestCase):
    def test_main_successful_dispatch_returns_zero(self):
        # Patch tokenize.main where it's imported into __main__, and ensure argv triggers dispatch
        with (
            patch("tokker.cli.tokenize.main", return_value=0) as mock_cli_main,
            patch("sys.argv", ["tok", "--models"]),
        ):
            import importlib

            if "tokker.__main__" in sys.modules:
                del sys.modules["tokker.__main__"]
            main_module = importlib.import_module("tokker.__main__")

            rc = main_module.main()
            self.assertEqual(rc, 0)
            mock_cli_main.assert_called_once()

    def test_main_handles_exceptions_and_returns_nonzero(self):
        # Simulate cli_main raising an unexpected exception and verify stderr message
        err = "boom"
        with (
            patch(
                "tokker.cli.tokenize.main", side_effect=RuntimeError(err)
            ) as mock_cli_main,
            patch("sys.argv", ["tok", "--models"]),
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
        ):
            import importlib

            if "tokker.__main__" in sys.modules:
                del sys.modules["tokker.__main__"]
            main_module = importlib.import_module("tokker.__main__")

            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            mock_cli_main.assert_called_once()
            self.assertEqual(stdout_buf.getvalue(), "")
            expected = msg.MSG_UNEXPECTED_ERROR_FMT.format(err=err) + "\n"
            self.assertEqual(stderr_buf.getvalue(), expected)

    def test_module_entrypoint_exits_with_return_code(self):
        # Ensure main returns the code from tokenize.main and that we can pass it to sys.exit
        with (
            patch("tokker.cli.tokenize.main", return_value=3) as mock_cli_main,
            patch("sys.argv", ["tok", "--models"]),
            patch("sys.exit") as mock_exit,
        ):
            import importlib

            if "tokker.__main__" in sys.modules:
                del sys.modules["tokker.__main__"]
            main_module = importlib.import_module("tokker.__main__")

            rc = main_module.main()
            self.assertEqual(rc, 3)
            mock_cli_main.assert_called_once()
            sys.exit(rc)
            mock_exit.assert_called_with(3)


class TestMainErrorMapping(unittest.TestCase):
    def _import_main_fresh(self):
        import importlib

        if "tokker.__main__" in sys.modules:
            del sys.modules["tokker.__main__"]
        return importlib.import_module("tokker.__main__")

    @patch("sys.argv", ["tok", "Hello", "--model", "gemini-2.5-flash"])
    @patch(
        "tokker.cli.tokenize.main",
        side_effect=RuntimeError("compute_tokens failed: auth"),
    )
    def test_google_guidance_by_model_prefix(self, _cli_main):
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
            patch(
                "tokker.error_handler.get_installed_providers", return_value={"Google"}
            ),
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected_lines = [
                msg.MSG_GOOGLE_AUTH_HEADER,
                msg.MSG_GOOGLE_AUTH_GUIDE_URL,
            ]
            expected = "\n".join(expected_lines) + "\n"
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "Hello", "--model", "cl100k_base"])
    @patch(
        "tokker.cli.tokenize.main",
        side_effect=RuntimeError("Google compute_tokens request failed"),
    )
    def test_google_guidance_by_error_marker(self, _cli_main):
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
            patch(
                "tokker.error_handler.get_installed_providers", return_value={"Google"}
            ),
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            # Expect only the Google guidance block
            expected_lines = [
                msg.MSG_GOOGLE_AUTH_HEADER,
                msg.MSG_GOOGLE_AUTH_GUIDE_URL,
            ]
            expected = "\n".join(expected_lines) + "\n"
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "Hello", "--output", "bogus"])
    @patch(
        "tokker.cli.tokenize.main",
        side_effect=ValueError("Unknown output format: bogus"),
    )
    def test_unknown_output_format_maps_to_friendly_message(self, _cli_main):
        """Invalid output format should be mapped to a friendly message and nothing printed to stdout."""
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected = msg.MSG_UNKNOWN_OUTPUT_FORMAT_FMT.format(value="bogus") + "\n"
            self.assertEqual(stderr_buf.getvalue(), expected)
            self.assertNotIn("Traceback (most recent call last)", stderr_buf.getvalue())

    @patch("sys.argv", ["tok", "Hello", "--model", "not_a_real_model"])
    @patch(
        "tokker.cli.tokenize.main",
        side_effect=RuntimeError("Model not found: not_a_real_model"),
    )
    def test_explicit_model_not_found_message_and_hint(self, _cli_main):
        """When error text includes 'not found' and a model arg is present, print standardized message and hints."""
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
            patch("tokker.error_handler.get_installed_providers", return_value=set()),
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected = _expected_model_not_found_with_hints("not_a_real_model", "none")
            self.assertEqual(stderr_buf.getvalue(), expected)
            self.assertNotIn("Traceback (most recent call last)", stderr_buf.getvalue())

    @patch("sys.argv", ["tok", "Hello", "--model", "something"])
    @patch(
        "tokker.cli.tokenize.main",
        side_effect=RuntimeError("No module named 'tiktoken'"),
    )
    def test_importerror_tiktoken_hints(self, _cli_main):
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
            patch("tokker.error_handler.get_installed_providers", return_value=set()),
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected = _expected_model_not_found_with_hints("something", "none")
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "Hello", "--model", "something"])
    @patch(
        "tokker.cli.tokenize.main",
        side_effect=RuntimeError("No module named 'transformers'"),
    )
    def test_importerror_transformers_hints(self, _cli_main):
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
            patch("tokker.error_handler.get_installed_providers", return_value=set()),
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected = _expected_model_not_found_with_hints("something", "none")
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "Hello", "--model", "something"])
    @patch(
        "tokker.cli.tokenize.main",
        side_effect=RuntimeError("No module named 'google.genai'"),
    )
    def test_importerror_google_hints(self, _cli_main):
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
            patch("tokker.error_handler.get_installed_providers", return_value=set()),
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected = _expected_model_not_found_with_hints("something", "none")
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "Hello", "--model", "nonexistent-model"])
    @patch("tokker.cli.tokenize.main", side_effect=RuntimeError("random failure"))
    def test_unknown_model_hint(self, _cli_main):
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            # Expect only the fallback unexpected error line (no generic list-models hint anymore)
            expected = msg.MSG_UNEXPECTED_ERROR_FMT.format(err="random failure") + "\n"
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "Hello"])
    @patch("tokker.cli.tokenize.main", side_effect=OSError("Permission denied"))
    def test_filesystem_error_hint(self, _cli_main):
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected = (
                msg.MSG_FILESYSTEM_ERROR_FMT.format(err="Permission denied") + "\n"
            )
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "Hello"])
    @patch(
        "tokker.cli.tokenize.main",
        side_effect=ValueError("JSONDecodeError: Expecting value"),
    )
    def test_json_decode_error_hint(self, _cli_main):
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected = (
                msg.MSG_CONFIG_ERROR_FMT.format(err="JSONDecodeError: Expecting value")
                + "\n"
            )
            self.assertEqual(stderr_buf.getvalue(), expected)

    @patch("sys.argv", ["tok", "--models"])
    @patch("tokker.cli.tokenize.main", side_effect=RuntimeError("completely unknown"))
    def test_fallback_unexpected_error(self, _cli_main):
        main_module = self._import_main_fresh()
        with (
            patch("sys.stderr", new_callable=StringIO) as stderr_buf,
            patch("sys.stdout", new_callable=StringIO) as stdout_buf,
        ):
            rc = main_module.main()
            self.assertNotEqual(rc, 0)
            self.assertEqual(stdout_buf.getvalue(), "")
            expected = (
                msg.MSG_UNEXPECTED_ERROR_FMT.format(err="completely unknown") + "\n"
            )
            self.assertEqual(stderr_buf.getvalue(), expected)


if __name__ == "__main__":
    unittest.main()
