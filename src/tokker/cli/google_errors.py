#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys

from tokker.cli import strings


def _print_google_auth_guidance() -> None:
    """
    Print concise guidance for setting up Google Application Default
    Credentials for Gemini models used via the CLI.
    """
    print(
        "Google auth setup required for Gemini (takes 3 mins):",
        file=sys.stderr,
    )
    print(f"  {strings.GOOGLE_AUTH_GUIDE}", file=sys.stderr)
    print("-----------", file=sys.stderr)
    print("Alternatively, sign in via browser:", file=sys.stderr)
    print(
        "  1. Install this: https://cloud.google.com/sdk/docs/install",
        file=sys.stderr,
    )
    print("  2. Run this command:", file=sys.stderr)
    print("     gcloud auth application-default login", file=sys.stderr)


def _handle_google_error(registry, model: str, original_error: Exception) -> None:
    """
    Handle provider-specific errors for Google Gemini models.

    Behavior:
      - If model's provider is not Google, re-raise the original error.
      - If GOOGLE_APPLICATION_CREDENTIALS is set but invalid, print a
        helpful message and guidance.
      - If gcloud is available, attempt 'gcloud auth application-default
        login' for a browser-based sign-in.
      - Otherwise, print guidance and exit.

    This function prints human-friendly messages to stderr and calls
    sys.exit(1) for terminal guidance flows. If the provider is not
    Google, the original error is re-raised so the caller can handle it.
    """
    try:
        provider_name = registry.get_provider(model).NAME
    except Exception:
        provider_name = None

    # Not a Google provider path: let the caller handle the error.
    if provider_name != "Google":
        raise original_error

    # If user points to ADC via env, validate the file exists.
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if adc_path:
        if not os.path.isfile(adc_path):
            print(
                f"GOOGLE_APPLICATION_CREDENTIALS points to a missing file: "
                f"{adc_path}",
                file=sys.stderr,
            )
            _print_google_auth_guidance()
            sys.exit(1)
        # If file exists but we still failed, just surface the error
        print(str(original_error), file=sys.stderr)
        sys.exit(1)

    # Try gcloud auth flow if available for a quick browser sign-in.
    gcloud_path = shutil.which("gcloud")
    if gcloud_path:
        print(
            "Attempting browser sign-in via gcloud (Application Default "
            "Credentials)...",
            file=sys.stderr,
        )
        try:
            proc = subprocess.run(
                [gcloud_path, "auth", "application-default", "login"],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.stdout:
                print(proc.stdout, file=sys.stderr, end="")
            if proc.stderr:
                print(proc.stderr, file=sys.stderr, end="")
        except Exception as ge:
            print(f"gcloud sign-in attempt failed: {ge}", file=sys.stderr)
        sys.exit(1)

    # Fallback: give guidance and exit.
    _print_google_auth_guidance()
    sys.exit(1)
