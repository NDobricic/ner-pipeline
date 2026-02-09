"""Integration tests for the CLI interface (el_pipeline/cli.py)."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict

import pytest


@pytest.mark.integration
class TestCLIExecution:
    """Tests for basic CLI execution."""

    def test_cli_runs_with_valid_args(self, temp_config_file: str, temp_text_file: str):
        """CLI executes successfully with valid arguments."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "el_pipeline.cli",
                "--config",
                temp_config_file,
                "--input",
                temp_text_file,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"

    def test_cli_processes_single_file(self, temp_config_file: str, temp_text_file: str):
        """CLI processes a single input file."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "el_pipeline.cli",
                "--config",
                temp_config_file,
                "--input",
                temp_text_file,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode == 0

    def test_cli_processes_multiple_files(
        self, temp_config_file: str, temp_text_file: str, sample_text: str
    ):
        """CLI processes multiple input files."""
        # Create a second temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(sample_text)
            second_file = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "el_pipeline.cli",
                    "--config",
                    temp_config_file,
                    "--input",
                    temp_text_file,
                    second_file,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
        finally:
            os.unlink(second_file)

    def test_cli_writes_output_file(self, temp_config_file: str, temp_text_file: str):
        """CLI writes valid JSONL output when --output is specified."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "el_pipeline.cli",
                    "--config",
                    temp_config_file,
                    "--input",
                    temp_text_file,
                    "--output",
                    output_path,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert os.path.exists(output_path)

            # Verify it's valid JSONL
            with open(output_path) as f:
                for line in f:
                    json.loads(line)  # Should not raise
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def test_cli_output_contains_entities(self, temp_config_file: str, temp_text_file: str):
        """CLI output contains entities array."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            output_path = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "el_pipeline.cli",
                    "--config",
                    temp_config_file,
                    "--input",
                    temp_text_file,
                    "--output",
                    output_path,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"

            with open(output_path) as f:
                for line in f:
                    data = json.loads(line)
                    assert "entities" in data
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)


@pytest.mark.integration
class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_cli_missing_config_arg(self, temp_text_file: str):
        """CLI exits with error when --config is missing."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "el_pipeline.cli",
                "--input",
                temp_text_file,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "config" in result.stderr.lower()

    def test_cli_missing_input_arg(self, temp_config_file: str):
        """CLI exits with error when --input is missing."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "el_pipeline.cli",
                "--config",
                temp_config_file,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "input" in result.stderr.lower()

    def test_cli_invalid_config_path(self, temp_text_file: str):
        """CLI fails with non-existent config file."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "el_pipeline.cli",
                "--config",
                "/nonexistent/path/config.json",
                "--input",
                temp_text_file,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode != 0

    def test_cli_invalid_config_json(self, temp_text_file: str):
        """CLI fails with malformed JSON config."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{ invalid json }")
            bad_config = f.name

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "el_pipeline.cli",
                    "--config",
                    bad_config,
                    "--input",
                    temp_text_file,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )
            assert result.returncode != 0
        finally:
            os.unlink(bad_config)

    def test_cli_invalid_input_path(self, temp_config_file: str):
        """CLI fails with non-existent input file."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "el_pipeline.cli",
                "--config",
                temp_config_file,
                "--input",
                "/nonexistent/path/input.txt",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )
        assert result.returncode != 0
