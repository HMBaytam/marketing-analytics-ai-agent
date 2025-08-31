"""Test runner and test suite management."""

import argparse
import subprocess
import sys
from pathlib import Path


class TestRunner:
    """Comprehensive test runner for Marketing AI Agent."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.test_dir = self.project_root / "tests"
        self.coverage_dir = self.project_root / "htmlcov"

    def run_unit_tests(self, verbose: bool = True, coverage: bool = True) -> int:
        """Run unit tests."""
        args = ["-m", "pytest"]

        # Add markers for unit tests
        args.extend(["-m", "unit or not slow and not integration and not e2e"])

        if verbose:
            args.append("-v")

        if coverage:
            args.extend(
                [
                    "--cov=src/marketing_ai_agent",
                    "--cov-report=html",
                    "--cov-report=term-missing",
                ]
            )

        args.append(str(self.test_dir))

        print(f"Running unit tests: python {' '.join(args)}")
        return subprocess.call([sys.executable] + args)

    def run_integration_tests(self, verbose: bool = True) -> int:
        """Run integration tests."""
        args = ["-m", "pytest", "-m", "integration"]

        if verbose:
            args.append("-v")

        args.append(str(self.test_dir))

        print(f"Running integration tests: python {' '.join(args)}")
        return subprocess.call([sys.executable] + args)

    def run_cli_tests(self, verbose: bool = True) -> int:
        """Run CLI-specific tests."""
        args = ["-m", "pytest", "-m", "cli"]

        if verbose:
            args.append("-v")

        args.append(str(self.test_dir / "test_cli.py"))

        print(f"Running CLI tests: python {' '.join(args)}")
        return subprocess.call([sys.executable] + args)

    def run_performance_tests(self, verbose: bool = True) -> int:
        """Run performance/benchmark tests."""
        args = ["-m", "pytest", "-m", "slow or performance"]

        if verbose:
            args.append("-v")

        # Add benchmark plugin if available
        try:
            import pytest_benchmark

            args.extend(["--benchmark-only", "--benchmark-sort=mean"])
        except ImportError:
            print("Warning: pytest-benchmark not available, running without benchmarks")

        args.append(str(self.test_dir))

        print(f"Running performance tests: python {' '.join(args)}")
        return subprocess.call([sys.executable] + args)

    def run_error_handling_tests(self, verbose: bool = True) -> int:
        """Run error handling specific tests."""
        args = ["-m", "pytest", "-m", "error_handling"]

        if verbose:
            args.append("-v")

        args.append(str(self.test_dir))

        print(f"Running error handling tests: python {' '.join(args)}")
        return subprocess.call([sys.executable] + args)

    def run_all_tests(
        self, verbose: bool = True, coverage: bool = True
    ) -> dict[str, int]:
        """Run all test suites."""
        results = {}

        print("🧪 Running Marketing AI Agent Test Suite")
        print("=" * 50)

        # Unit tests
        print("\n📋 Running Unit Tests...")
        results["unit"] = self.run_unit_tests(verbose=verbose, coverage=coverage)

        # Integration tests
        print("\n🔗 Running Integration Tests...")
        results["integration"] = self.run_integration_tests(verbose=verbose)

        # CLI tests
        print("\n💻 Running CLI Tests...")
        results["cli"] = self.run_cli_tests(verbose=verbose)

        # Error handling tests
        print("\n🚨 Running Error Handling Tests...")
        results["error_handling"] = self.run_error_handling_tests(verbose=verbose)

        # Performance tests (optional)
        print("\n⚡ Running Performance Tests...")
        results["performance"] = self.run_performance_tests(verbose=verbose)

        self.print_test_summary(results)
        return results

    def run_quick_tests(self) -> int:
        """Run quick test subset for development."""
        args = [
            "-m",
            "pytest",
            "-m",
            "not slow and not performance and not integration",
            "-x",  # Stop on first failure
            "--tb=short",
            str(self.test_dir),
        ]

        print(f"Running quick tests: python {' '.join(args)}")
        return subprocess.call([sys.executable] + args)

    def run_specific_test(self, test_path: str, verbose: bool = True) -> int:
        """Run specific test file or test function."""
        args = ["-m", "pytest"]

        if verbose:
            args.append("-v")

        args.append(test_path)

        print(f"Running specific test: python {' '.join(args)}")
        return subprocess.call([sys.executable] + args)

    def generate_coverage_report(self) -> int:
        """Generate detailed coverage report."""
        if not (self.project_root / ".coverage").exists():
            print("No coverage data found. Run tests with coverage first.")
            return 1

        args = ["-m", "coverage", "html", "--directory", str(self.coverage_dir)]
        print(f"Generating coverage report: python {' '.join(args)}")
        result = subprocess.call([sys.executable] + args)

        if result == 0:
            print(f"Coverage report generated: {self.coverage_dir / 'index.html'}")

        return result

    def print_test_summary(self, results: dict[str, int]):
        """Print test execution summary."""
        print("\n" + "=" * 50)
        print("🧪 Test Suite Summary")
        print("=" * 50)

        total_passed = 0
        total_suites = len(results)

        for suite_name, exit_code in results.items():
            status = "✅ PASSED" if exit_code == 0 else "❌ FAILED"
            print(f"{suite_name.title()} Tests: {status}")
            if exit_code == 0:
                total_passed += 1

        print("-" * 50)
        print(f"Overall: {total_passed}/{total_suites} test suites passed")

        if total_passed == total_suites:
            print("🎉 All tests passed!")
        else:
            print("⚠️  Some tests failed. Check logs above for details.")

        # Coverage info
        if (self.project_root / ".coverage").exists():
            print(
                f"\n📊 Coverage report available at: {self.coverage_dir / 'index.html'}"
            )

    def validate_test_environment(self) -> bool:
        """Validate test environment setup."""
        print("🔍 Validating test environment...")

        issues = []

        # Check Python version

        # Check required packages
        required_packages = [
            "pytest",
            "pytest-cov",
            "pytest-mock",
            "factory-boy",
            "faker",
            "responses",
            "typer",
            "pandas",
            "numpy",
        ]

        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                issues.append(f"Missing package: {package}")

        # Check test directory structure
        required_files = [
            self.test_dir / "conftest.py",
            self.test_dir / "factories.py",
            self.test_dir / "test_core.py",
            self.test_dir / "test_cli.py",
        ]

        for file_path in required_files:
            if not file_path.exists():
                issues.append(f"Missing test file: {file_path}")

        if issues:
            print("❌ Test environment validation failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Test environment validation passed!")
            return True


def main():
    """Main test runner CLI."""
    parser = argparse.ArgumentParser(description="Marketing AI Agent Test Runner")
    parser.add_argument(
        "suite",
        nargs="?",
        choices=["unit", "integration", "cli", "performance", "error", "all", "quick"],
        default="all",
        help="Test suite to run",
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Skip coverage reporting"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--validate", action="store_true", help="Validate test environment only"
    )
    parser.add_argument(
        "--specific", type=str, help="Run specific test file or function"
    )
    parser.add_argument(
        "--coverage-report",
        action="store_true",
        help="Generate HTML coverage report only",
    )

    args = parser.parse_args()

    # Initialize runner
    runner = TestRunner()

    # Validate environment if requested
    if args.validate:
        return 0 if runner.validate_test_environment() else 1

    # Generate coverage report only
    if args.coverage_report:
        return runner.generate_coverage_report()

    # Run specific test
    if args.specific:
        return runner.run_specific_test(args.specific, verbose=not args.quiet)

    # Validate environment before running tests
    if not runner.validate_test_environment():
        print(
            "\n❌ Environment validation failed. Please fix issues before running tests."
        )
        return 1

    # Run requested test suite
    verbose = not args.quiet
    coverage = not args.no_coverage

    if args.suite == "unit":
        return runner.run_unit_tests(verbose=verbose, coverage=coverage)
    elif args.suite == "integration":
        return runner.run_integration_tests(verbose=verbose)
    elif args.suite == "cli":
        return runner.run_cli_tests(verbose=verbose)
    elif args.suite == "performance":
        return runner.run_performance_tests(verbose=verbose)
    elif args.suite == "error":
        return runner.run_error_handling_tests(verbose=verbose)
    elif args.suite == "quick":
        return runner.run_quick_tests()
    elif args.suite == "all":
        results = runner.run_all_tests(verbose=verbose, coverage=coverage)
        # Return non-zero if any suite failed
        return 0 if all(code == 0 for code in results.values()) else 1

    return 0


if __name__ == "__main__":
    exit(main())
