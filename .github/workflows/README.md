# GitHub Workflows Documentation

This directory contains GitHub Actions workflows for the Marketing AI Agent project, providing comprehensive CI/CD, security scanning, and automated dependency management.

## 🚀 Workflows Overview

### 1. CI Pipeline (`ci.yml`)
**Triggers:** Push to main/develop, Pull Requests
**Purpose:** Comprehensive quality assurance and testing

**Jobs:**
- **Code Quality** - Black formatting, Ruff linting, MyPy type checking, Bandit security
- **Tests** - Matrix testing across Python 3.11, 3.12, 3.13 on Linux, Windows, macOS
- **Build** - Package building and verification
- **Documentation** - MkDocs documentation building
- **Dependency Check** - Basic security vulnerability scanning
- **CLI Test** - Command-line interface functionality testing

**Key Features:**
- ✅ Parallel execution for speed
- ✅ Test coverage reporting with 80% minimum
- ✅ Cross-platform compatibility testing
- ✅ Artifact uploads for debugging
- ✅ Codecov integration (requires token)

### 2. Security Scan (`security.yml`)
**Triggers:** Push, Pull Requests, Daily schedule (2 AM UTC), Manual dispatch
**Purpose:** Comprehensive security analysis

**Jobs:**
- **Dependency Security** - Safety, pip-audit vulnerability scanning
- **Code Security** - Bandit SAST, Semgrep security analysis
- **Secret Scanning** - TruffleHog secret detection
- **License Check** - License compliance verification
- **CodeQL** - GitHub's semantic code analysis
- **Security Summary** - Consolidated security reporting

**Key Features:**
- 🔒 Multi-layered security scanning
- 🔒 Automated vulnerability detection
- 🔒 License compliance checking
- 🔒 Secret scanning with full history
- 🔒 PR commenting with results

### 3. Dependency Updates (`dependencies.yml`)
**Triggers:** Weekly schedule (Monday 9 AM UTC), Manual dispatch
**Purpose:** Automated dependency maintenance

**Jobs:**
- **Security Updates** - Priority updates for vulnerable dependencies
- **Regular Updates** - Patch and minor version updates
- **Major Updates** - Careful major version updates with review

**Key Features:**
- 📦 Automated security patches
- 📦 Configurable update types (patch/minor/major)
- 📦 Test validation before PR creation
- 📦 Comprehensive update summaries
- 📦 Draft PRs for major updates

### 4. Release (`release.yml`)
**Triggers:** Version tags (v*.*.*), Manual dispatch
**Purpose:** Automated release management

**Jobs:**
- **Version Validation** - Semantic version format validation
- **Test Suite** - Full test suite execution
- **Security Scan** - Pre-release security validation
- **Build** - Distribution package creation
- **Changelog** - Automated release notes generation
- **Release Creation** - GitHub release with assets
- **Notifications** - Success/failure notifications

**Key Features:**
- 🚀 Automated release creation
- 🚀 Comprehensive pre-release testing
- 🚀 Automatic changelog generation
- 🚀 Security validation before release
- 🚀 Asset uploading (wheel + tarball)

## 🔧 Setup Requirements

### Required Secrets
Configure these in your repository settings:

#### Optional (Recommended)
- `CODECOV_TOKEN` - For test coverage reporting
- `PYPI_API_TOKEN` - For automated PyPI publishing (commented out by default)

### Branch Protection
Recommended branch protection rules for `main`:

```yaml
required_status_checks:
  strict: true
  contexts:
    - "Code Quality Checks"
    - "Tests (ubuntu-latest, 3.11)"
    - "Build Package"
    - "Dependency Security Scan"
enforce_admins: false
required_pull_request_reviews:
  required_approving_review_count: 1
  dismiss_stale_reviews: true
restrictions: null
```

## 📋 Workflow Triggers Summary

| Workflow | Push | PR | Schedule | Manual |
|----------|------|----|---------| -------|
| CI | ✅ main/develop | ✅ | ❌ | ✅ |
| Security | ✅ main/develop | ✅ | ✅ Daily | ✅ |
| Dependencies | ❌ | ❌ | ✅ Weekly | ✅ |
| Release | ❌ | ❌ | ❌ | ✅ Tags only |

## 🎯 Quality Gates

All PRs must pass these quality gates:

### Code Quality
- ✅ Black code formatting
- ✅ Ruff linting (no errors)
- ✅ MyPy type checking
- ✅ Bandit security linting

### Testing
- ✅ All tests passing (unit + integration)
- ✅ Test coverage ≥ 80%
- ✅ Cross-platform compatibility (Linux, Windows, macOS)
- ✅ Multiple Python versions (3.11, 3.12, 3.13)

### Security
- ✅ No known vulnerabilities in dependencies
- ✅ No hardcoded secrets detected
- ✅ License compliance verified
- ✅ Security analysis passed

### Build
- ✅ Package builds successfully
- ✅ CLI functionality verified
- ✅ Documentation builds without errors

## 🚨 Failure Handling

### Automatic Actions
- **Security Issues:** Auto-create high-priority PRs for vulnerable dependencies
- **Test Failures:** Detailed error reporting with artifacts
- **Build Failures:** Artifact uploads for debugging

### Manual Interventions Required
- **Major Version Updates:** Manual review required for breaking changes
- **Security Vulnerabilities:** Immediate attention for critical issues
- **Release Failures:** Manual investigation and retry

## 📊 Monitoring and Reporting

### Artifacts Generated
- Test results (JUnit XML)
- Coverage reports (HTML/XML)
- Security scan reports (JSON)
- Build artifacts (wheel/tarball)
- Documentation builds

### Notifications
- **PR Comments:** Security scan summaries
- **Issues:** Release success/failure notifications
- **Labels:** Automated labeling for categorization

## 🔄 Maintenance

### Weekly Tasks
- Review dependency update PRs
- Check security scan results
- Monitor workflow performance

### Monthly Tasks
- Review and update workflow versions
- Audit security findings
- Update documentation

### As Needed
- Configure additional secrets
- Adjust quality gates
- Update Python version matrix

## 🐛 Troubleshooting

### Common Issues

**Workflow fails with permission errors:**
```yaml
permissions:
  contents: write
  issues: write
  pull-requests: write
```

**Tests fail in CI but pass locally:**
- Check environment differences
- Verify all dependencies are locked
- Review test isolation

**Security scans fail:**
- Review security reports in artifacts
- Update vulnerable dependencies
- Check for false positives

**Release workflow fails:**
- Verify version format (semver)
- Check all tests are passing
- Ensure no security vulnerabilities

### Debug Tips
1. Check workflow logs for detailed error messages
2. Download artifacts for local investigation
3. Use `workflow_dispatch` for manual testing
4. Review individual job outputs

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Poetry Documentation](https://python-poetry.org/docs/)
- [Security Best Practices](https://docs.github.com/en/code-security)
- [Semantic Versioning](https://semver.org/)