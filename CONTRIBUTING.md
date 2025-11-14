# Contributing to python-ai

Thank you for your interest in contributing to python-ai! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and constructive in all interactions with the community.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/fovi-llc/python-ai.git
   cd python-ai
   ```

2. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

## Making Changes

### 1. Create a Branch

Create a new branch for your changes:
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

- Follow the existing code style
- Keep changes minimal and focused
- Add tests for new functionality
- Update documentation as needed

### 3. Code Style

We use:
- **Black** for code formatting (line length: 100)
- **Ruff** for linting
- **Type hints** for all public APIs

Format your code:
```bash
black python_ai tests examples
ruff check python_ai tests examples --fix
```

### 4. Run Tests

Make sure all tests pass:
```bash
pytest tests/ -v
```

### 5. Update Documentation

If you're adding new features:
- Update the README.md
- Add examples if applicable
- Update ARCHITECTURE.md if changing structure
- Add docstrings to new functions/classes

## Submitting Changes

### Pull Request Process

1. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Brief description of changes"
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request:**
   - Go to GitHub and create a PR from your branch
   - Fill in the PR template
   - Link any related issues

### PR Guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Explain what changes you made and why
- **Tests**: Include tests for new functionality
- **Documentation**: Update docs as needed
- **Focused**: Keep PRs focused on a single feature/fix

## What to Contribute

### Good First Issues

Look for issues labeled `good first issue` - these are great starting points.

### Areas for Contribution

- **Bug fixes**: Found a bug? Fix it!
- **Documentation**: Improve examples, clarify docs
- **Tests**: Add more test coverage
- **Examples**: Add more usage examples
- **Features**: Propose and implement new features

### Feature Proposals

For significant changes:
1. Open an issue first to discuss the idea
2. Get feedback from maintainers
3. Implement after approval

## IDL Specification

When adding features, ensure they match the Web IDL specification:
- [Prompt API IDL](https://webmachinelearning.github.io/prompt-api/#idl-index)
- [Chrome Documentation](https://developer.chrome.com/docs/ai/prompt-api)

Key principles:
- **Direct mapping**: Features should map directly to the IDL
- **Minimal code**: Avoid unnecessary abstractions
- **Type safety**: Use proper type hints
- **Error handling**: Preserve error types from JavaScript

## Project Structure

```
python-ai/
├── python_ai/             # Main package
│   ├── __init__.py        # Package exports
│   ├── models.py          # Data models
│   └── language_model.py  # Main implementation
├── tests/                 # Unit tests
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Testing

### Writing Tests

- Use pytest
- Test data models thoroughly
- Mock JavaScript interactions when needed
- Aim for high coverage

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_models.py -v

# With coverage
pytest tests/ --cov=python_ai
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def my_function(arg1: str, arg2: int) -> bool:
    """Brief description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When something goes wrong
    """
```

### Updating README

When adding features:
1. Add to the appropriate section
2. Include a code example
3. Keep examples concise
4. Test that examples work

## Release Process

Maintainers handle releases:
1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Tag the release
4. Build and publish to PyPI

## Getting Help

- **Questions**: Open a GitHub Discussion
- **Bugs**: Open a GitHub Issue
- **Features**: Open a GitHub Issue with proposal
- **Chat**: Coming soon!

## Recognition

Contributors will be:
- Listed in the CONTRIBUTORS file
- Mentioned in release notes
- Credited in commit messages

Thank you for contributing! 🎉
