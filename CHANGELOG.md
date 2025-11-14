# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-13

### Added
- Initial release of python-genai
- Full implementation of Chrome Prompt API wrapper
- AnyWidget integration for Jupyter notebooks
- Traitlets-based bidirectional communication
- Core features:
  - Session management (create, clone, destroy)
  - Prompting (synchronous and streaming)
  - System prompts and multi-turn conversations
  - Token usage tracking and measurement
  - Structured output with JSON schema
  - Multimodal input support (text, images, audio)
  - Tool use capabilities
- Data models matching Web IDL specification:
  - `LanguageModel` main class
  - `LanguageModelParams` for model parameters
  - `LanguageModelMessage` and related message types
  - Configuration options classes
  - Enums for availability, roles, and types
- Comprehensive documentation:
  - README with API reference
  - Quick start guide
  - Architecture documentation
  - Contributing guidelines
- Examples:
  - Python script with usage examples
  - Jupyter notebook demo
- Testing:
  - 11 unit tests covering data models
  - All tests passing
- CI/CD:
  - GitHub Actions workflow for testing
  - Support for Python 3.10-3.12

### Security
- No vulnerabilities in dependencies
- CodeQL analysis passed
- Cross-origin restrictions enforced

[0.1.0]: https://github.com/fovi-llc/python-genai/releases/tag/v0.1.0
