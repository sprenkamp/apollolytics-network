# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run a Python script: `python src/path/to/file.py`
- Run a specific test: `python -m pytest tests/api/multiple_connections.py`
- Interactive notebook: `jupyter notebook src/path/to/notebook.ipynb`

## Code Style Guidelines
- Imports: Standard library first, third-party next, local modules last
- Use snake_case for variables and functions, PascalCase for classes
- Include type hints for function parameters and return values
- Error handling: Use try/except blocks with specific exceptions
- Logging: Use the logging module with appropriate levels
- Database connections: Always use connection pools and close connections
- Batch processing: Use asyncio for parallel operations
- Environment variables: Load from .env files using dotenv
- Documentation: Add docstrings to functions describing purpose, parameters, and return values
- Prefer async/await pattern for database and I/O operations