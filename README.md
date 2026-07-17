# collabsort-agent

An agent for a [collaborative sorting task](https://github.com/bpesquet/gym-collabsort).

## Installation

> [uv](https://docs.astral.sh/uv/) needs to be available on your system.

```bash
git clone https://github.com/bpesquet/collabsort-agent
cd collabsort-agent
uv sync
```

## Usage

This project uses [tyro](https://github.com/brentyi/tyro) to easily obtain configuration values from the CLI.

```bash
# Show all possible options for training
uv run python src/collabsort_agent/train.py --help

# Train the agent, using a configuration specified by {options}
uv run python src/collabsort_agent/train.py {options}

# Visualize training results through TensorBoard at http://localhost:6006
uv run tensorboard --logdir runs/

# Launch a demo of a trained agent identified by {train_dir}
uv run python src/collabsort_agent/demo.py --train_dir runs/{train_dir}
```

## Development notes

This project is built and tested with the following software:

- [ruff](https://docs.astral.sh/ruff/) for code formatting and linting;
- [ty](https://docs.astral.sh/ty/) for type checking;
- [pytest](https://docs.pytest.org) for testing.

### Useful commands

```bash
# Format all Python files
uvx ruff format

# Lint all Python files and fix any fixable errors
uvx ruff check --fix

# Check for type-related mistakes
uvx ty check

# Test the codebase. See pyproject.toml for pytest configuration.
# The optional -s flag prints code output.
# Code coverage reporting is configured in pyproject.toml
uv run pytest [-s]
```

## License

[MIT](LICENSE).

Copyright © 2025-present [Baptiste Pesquet](https://www.bpesquet.fr).
