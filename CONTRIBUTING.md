# Contributing to open-veanu/fraudcrawler

First, let us thank you for considering contributing to open-veanu/fraudcrawler! Your involvement is vital to the project's success.

## How Can You Contribute?

- **Reporting Bugs**: If you encounter any issues, please [open an issue](https://github.com/open-veanu/fraudcrawler/issues) with detailed information.
- **Suggesting Enhancements**: Have an idea to improve open-veanu/fraudcrawler? [Submit a feature request](https://github.com/open-veanu/fraudcrawler/issues) outlining your suggestion.
- **Submitting Pull Requests**: Ready to code? Follow the steps below to submit your contributions.

## Getting Started

1. **Clone the Repository**: Clone the open-veanu/fraudcrawler repository to your local machine:
   ```bash
   git clone https://github.com/open-veanu/fraudcrawler.git
   cd fraudcrawler
   ```

2. **Create a Branch**: Start by creating a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/<your-github-name>/<your-feature-name>
   ```
   

3. **Install Dependencies**: Use Poetry to install the project's dependencies:
   ```bash
   poetry install
   ```

4. **Make Changes**: Implement your changes in the codebase.

5. **Run Tests**: Ensure all tests pass before submitting:
   ```bash
   poetry run pytest tests/unittests/test_* -s -v
   ```

6. **Commit Your Changes**: Use a descriptive commit message:
   ```bash
   git add .
   git commit -m "Add feature: your feature description"
   ```

7. **Push Your Branch**: Push your branch to the main repository:
   ```bash
   git push origin feature/<your-github-name>/<your-feature-name>
   ```

8. **Open a Pull Request**: Go to the open-veanu/fraudcrawler repository on GitHub and open a pull request from your branch.

## Code Style

Please adhere to the following coding standards:

- **PEP 8**: Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code.
- **Docstrings**: Use clear and concise docstrings for functions and classes.
- **Type Annotations**: Include type annotations everywhere.
- **Use RUFF for linting**: Use the `ruff format` and `ruff check --fix` commands to ensure proper formatting.