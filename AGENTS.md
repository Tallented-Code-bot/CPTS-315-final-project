# Agent Instructions for Rock-Paper-Scissors Data Mining Project

This repository contains a Python project for data collection and analysis of Rock-Paper-Scissors games, alongside LaTeX documentation and slides.

## Build and Execution Commands

### LaTeX (Tectonic)
The project uses the Tectonic compiler for LaTeX. The preamble and postamble are split into `_preamble.tex` and `_postamble.tex`.
- **Build all:** `tectonic -X build`
- **Build specific directory:** Use `tectonic -X build` within `slides/` or the root for the main document.

### Python
- **Run data collection:** `python code/data_collector.py`
- **Linting:** Use `ruff check .` (if installed) or follow PEP 8.
- **Testing:** No formal test suite exists yet. Create scripts in `code/` to verify logic. To test a specific function, use `python -c "from code.algorithms import ...; print(...)"`.

## Code Style Guidelines

### Python (General)
- **Style:** Follow PEP 8. Use 4 spaces for indentation.
- **Naming:** 
  - Functions and variables: `snake_case` (e.g., `generate_candidates`).
  - Constants: `UPPER_SNAKE_CASE` (e.g., `ROCK`, `PLAYER`).
  - Note: Some existing code uses `camelCase` (e.g., `computerPlay`, `playSingleGame`). When modifying existing functions, maintain the established naming, but prefer `snake_case` for new code.
- **Typing:** Use type hints from `typing` and `numpy.typing`. 
  - `GameType`: `Annotated[NDArray[np.float64], 3, 2]`
  - `PredictionsType`: `Annotated[NDArray[np.float64], 3]`
- **Imports:** 
  - Standard library first, then third-party (numpy, pandas), then local modules.
  - Group and sort imports alphabetically within categories.
- **Error Handling:** Use `match/case` for branching on discrete options and raise `ValueError` for invalid states.

### LaTeX
- **Structure:** Content resides in `index.tex`. Avoid putting preamble/postamble code directly in `index.tex`; use the dedicated `_preamble.tex` and `_postamble.tex` files.
- **Compilation:** Always use `tectonic -X build` to ensure the project structure is handled correctly.

## Project Structure
- `code/`: Python algorithms and data collection scripts.
- `src/`: Main LaTeX report source.
- `slides/`: LaTeX presentation source.
- `build/`: Output directory for compiled PDFs.
