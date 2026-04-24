# Makefile — io-hotspot-prediction
# Common development commands. Run from project root.
# Usage: make <target>

.PHONY: help env install test test-verbose lint pipeline dashboard clean clean-processed

# ── Default ────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  io-hotspot-prediction"
	@echo "  ─────────────────────────────────────────────────────"
	@echo "  make env            Create conda environment"
	@echo "  make install        Install pip dependencies (fallback)"
	@echo ""
	@echo "  make test           Run all tests"
	@echo "  make test-verbose   Run tests with verbose output"
	@echo "  make lint           Check code style (requires ruff)"
	@echo ""
	@echo "  make grid           Build base 1°×1° grid"
	@echo "  make features       Build full feature matrix"
	@echo "  make train          Train logistic regression model"
	@echo "  make pipeline       Run grid → features → train in sequence"
	@echo ""
	@echo "  make dashboard      Launch Streamlit dashboard"
	@echo "  make notebooks      Launch JupyterLab"
	@echo ""
	@echo "  make clean          Remove Python cache files"
	@echo "  make clean-processed Remove processed data (keeps raw)"
	@echo ""

# ── Environment ────────────────────────────────────────────────────────────
env:
	conda env create -f environment.yml
	@echo ""
	@echo "  Run: conda activate io-hotspot"

install:
	pip install -r requirements.txt

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -q

test-verbose:
	python -m pytest tests/ -v

lint:
	@command -v ruff >/dev/null 2>&1 || (echo "ruff not found. Install: pip install ruff" && exit 1)
	ruff check .

# ── Pipeline ───────────────────────────────────────────────────────────────
grid:
	python -m preprocess.grid

features:
	python -m features.build

train:
	python -m models.train

pipeline: grid features train
	@echo ""
	@echo "  ✅ Pipeline complete."
	@echo "  Run 'make dashboard' to explore results."

# ── Apps ───────────────────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard/app.py

notebooks:
	jupyter lab notebooks/

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	@echo "  ✅ Cache files removed."

clean-processed:
	@echo "  ⚠️  This will delete all processed data. Raw data is preserved."
	@read -p "  Continue? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	find data/processed -name "*.parquet" -delete 2>/dev/null || true
	find models -name "*.pkl" -delete 2>/dev/null || true
	@echo "  ✅ Processed data removed."
