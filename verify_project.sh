#!/bin/bash
# Quick verification script for the project

echo "=========================================="
echo "Project Verification Script"
echo "=========================================="
echo ""

# Check Python
echo "1. Checking Python version..."
python --version
echo ""

# Check directory structure
echo "2. Checking directory structure..."
for dir in src tests configs scripts models checkpoints results; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/"
    else
        echo "  ✗ $dir/ MISSING"
    fi
done
echo ""

# Check key files
echo "3. Checking key files..."
for file in README.md LICENSE requirements.txt pyproject.toml .gitignore; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file MISSING"
    fi
done
echo ""

# Check scripts
echo "4. Checking scripts..."
for script in scripts/train.py scripts/evaluate.py scripts/predict.py; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo "  ✓ $script (executable)"
    elif [ -f "$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ✗ $script MISSING"
    fi
done
echo ""

# Check configs
echo "5. Checking configurations..."
for config in configs/default.yaml configs/ablation.yaml; do
    if [ -f "$config" ]; then
        echo "  ✓ $config"
    else
        echo "  ✗ $config MISSING"
    fi
done
echo ""

# Test imports
echo "6. Testing Python imports..."
python -c "import sys; sys.path.insert(0, 'src'); from phoneme_aware_curriculum_diarization_with_adaptive_merging import DualEncoderDiarizationModel; print('  ✓ Module imports successfully')" 2>&1
echo ""

# Test config loading
echo "7. Testing YAML config loading..."
python -c "import yaml; config = yaml.safe_load(open('configs/default.yaml')); print(f'  ✓ Config loaded with {len(config)} sections')" 2>&1
echo ""

# Count lines
echo "8. Project statistics..."
echo "  Python files: $(find src scripts tests -name '*.py' | wc -l)"
echo "  Total LoC: $(find src scripts tests -name '*.py' -exec cat {} \; | wc -l)"
echo "  Test files: $(find tests -name 'test_*.py' | wc -l)"
echo ""

echo "=========================================="
echo "Verification complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Install dependencies: pip install -r requirements.txt"
echo "  2. Run tests: PYTHONPATH=src:. pytest tests/ -v"
echo "  3. Train model: python scripts/train.py --debug"
echo "  4. Evaluate: python scripts/evaluate.py"
echo ""
