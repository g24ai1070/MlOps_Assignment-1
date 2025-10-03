# Boston Housing — MLOps Assignment 1 (dtree + kernelridge)

Implements the required workflow:
- Generic pipeline in `misc.py`
- Branch `dtree`: `requirements.txt` and `train.py` (DecisionTreeRegressor) using functions from `misc.py`
- Merge `dtree` -> `main`
- Branch `kernelridge`: `train2.py` (KernelRidge) reusing the same utilities
- GitHub Actions: any push to `kernelridge` runs both scripts and prints test MSE

Dataset is loaded manually from http://lib.stat.cmu.edu/datasets/boston and cached to `boston_housing_cached.csv`.

## Setup
```bash
conda create -n mlops-a1 python=3.10 -y
conda activate mlops-a1
pip install -r requirements.txt
```

## Git workflow (commands)
```bash
# init with README only
git init
git add README.md
git commit -m "init: README"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main

# dtree branch
git checkout -b dtree
git add misc.py train.py requirements.txt .github/workflows/ci.yml
git commit -m "feat(dtree): add misc, train.py, requirements"
git push -u origin dtree

# merge to main
git checkout main
git merge dtree
git push origin main

# kernelridge branch
git checkout -b kernelridge
git add train2.py
git commit -m "feat(kernelridge): add train2"
git push -u origin kernelridge
```

## Run locally
```bash
python train.py
python train2.py
```

## Screenshots for report
1) Branches dropdown showing `main`, `dtree`, `kernelridge`  
2) Actions logs on `kernelridge` run with MSE lines visible  
3) Commit graph showing merges
