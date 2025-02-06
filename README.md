# PhenotypicLandscapeCryptococcus

Setting up the environment:
```
python3.10 -m venv .
source bin/activate
pip install -r requirements.txt
```

Available source files
- ```cryptococcus.py```: loading input data and producing intermediate outputs (only to regenerate intermediate results; beware some minor statistical oscillation can occur)
- ```across_features.py```: computing metrics depending on input features (ordered with feature importance) and produces the resulting figures for visualization.
- ```feature_importances.py```: converts intermediate results into excel format


