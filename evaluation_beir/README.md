## Evaluation UniHGKR-base on BEIR

Our variant model adapted for evaluation on BEIR can be found at: https://huggingface.co/ZhishanQ/UniHGKR-base-beir

### Installation

```
pip install beir
pip install -U sentence-transformers
```

### Evaluation 12 dataset using our code
```
python benchmark_on_beir.py
```

The dataset will be downloaded to ./datasets, and the metrics result will be output to the console and the JSON file of metrics result under ./result_path.