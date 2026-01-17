.PHONY: train_baseline train_hybrid eval infer

train_baseline:
	bash scripts/train_baseline.sh

train_hybrid:
	bash scripts/train_hybrid.sh

eval:
	bash scripts/eval_latest.sh

infer:
	bash scripts/infer_example.sh
