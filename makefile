PYTHON := $(shell if [ -f .venv/bin/python3 ]; then echo "$$(pwd)/.venv/bin/python3"; else echo "python3"; fi)

train:
	task train


analyse:
	cat mainrun/logs/baseline.log

submit: 
	task submit

generate_report:
	./scripts/generate_report.sh

train-hp-02:
	cd mainrun && $(PYTHON) train.py --lr 6e-4 --vocab_size 16000 --block_size 128 --batch_size 64  --weight_decay 0.1 --dropout 0.1

train-hp-03:
	cd mainrun && $(PYTHON) train.py --lr 8e-4 --vocab_size 8000 --block_size 32 --batch_size 128 --weight_decay 0.1 --dropout 0.0

# -s flag enables printing
test:
	python3 -m pytest -s tests/test.py