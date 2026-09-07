PYTHON := $(shell if [ -f .venv/bin/python3 ]; then echo "$$(pwd)/.venv/bin/python3"; else echo "python3"; fi)
	
analyse_train_run:
	cat mainrun/logs/baseline.log

submit: 
	task submit

generate_report:
	./scripts/generate_report.sh

# train commands
train_local:
	cd mainrun && $(PYTHON) train.py
	
train_custom:
	cd mainrun && $(PYTHON) train.py --lr 6e-4 --vocab_size 16000 --block_size 128 --batch_size 64  --weight_decay 0.1 --dropout 0.1


# test commands
test:
	$(PYTHON) -m pytest -s tests/

# -s flag enables printing
test_data:
	$(PYTHON) -m pytest -s tests/test_data.py

test_llama:
	$(PYTHON) -m pytest -s tests/test_llama.py