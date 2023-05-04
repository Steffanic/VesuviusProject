train: TrainUNet.py
	python TrainUNet.py

download:
	kaggle competitions download -c vesuvius-challenge-ink-detection
	unzip vesuvius-challenge-ink-detection.zip
