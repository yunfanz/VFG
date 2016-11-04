#remove unnecessary files. Requires admin privileges
clean:
	rm -f *.pyc	
	rm -rf logs/
	rm -rf out/
	rm -rf samples/

#download dependencies
download:
	pip install -r requirements.txt

#download data
download-data:
	python download.py VSTK

#train model with fft augmentation (requires kernel w/ GPU support)
train:
	python main.py --data_dir=./data/VCTK-Corpus/ --mode='train' --use_fourier=True

#train model without fft augmentation (FFT kernel ops only supported on GPU)
train-cpu:
	python main.py --data_dir=./data/VCTK-Corpus/ --use_fourier=False

#generate samples using trained model
generate:
	python main.py --mode='generate'


