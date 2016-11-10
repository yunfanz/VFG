#remove unnecessary files
clean:
	rm *.pyc	

#download dependencies
download:
	pip install -r requirements.txt

#Run model with default options
run:
	python main.py --data_dir=./data --mode='train' --use_fourier=False
