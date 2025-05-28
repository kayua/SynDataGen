pipenv run python3 main.py $1\
	-i Datasets/binaries/kronodroid_emulador-balanced.csv \
	-c RandomForest \
	--model_type autoencoder  \
	--number_samples_per_class 0:10,1:10 \
	--autoencoder_number_epochs 5 \
	--number_k_folds 2 
  	
  	#'adversarial'  "wasserstein" "diffusion" "variational" 'autoencoder' 
