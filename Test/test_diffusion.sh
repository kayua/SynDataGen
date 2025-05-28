pipenv run python main.py $1\
	-i Datasets/binaries/kronodroid_emulador-balanced.csv \
	-c RandomForest \
	--model_type diffusion  \
	--number_samples_per_class 0:10,1:10 \
	--number_k_folds 2 \
	--diffusion_unet_epochs 5 \
	--diffusion_autoencoder_epochs 5 \
	
  	#'adversarial'  "wasserstein" "diffusion" "variational" 'autoencoder' 
