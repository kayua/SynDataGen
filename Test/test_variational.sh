python3 main.py 
	-i Datasets/binaries/kronodroid_emulador-balanced.csv \
	-c RandomForest \
	--model_type variational  \
	--number_samples_per_class 0:1000,1:100 \
	--number_k_folds 2 \
	--variational_autoencoder_number_epochs 200
