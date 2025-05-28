#!/bin/sh
if command -v pip >/dev/null 2>&1; then
	if command -v pipenv >/dev/null 2>&1; then
      		pipenv install -r requirements.txt
	else
	      pip install pipenv 
	      pipenv install -r requirements.txt

	fi
else
	echo "pip não instalado, execute o comando: sudo apt install python3-pip"	
fi
