#!/bin/bash
if ! docker info >/dev/null 2>&1; then
    sudo apt install -y docker docker.io
    sudo usermod -aG docker $USER # necessário apenas se o usuário ainda não utilizar docker
    echo "Docker instalado com sucesso. Por favor, faça logout e login novamente para que as alterações de grupo tenham efeito."
else
    echo "Docker já está instalado"
fi