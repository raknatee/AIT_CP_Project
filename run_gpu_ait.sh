# docker-compose build
# dockr run -it --gpus all -p 8888:8888 -v "/mnt/c/Users/rakna/Desktop/AIT_working/AIT_ML_Lab/src/lecture/009_FCC_MNIST/src:/home/jovyan/work" 009_fcc_mnist_app:latest

docker build -f Dockerfile.gpu -t my_pytorch . && docker run -it --gpus all -p 8888:8888 -e CHOWN_HOME=yes -e NB_UID=$(id -u) -e CHOWN_EXTRA_OPTS='-R' --user $(id -u) --group-add $(id -g) -v "$(pwd):/tf/notebooks"   my_pytorch
