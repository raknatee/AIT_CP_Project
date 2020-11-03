FROM jupyter/datascience-notebook
USER root
WORKDIR /home/jovyan/work
RUN pip install jupyter-contrib-nbextensions==0.5.1
RUN jupyter contrib nbextension install

# COPY ./req.txt .
RUN pip install 'pillow==7.2.0'
RUN pip install 'torchvision==0.2.2'
RUN pip install 'torch==1.6.0'

RUN pip install "opencv-python==4.4.0.44"
RUN pip install "sklearn"
RUN pip install "pandas"
RUN pip install "seaborn"
RUN pip install "matplotlib"



# CMD tail -f /dev/null
CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root