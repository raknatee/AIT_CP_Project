FROM jupyter/datascience-notebook
USER root
WORKDIR /home/jovyan/work
RUN pip install jupyter-contrib-nbextensions==0.5.1
RUN jupyter contrib nbextension install

# COPY ./req.txt .

# when export data > conda list --export > req.txt
# RUN pip install -r req.txt




# CMD tail -f /dev/null
CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root