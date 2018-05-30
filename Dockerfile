FROM jupyter/scipy-notebook
MAINTAINER Alex Lubbock <code@alexlubbock.com>

USER jovyan

RUN conda install -c alubbock pysb
RUN git clone https://github.com/lolab-vu/pysb-tutorials.git /home/jovyan/work/examples
