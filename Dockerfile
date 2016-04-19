FROM jupyter/scipy-notebook
MAINTAINER Alex Lubbock <code@alexlubbock.com>

USER root

RUN ln -snf /bin/bash /bin/sh

RUN apt-get install -y wget

RUN wget "http://www.csb.pitt.edu/Faculty/Faeder/?smd_process_download=1&download_id=142" -O /BioNetGen-2.2.6-stable.tar.gz
RUN mkdir /BioNetGen && tar xzf /BioNetGen-2.2.6-stable.tar.gz -C /BioNetGen
RUN ln -s /BioNetGen/BioNetGen-2.2.6-stable /usr/local/share/BioNetGen

USER jovyan

RUN source activate python2 && pip install git+git://github.com/pysb/pysb.git
RUN git clone https://github.com/lolab-vu/pysb-tutorials.git /home/jovyan/work/examples
