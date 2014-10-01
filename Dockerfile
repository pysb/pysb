FROM ipython/scipyserver
MAINTAINER Alex Lubbock <alex.lubbock@ed.ac.uk>

RUN apt-get install -y wget unzip

RUN wget http://bionetgen.googlecode.com/files/BioNetGen-2.2.5-stable.zip -O /BioNetGen-2.2.5-stable.zip
RUN unzip /BioNetGen-2.2.5-stable.zip -d /
RUN ln -s /BioNetGen-2.2.5-stable /usr/local/share/BioNetGen

RUN git clone https://github.com/pysb/pysb.git /pysb
RUN cd /pysb && python setup.py install

RUN sed -i.bak 's/ipython notebook/ipython2 notebook/g' /notebook.sh
