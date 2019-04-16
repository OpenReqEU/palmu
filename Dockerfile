FROM frolvlad/alpine-glibc:alpine-3.9

ENV CONDA_DIR="./conda"
ENV PATH="$CONDA_DIR/bin:$PATH"

# Install conda
RUN CONDA_VERSION="4.5.12" && \
    CONDA_MD5_CHECKSUM="866ae9dff53ad0874e1d1a60b1ad1ef8" && \
    \
    apk add --no-cache --virtual=.build-dependencies wget ca-certificates bash && \
    \
    mkdir -p "$CONDA_DIR" && \
    wget "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh" -O miniconda.sh && \
    echo "$CONDA_MD5_CHECKSUM  miniconda.sh" | md5sum -c && \
    bash miniconda.sh -f -b -p "$CONDA_DIR" && \
    echo "export PATH=$CONDA_DIR/bin:\$PATH" > /etc/profile.d/conda.sh && \
    rm miniconda.sh && \
    \
    conda update --all --yes && \
    conda config --set auto_update_conda False && \
    rm -r "$CONDA_DIR/pkgs/" && \
    \
    apk del --purge .build-dependencies && \
    \
    mkdir -p "$CONDA_DIR/locks" && \
    chmod 777 "$CONDA_DIR/locks"

ADD fast_search.py /

ADD prepare_data.py /

ADD server.py /

ADD data /data

RUN apk add --no-cache bash  

RUN wget http://nlp.stanford.edu/data/glove.6B.zip && unzip -j glove.6B.zip glove.6B.200d.txt -d data && rm glove.6B.zip

RUN conda update conda && conda install python && conda install faiss-cpu -c pytorch  && conda install -c anaconda pandas && conda install flask && conda install cherrypy && python prepare_data.py

EXPOSE 9210

CMD python3 server.py
