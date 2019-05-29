FROM ubuntu:18.04

RUN apt-get update &&   apt-get install -y  gcc libhdf5-serial-dev libblas-dev liblapack-dev git python-pip python-dev python3-pip python3-dev libc-dev build-essential libpcre3-dev python3 python3-venv swig python3.6-dev libpython3-dev libpython-dev python-numpy python3-numpy &&  python -c "import sysconfig; print(sysconfig.get_path('include'))"  && git clone https://github.com/facebookresearch/faiss.git && cd faiss && ./configure --without-cuda --with-python=python3 &&  make  &&  make install &&  cd python && make && make install


COPY . /app
WORKDIR /app

#ENV VIRTUAL_ENV=/opt/venv
#CMD pip install --upgrade pip
#CMD pip3 install virtualenv
#RUN python3 -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH" 


#CMD mkdir virtual-env && cd virtual-env
#CMD python3 -m venv env 
#CMD source env/bin/activate

RUN pip3 install -r requirements.txt

EXPOSE 9210

CMD ["python3" , "server.py" ]