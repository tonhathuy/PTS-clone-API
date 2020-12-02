FROM tensorflow/tensorflow:1.14.0-py3

RUN apt update  

COPY . /FLASK_API

RUN cd FLASK_API && \
    pip3 install -r requirements.txt

WORKDIR /FLASK_API

#CMD python3 app.py
# CMD ["/bin/bash", "entrypoint.sh"]