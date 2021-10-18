FROM ubuntu:20.04

ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND noninteractive

RUN apt update
RUN apt install -y nano curl iputils-ping net-tools telnet gcc musl-dev python3-pip python3-dev libffi-dev libssl-dev build-essential cargo

RUN pip3 install cryptography
RUN pip3 install transformers
RUN pip3 install torchinfo
RUN pip3 install xlsxwriter
RUN pip3 install nltk

RUN python3 -m nltk.downloader -d /usr/local/share/nltk_data all

RUN pip3 install pandas
RUN pip3 install torch torchvision
RUN pip3 install matplotlib
RUN pip3 install scikit-learn

WORKDIR /opt/script
COPY chat.csv /opt/script/chat.csv
COPY intents.json /opt/script/intents.json
COPY start.py /opt/script/start.py

ENTRYPOINT [ "python3", "start.py" ]
