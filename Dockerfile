FROM python:3

RUN apt-get update; apt-get install git

## TODO: Add user with least privelages

WORKDIR /model

COPY . /model


RUN pip install -r requirements.txt

ENTRYPOINT ["./entrypoint.sh"]