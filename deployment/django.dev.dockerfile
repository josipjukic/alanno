FROM nvidia/cuda:11.3.0-base-ubuntu20.04

ENV TZ=Europe/Zagreb
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt update && apt install -y python3-dev libpq-dev
RUN apt-get update && apt-get install -y python3.8 python3-pip
RUN pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm

WORKDIR /app

CMD bash -c "python3 manage.py migrate && python3 manage.py initadmin && python3 manage.py initgroups && python3 manage.py initoptions && python3 manage.py collectstatic --noinput -v 0 && python3 manage.py runserver 0.0.0.0:8000" && bash
