FROM python:3.7-slim

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/
RUN pip install -r requirements.txt
ADD . /app/
EXPOSE 5000
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app