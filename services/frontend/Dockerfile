FROM python:3.10.3-slim-buster

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install -r requirements.txt

COPY ./app /code

EXPOSE 8501

ENTRYPOINT [ "streamlit", "run" ]

CMD [ "main.py" ]