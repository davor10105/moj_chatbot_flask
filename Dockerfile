FROM python:3.10-bookworm

WORKDIR /moj_chatbot

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./moj_chatbot .

CMD ["python3", "run.py"]