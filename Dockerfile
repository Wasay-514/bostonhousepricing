FROM  python:3.14
COPY . /app
WORKDIR /app
RUN pip install -r requirement.txt
EXPOSE $PORT
CMD guincorn --workers=4 --bind 0,0,0,0:$PORT app:app