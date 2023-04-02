FROM continuumio/anaconda

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME

COPY app.py optimize.py spec-file.txt shops.json ./
COPY static ./static
RUN conda create --name myenv --file spec-file.txt
RUN conda init bash
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
RUN python -c "import flask"

ENTRYPOINT ["conda", "run", "-n", "myenv", "gunicorn", "--bind", ":$PORT", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]