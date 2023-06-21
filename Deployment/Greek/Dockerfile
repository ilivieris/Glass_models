FROM python:3.8

# Setup working directory
WORKDIR /code

# Copy files
COPY main.py /code
COPY requirements.txt /code
COPY checkpoint /code/checkpoint
COPY application /code/application

RUN pip install -r requirements.txt


# Run the application
EXPOSE 8000

CMD ["python", "main.py"]