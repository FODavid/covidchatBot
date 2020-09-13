FROM python:3
WORKDIR /app
ADD . /app
RUN pip install --trusted-host pypi.org --trusted-host=files.pythonhosted.org -r requirements.txt
EXPOSE 5000
ENTRYPOINT [ "python3" ]
CMD ["python", "Chappy_Bot_endpoint.py"]