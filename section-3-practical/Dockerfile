FROM tiangolo/uvicorn-gunicorn:python3.10
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y tree
RUN apt update && \
    apt install -y htop libgl1-mesa-glx libglib2.0-0
WORKDIR /app
COPY requirements.txt .
RUN python -m venv venv
RUN /bin/bash -c "source venv/bin/activate"
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]