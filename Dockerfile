FROM python:3.8

# Add user
RUN adduser --quiet --disabled-password appuser && usermod -a -G audio appuser

# copy the code files
WORKDIR /app
COPY src /app
COPY requirements.txt /app/
COPY setup.py /app/
COPY README.md /app/
# create the data folder
RUN mkdir -p  data/download
RUN mkdir -p config
RUN mkdir -p logs

# Build the environement
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# # Code PyTesting
# RUN pip install pytest
# RUN pytest

# Run the software
EXPOSE 8080
# CMD ["ls", "-a"]
CMD ["uvicorn", "webapp:app", "--host", "0.0.0.0", "--port", "8080"]