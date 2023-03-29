FROM locustio/locust

# Set working directory
WORKDIR /locust

# Create folder structure
COPY ./runs ./runs
COPY ./locustfile.py ./locustfile.py

# Set environment variables
ENV LOCUST_MODE standalone
ENV LOCUSTFILE_PATH /locust/locustfile.py

# Expose port
# > Locust
EXPOSE 8089