# Use an official Python runtime as a parent image
FROM python:3.7

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY app/ /app

# Install any needed packages specified in requirements.txt
RUN pip install streamlit
RUN pip install pandas

# Make port 80 available to the world outside this container
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]

# Run app.py when the container launches
CMD ["app.py"]



