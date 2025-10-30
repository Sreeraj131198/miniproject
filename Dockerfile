FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY app.py .
COPY marital.pkl .
COPY education.pkl .
COPY contact.pkl .
COPY poutcome.pkl .
COPY scaling.pkl .
COPY feature_selector.pkl .
COPY RF_model.pkl .
COPY templates/ templates/

# Verify files are copied
RUN ls -la

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]