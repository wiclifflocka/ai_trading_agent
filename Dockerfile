FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the trading bot
CMD ["python", "execution/scalping_strategy.py"]

