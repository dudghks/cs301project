FROM python:3.12.13

WORKDIR /proj

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy script and data
COPY CS301_Project_Milestone_2.py .
COPY content/ content/

# Create output folder
RUN mkdir -p /proj/output

# Execute script
CMD ["python", "CS301_Project_Milestone_2.py"]
