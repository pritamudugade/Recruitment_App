import csv
from faker import Faker
import random

# Initialize the Faker library
fake = Faker()

# Define lists of possible values for fields
designations = ["Software Engineer", "Data Scientist", "Product Manager", "HR Manager", "Sales Executive"]
domains = ["IT", "Healthcare", "Finance", "Engineering", "Marketing"]
expertise = ["Python", "Machine Learning", "Data Analysis", "Project Management", "Sales"]
specializations = ["Web Development", "Database Administration", "AI/ML", "Network Security", "Digital Marketing"]
tools = ["Python", "JIRA", "Tableau", "Excel", "SQL", "Power BI", "AWS", "GitHub", "Adobe Photoshop", "Salesforce"]
education_levels = ["High School", "Bachelor's Degree", "Master's Degree", "Ph.D."]
languages = ["English", "Spanish", "French", "German", "Chinese"]
certifications = ["PMP", "AWS Certified", "Google Analytics", "CISSP", "CCNA"]

# Create an empty list to store candidate data
candidates = []

# Generate 500 candidate records
for _ in range(500):
    candidate = {
        "Name": fake.name(),
        "Age": random.randint(22, 65),
        "Designation": random.choice(designations),
        "Domain": random.choice(domains),
        "Experience (years)": random.randint(0, 20),
        "Expertise": random.sample(expertise, random.randint(1, len(expertise))),
        "Specialization": random.choice(specializations),
        "Tools Used": random.sample(tools, random.randint(1, len(tools))),
        "Education": random.choice(education_levels),
        "Languages": random.sample(languages, random.randint(1, len(languages))),
        "Certifications": random.sample(certifications, random.randint(0, len(certifications))),
        "Email": fake.email(),
        "Phone Number": fake.phone_number(),
        "Work History": [
            {
                "Employer": fake.company(),
                "Job Title": random.choice(designations),
                "Location": fake.city(),
                "Start Date": fake.date_between(start_date='-10y', end_date='today').strftime('%Y-%m-%d'),
                "End Date": fake.date_between(start_date='-2y', end_date='today').strftime('%Y-%m-%d'),
                "Projects Handled": [
                    {
                        "Project Name": fake.catch_phrase(),
                        "Description": fake.paragraph(),
                    }
                    for _ in range(random.randint(1, 5))
                ],
            }
            for _ in range(random.randint(1, 3))
        ],
        "Location": fake.city(),
    }
    candidates.append(candidate)

# Specify the CSV file path
csv_file = "candidates_dataset.csv"

# Write the data to the CSV file
with open(csv_file, mode="w", newline="") as file:
    fieldnames = [
        "Name", "Age", "Designation", "Domain", "Experience (years)", "Expertise", "Specialization", "Tools Used",
        "Education", "Languages", "Certifications", "Email", "Phone Number", "Work History", "Location"
    ]  # Include additional fields
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Write the candidate records
    for candidate in candidates:
        writer.writerow(candidate)

print(f"Data saved to {csv_file}")
