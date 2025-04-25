import csv
import random
import datetime

def generate_dns_logs(csv_file, num_entries=1000):
    """
    Generate synthetic DNS logs and save to a CSV file.
    """
    fields = ["timestamp", "query_name", "transaction_id", "ttl", "response_ip"]
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        
        for _ in range(num_entries):
            timestamp = datetime.datetime.now() + datetime.timedelta(seconds=random.randint(0, 3600))
            query_name = f"example{random.randint(1, 100)}.com"
            transaction_id = random.randint(0, 65535)
            ttl = random.choice([60, 120, 300, 0])  # Include 0 for anomalies
            response_ip = f"192.168.1.{random.randint(2, 254)}"
            
            writer.writerow([timestamp, query_name, transaction_id, ttl, response_ip])
    
    print(f"Generated synthetic DNS logs in {csv_file}")

# Generate synthetic DNS logs
generate_dns_logs("synthetic_dns_logs.csv", num_entries=500)
