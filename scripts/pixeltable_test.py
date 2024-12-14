import pixeltable as pxt
import os

# Initialize pixeltable
pxt.init()

# Print connection info
print("Pixeltable data directory:", os.path.expanduser("~/.pixeltable/pgdata"))

# Try to find the port number from postmaster.pid
pid_file = os.path.expanduser("~/.pixeltable/pgdata/postmaster.pid")
if os.path.exists(pid_file):
    with open(pid_file) as f:
        lines = f.readlines()
        if len(lines) > 3:
            port = lines[3].strip()
            print(f"Pixeltable PostgreSQL port: {port}")