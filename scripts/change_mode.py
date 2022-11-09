import os
import sys

path = sys.argv[1]

os.chmod(path, 0o755)