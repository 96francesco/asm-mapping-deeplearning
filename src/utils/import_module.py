# src/utils/clone_repo.py (or similar script for cloning)
import os

# Path to the utils directory
utils_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

# Path to clone the repository
repo_path = os.path.join(utils_dir, 'gee_s1_ard')

# Clone the repository into the utils directory
if not os.path.exists(repo_path):
    os.system(f'git clone https://github.com/adugnag/gee_s1_ard.git {repo_path}')
