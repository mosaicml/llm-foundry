import subprocess
import sys
# execute some shell commands

# git clone git@github.com:databricks-mosaic/mcloud.git to /tmp/mcloud
# git checkout pyhookbuffered
subprocess.run(["git", "clone", "git@github.com:databricks-mosaic/mcloud.git", "/tmp/mcloud"])
subprocess.run(["git", "checkout", "pyhookbuffered"], cwd="/tmp/mcloud")


# add the following to the PYTHONPATH:
# /tmp/mcloud/finetuning/pyhook
sys.path.append("/tmp/mcloud/finetuning/pyhook")