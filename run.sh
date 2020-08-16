#!/bin/bash
# Example Bash Script to execute <executable_name> on batch

#Change to the directory that the job was submitted from
cd /lyceum/rz2u19/DeepRL

#Run <executable_name> from the working directory
source /lyceum/rz2u19/anaconda3/bin/activate
python test.py
