#!/bin/bash

# Check if a username is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <username>"
  exit 1
fi

USER=$1

# Get list of job IDs for the specified user
job_ids=$(qstat -u "$USER" | awk 'NR>2 {print $1}')

# Check if any jobs are found
if [ -z "$job_ids" ]; then
  echo "No jobs found for user $USER"
  exit 0
fi

# Delete each job found for the user
for job_id in $job_ids; do
  qdel "$job_id"
  echo "Deleted job $job_id for user $USER"
done

echo "All jobs for user $USER have been deleted."
