#!/bin/bash

# Check if a username is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <username>"
  exit 1
fi

USER=$1

# Get list of job IDs and job names for the specified user
qstat -u "$USER" | awk 'NR>2 {split($1, a, "."); print a[1], $3}' | while read -r job_id job_name; do
  # Skip jobs with the name "stdin"
  if [ "$job_name" = "STDIN" ]; then
    echo "Skipping job $job_id with name 'stdin'"
    continue
  fi

  # Delete the job if it doesn't have the name "stdin"
  qdel "$job_id"
  echo "Deleted job $job_id for user $USER (name: $job_name)"
done

echo "Finished processing jobs for user $USER."
