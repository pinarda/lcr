"""
Currently this is used to remove all lines from a csv file at a specified zfp level or above
"""
import csv
import regex as re

if __name__ == "__main__":
    with open('../../data/everything/daily_dssims.csv', 'w', newline='') as writefile:
        writer = csv.writer(writefile, delimiter=',')
        with open("../../data/batch_data/all_calcs.csv", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                m = re.search('zfp_p_2+', row[0])
                if m is not None:
                    print(', '.join(row))
                else:
                    writer.writerow(row)

