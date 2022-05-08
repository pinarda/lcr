import csv

if __name__ == "__main__":
    csvfilename = "../data/daily_zfp_bg_sz_comp_slices.csv"
    dssim_csvfilename = "../data/daily_zfp_bg_sz_comp_slices_dssim.csv"
    with open(csvfilename, newline='') as csvfile:
        with open(dssim_csvfilename, newline='') as dssim_csvfile:
            reader = csv.reader(csvfile)
            dssim_reader = csv.reader(dssim_csvfile)
            reader.__next__()
            dssim_reader.__next__()
            for row in reader:
                print(row)
                dssim_row = dssim_reader.__next__()
                print(dssim_row)
            # for i in range(0, reader.line_num):
            #     print(reader[i])
            #     print(dssim_reader[i])
            # for i in range(0,len(reader)):
            #     reader[i]
            #     dssim_reader[i]