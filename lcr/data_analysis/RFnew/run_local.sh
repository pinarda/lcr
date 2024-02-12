rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*
rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*
rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5

python main2.py --onlydata True --runonlydata True --metric dssim -j RF_local_dssim.json
python main2.py --onlydata False --runonlydata False --metric dssim -j RF_local_dssim.json

#
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
python main2.py --onlydata True --runonlydata True --metric spre -j RF_local_spre.json
python main2.py --onlydata False --runonlydata False --metric spre -j RF_local_spre.json
#
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
python main2.py --onlydata True --runonlydata True --metric pcc -j RF_local_pcc.json
python main2.py --onlydata False --runonlydata False --metric pcc -j RF_local_pcc.json
#
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
python main2.py --onlydata True --runonlydata True --metric ks -j RF_local_ks.json
python main2.py --onlydata False --runonlydata False --metric ks -j RF_local_ks.json