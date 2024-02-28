rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*
rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*
rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5

python main2.py --onlydata True --runonlydata True --metric dssim -j RF_template_dssim.json --labelsonly True
#python main2.py --onlydata False --runonlydata False --metric dssim -j RF_template_dssim.json --labelsonly True
#
##
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
#python main2.py --onlydata True --runonlydata True --metric spre -j RF_template_spre.json --labelsonly True
#python main2.py --onlydata False --runonlydata False --metric spre -j RF_template_spre.json --labelsonly True
#
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
##rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
#python main2.py --onlydata True --runonlydata True --metric pcc -j RF_template_pcc.json --labelsonly True
#python main2.py --onlydata False --runonlydata False --metric pcc -j RF_template_pcc.json --labelsonly True
#
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
#python main2.py --onlydata True --runonlydata True --metric ks -j RF_template_ks.json --labelsonly True
#python main2.py --onlydata False --runonlydata False --metric ks -j RF_template_ks.json --labelsonly True