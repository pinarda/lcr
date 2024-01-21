rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
python main2.py --onlydata True --runonlydata True --metric dssim
#python main2.py --onlydata False --metric dssim

#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
#python main2.py --onlydata True --runonlydata True --metric spre
#python main2.py --onlydata False --metric spre

#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
#python main2.py --onlydata True --runonlydata True --metric pcc
#python main2.py --onlydata False --metric pcc

#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/data/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/plots/*.npy
#rm -f /Users/alex/git/lcr/lcr/data_analysis/RFnew/*.h5
#python main2.py --onlydata True --runonlydata True --metric ks
#python main2.py --onlydata False --metric ks