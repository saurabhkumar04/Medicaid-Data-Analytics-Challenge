module load python/3.6.0

python main_knn3.py -i MHA_model2.npz -n 17 -weights uniform > result/boxplot_original2.dat
python main_svm3.py -i MHA_model2.npz -C 0.125 -kernel rbf -gamma 0.03125 >> result/boxplot_original2.dat
python main_rf3.py -i MHA_model2.npz -criterion gini -n 8 -minss 8 >> result/boxplot_original2.dat
python main_nn3.py -i MHA_model2.npz -hls 16 16 16 -activation identity -solver adam -alpha 0.0625 >> result/boxplot_original2.dat

python plot_best_four_boxplot.py -i result/boxplot_original2.dat -k 5 -t Model2 -o plots/boxplot_original2.png
