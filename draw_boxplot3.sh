module load python/3.6.0

python main_knn3.py -i MHA_model4.npz -n 17 -weights uniform > result/boxplot_original4.dat
python main_svm3.py -i MHA_model4.npz -C 0.125 -kernel rbf -gamma 0.03125 >> result/boxplot_original4.dat
python main_rf3.py -i MHA_model4.npz -criterion gini -n 8 -minss 8 >> result/boxplot_original4.dat
python main_nn3.py -i MHA_model4.npz -hls 16 16 16 -activation identity -solver adam -alpha 0.0625 >> result/boxplot_original4.dat

python plot_best_four_boxplot.py -i result/boxplot_original4.dat -k 5 -t Model3 -o plots/boxplot_original3.png

