rank=10
eta=0.00001
iter=20

echo The rank is $rank
echo The eta is $eta
echo The iter is $iter

mpicxx dsgd_mf_template1.cpp -Wall -I/usr/local/include -L/usr/local/lib  -lgsl -lgslcblas -lm -o dsgd_mf
mpirun -n 4 ./dsgd_mf 3883 6040 $rank $iter $eta 1

mpicxx compute_rmse.cpp -Wall -I/usr/local/include -L/usr/local/lib  -lgsl -lgslcblas -lm -o compute_rmse
mpirun -n 10 ./compute_rmse 3883 6040 $rank $iter 4
