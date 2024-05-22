set title "Loss over time"
set xlabel "Step"
set ylabel "Loss"
plot "loss_data.txt" using 1:2 with lines title "Loss"