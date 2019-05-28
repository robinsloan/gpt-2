PYTHONPATH=src \
python ./train.py \
--run_name fantasy400_1 \
--dataset fantasy-400M-newlines.txt.npz \
--noise 0.1 \
--model_name 345M \
--memory_saving_gradients \
--batch_size 1 \
--save_every 10000 \
--sample_every 5000 \
--val_every 5000
