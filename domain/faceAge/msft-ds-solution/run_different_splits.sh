g=$1
s=$2
python train_model.py --gpu_index $g --image_dir /home/jason/.kaggle/datasets/nih-chest-xrays/data/images --output_dir models/keep_gold_standard_norm_pleural_seed_${s} --resplit_data --seed $s --evaluate --keep_gold_standard --decay --augment --epochs 50 --pretrain --batch_size 48 --preprocess

