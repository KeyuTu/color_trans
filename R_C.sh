cd /ghome/tuky/Summer/
#python train_sampling.py --dataset multi --source real --target clipart --net resnet34 --num 3 --NCE_weight 0.2 --save_check --random_sampling True --fixmatch False
python train_sampling.py  --source real --target clipart --random_sampling 1 --fixmatch 0