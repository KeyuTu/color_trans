cd /ghome/lijj/DA/DA_ours/
#CUDA_VISIBLE_DEVICES=$1 python main_eposide.py --method $2 --dataset office_home --source Real --target Clipart --net $3 --save_check 
#CUDA_VISIBLE_DEVICES=$1 python main_eposide_transformer.py --method MME --dataset office_home --source Clipart --target Art --net resnet34 --save_check 
#CUDA_VISIBLE_DEVICES=$1 python main_eposide_transformer.py --method MME --dataset office_home --source Art --target Real --net resnet34 --save_check 
#CUDA_VISIBLE_DEVICES=$1 python main.py --method MME --dataset office_home --source Art --target Clipart --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Real --target Clipart --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Real --target Art --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Real --target Product --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Clipart --target Real --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Clipart --target Art --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Clipart --target Product --net resnet34 --save_check

#CUDA_VISIBLE_DEVICES=$1 python main_eposide.py --method MME --dataset office_home --source Art --target Clipart --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python main_data_augment.py --method MME --dataset office_home --source Art --target Clipart --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python main_eposide_transformer_fuseunlabel.py --method MME --dataset office_home --source Art --target Clipart --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl.py --method MME --dataset office_home --source Art --target Clipart --net resnet34 --save_check
python train_strong_data_augment.py --method MME --dataset office_home --source Art --target Clipart --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Art --target Real --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Art --target Product --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Real --target Art --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Real --target Product --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Real --target Clipart --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Clipart --target Art --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Clipart --target Product --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Clipart --target Real --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Product --target Art --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Product --target Clipart --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Product --target Real --net resnet34 --reconstruction --save_check


