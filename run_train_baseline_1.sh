#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Art --target Real --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Art --target Product --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Product --target Real --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Product --target Art --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=1 python main.py --method MME --dataset office_home --source Product --target Clipart --net resnet34 --save_check

#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Clipart --target Art --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Clipart --target Product --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Clipart --target Real --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Product --target Art --net resnet34 --reconstruction --save_check
#CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Product --target Clipart --net resnet34 --reconstruction --save_check
CUDA_VISIBLE_DEVICES=$1 python train_ifsl_restruction.py --method MME --dataset office_home --source Product --target Real --net resnet34 --reconstruction --save_check

