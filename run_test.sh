cd /ghome/lijj/DA/DA_ours/
#CUDA_VISIBLE_DEVICES=$1 python main_eposide.py --method $2 --dataset office_home --source Real --target Clipart --net $3 --save_check 
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Art --target Clipart --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Product --target Art --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Product --target Real --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Product --target Clipart --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Art --target Real --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Art --target Product --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Art --target Clipart --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Clipart --target Product --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Clipart --target Art --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Clipart --target Real --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Real --target Art --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Real --target Clipart --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset office_home --source Real --target Product --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset multi --source real --target painting --net resnet34 --save_check
#CUDA_VISIBLE_DEVICES=$1 python test.py --method MME --dataset multi --source real --target clipart --net resnet34 --save_check
python test.py --method MME --dataset multi --source real --target painting --net resnet34 --save_check
python test.py --method MME --dataset multi --source real --target clipart --net resnet34 --save_check
python test.py --method MME --dataset multi --source clipart --target sketch --net resnet34 --save_check
python test.py --method MME --dataset multi --source sketch --target painting --net resnet34 --save_check
python test.py --method MME --dataset multi --source real --target sketch --net resnet34 --save_check
python test.py --method MME --dataset multi --source painting --target clipart --net resnet34 --save_check
#python test.py --method MME --dataset multi --source painting --target real --net resnet34 --save_check