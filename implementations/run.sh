# mkdir test
# python3 main.py

# python3 1training.py --dataset income --epochs 1000 --lr 0.003 --hidden 79 --weight_decay 0.0001 --dropout 0.5
# python3 2_influence_computation_and_save.py --dataset income
# python3 3_removing_and_testing.py --dataset income --helpfulness_collection 0
# python3 3_removing_and_testing.py --dataset income --helpfulness_collection 1
python3 debiasing_gnns.py --dataset income

# python3 1training.py --dataset pokec1 --epochs 1000 --lr 0.002 --hidden 131 --weight_decay 0.001 --dropout 0.23
# python3 2_influence_computation_and_save.py --dataset pokec1
# python3 3_removing_and_testing.py --dataset pokec1 --helpfulness_collection 0
# python3 3_removing_and_testing.py --dataset pokec1 --helpfulness_collection 1
python3 debiasing_gnns.py --dataset pokec1

# python3 1training.py --dataset bail --epochs 1000 --lr 0.008 --hidden 146 --weight_decay 0.0006 --dropout 0.5
# python3 2_influence_computation_and_save.py --dataset bail
# python3 3_removing_and_testing.py --dataset bail --helpfulness_collection 0
# python3 3_removing_and_testing.py --dataset bail --helpfulness_collection 1
python3 debiasing_gnns.py --dataset bail

# python3 1training.py --dataset pokec2 --epochs 1000 --lr 0.01 --hidden 44 --weight_decay 0.0005 --dropout 0.7
# python3 2_influence_computation_and_save.py --dataset pokec2
# python3 3_removing_and_testing.py --dataset pokec2 --helpfulness_collection 0
# python3 3_removing_and_testing.py --dataset pokec2 --helpfulness_collection 1
python3 debiasing_gnns.py --dataset pokec2
./run.sh | tee output.txt