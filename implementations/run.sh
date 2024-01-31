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

python3 1training.py --dataset bail --epochs 1000 --lr 0.008 --hidden 146 --weight_decay 0.0006 --dropout 0.5
# python3 2_influence_computation_and_save.py --dataset bail
# python3 3_removing_and_testing.py --dataset bail --helpfulness_collection 0
# python3 3_removing_and_testing.py --dataset bail --helpfulness_collection 1
python3 debiasing_gnns.py --dataset bail

# python3 1training.py --dataset pokec2 --epochs 1000 --lr 0.01 --hidden 44 --weight_decay 0.0005 --dropout 0.7
# python3 2_influence_computation_and_save.py --dataset pokec2
# python3 3_removing_and_testing.py --dataset pokec2 --helpfulness_collection 0
# python3 3_removing_and_testing.py --dataset pokec2 --helpfulness_collection 1
python3 debiasing_gnns.py --dataset pokec2
# ./run.sh | tee output.txt

python 1training.py --dataset nba
python 2_influence_computation_and_save.py --dataset nba
python 3_removing_and_testing.py --dataset nba --helpfulness_collection 1

python 1training.py --dataset income
python 2_influence_computation_and_save.py --dataset income
python 3_removing_and_testing.py --dataset income --helpfulness_collection 1
python3 debiasing_gnns.py --dataset income

python 1training.py --dataset nba
python 2_influence_computation_and_save.py --dataset nba
python 3_removing_and_testing.py --dataset nba --helpfulness_collection 1
python3 debiasing_gnns.py --dataset nba

python3 debiasing_gnns.py --dataset income
python3 debiasing_gnns.py --dataset bail
python3 debiasing_gnns.py --dataset pokec1
python3 debiasing_gnns.py --dataset pokec2


python param_tune.py --dataset bail --model gcn --seed 2
python param_tune.py --dataset bail --model gcn --seed 5


python param_tune.py --dataset pokec2 --model sage --seed 1
python param_tune.py --dataset pokec2 --model sage --seed 2
python param_tune.py --dataset pokec2 --model sage --seed 3
python param_tune.py --dataset pokec2 --model sage --seed 4
python param_tune.py --dataset pokec2 --model sage --seed 5



python param_tune.py --dataset nba --model sage --seed 2
python param_tune.py --dataset nba --model sage --seed 3
python param_tune.py --dataset nba --model sage --seed 4
python param_tune.py --dataset nba --model sage --seed 5

python param_tune.py --dataset nba --model gat --seed 1
python param_tune.py --dataset nba --model gat --seed 2
python param_tune.py --dataset nba --model gat --seed 3
python param_tune.py --dataset nba --model gat --seed 4
python param_tune.py --dataset nba --model gat --seed 5


python param_tune.py --dataset income --model sage --seed 5
python param_tune.py --dataset income --model sage --seed 3

python param_tune.py --dataset bail --model gat --seed 1
python param_tune.py --dataset bail --model gat --seed 2
python param_tune.py --dataset bail --model gat --seed 5

python param_tune.py --dataset nba --model gcn --seed 1
python param_tune.py --dataset income --model sage --seed 2
python param_tune.py --dataset income --model sage --seed 3
python param_tune.py --dataset income --model sage --seed 4
python param_tune.py --dataset income --model sage --seed 5


hidden, dropout, lr, weight_decay, ap

python param_tune.py --dataset pokec2 --model gcn --seed 5


python test_ALL.py --dataset bail --model gat --seed 1 \
    --hidden 16 --dropout 0.07 --lr 0.01 --weight_decay 0.0001 --ap 1000
python test_ALL.py --dataset bail --model gat --seed 2 \
    --hidden 128 --dropout 0.6 --lr 0.01 --weight_decay 0.001 --ap 200
python test_ALL.py --dataset bail --model gat --seed 3  \
    --hidden 64 --dropout 0.3 --lr 0.01 --weight_decay 0.001 --ap 1000
python test_ALL.py --dataset bail --model gat --seed 4 \
    --hidden 4 --dropout 0.6 --lr 0.01 --weight_decay 0.001 --ap 60

python test_ALL.py --dataset bail --model sage --seed 3 \
    --hidden 64 --dropout 0.6 --lr 0.01 --weight_decay 0.0001 --ap 1000


64,0.7,0.01,0.001,1000,bail,1,0.9769018859927951,BIND_sage seed = 1
128,0.4,0.001,0.001,1000,bail,2,0.975206611570248,BIND_sage seed = 2
64,0.6,0.01,0.0001,1000,bail,3,0.9749947022674296,BIND_sage seed = 3
128,0.5,0.0001,0.0001,200,bail,4,0.9044289044289044,BIND_sage seed = 4
128,0.4,0.0001,0.01,200,bail,5,0.8948929858020768,BIND_sageseed = 5

python test_ALL.py --dataset bail --model sage --seed 1 --hidden 64 --dropout 0.7 --lr 0.01 --weight_decay 0.001 --ap 1000
python test_ALL.py --dataset bail --model sage --seed 2 --hidden 128 --dropout 0.4 --lr 0.001 --weight_decay 0.001 --ap 1000
python test_ALL.py --dataset bail --model sage --seed 3 --hidden 64 --dropout 0.6 --lr 0.01 --weight_decay 0.0001 --ap 1000
python test_ALL.py --dataset bail --model sage --seed 4 --hidden 128 --dropout 0.5 --lr 0.0001 --weight_decay 0.0001 --ap 200
python test_ALL.py --dataset bail --model sage --seed 5 --hidden 128 --dropout 0.2 --lr 0.0001 --weight_decay 0.01 --ap 200