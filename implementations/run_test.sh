# BIND_GAT section
# bail BIND_GAT
# python test_ALL.py --dataset bail --model gat --seed 1 --hidden 16 --dropout 0.07 --lr 0.01 --weight_decay 0.0001 --ap 1000
# python test_ALL.py --dataset bail --model gat --seed 2 --hidden 128 --dropout 0.6 --lr 0.01 --weight_decay 0.001 --ap 200
# python test_ALL.py --dataset bail --model gat --seed 3 --hidden 64 --dropout 0.3 --lr 0.01 --weight_decay 0.001 --ap 1000
# python test_ALL.py --dataset bail --model gat --seed 4 --hidden 4 --dropout 0.6 --lr 0.01 --weight_decay 0.001 --ap 60
# python test_ALL.py --dataset bail --model gat --seed 5 --hidden 4 --dropout 0.4 --lr 0.01 --weight_decay 0.00001 --ap 200

# # # pokec1 BIND_GAT, pokec1 is the same as pokec_z in PyG
# python test_ALL.py --dataset pokec1 --model gat --seed 1 \
#     --hidden 128 --dropout 0.7 --lr 0.0001 --weight_decay 0.01 --ap 10
# python test_ALL.py --dataset pokec1 --model gat --seed 2 \
#     --hidden 128 --dropout 0.6 --lr 0.0001 --weight_decay 0.00001 --ap 100
# python test_ALL.py --dataset pokec1 --model gat --seed 3  \
#     --hidden 64 --dropout 0.7 --lr 0.0001 --weight_decay 0.01 --ap 10
# python test_ALL.py --dataset pokec1 --model gat --seed 4 \
#     --hidden 128 --dropout 0.7 --lr 0.0001 --weight_decay 0.00001 --ap 10
# python test_ALL.py --dataset pokec1 --model gat --seed 5 \
#     --hidden 64 --dropout 0.3 --lr 0.0001 --weight_decay 0.0001 --ap 200

# # # pokec2 BIND_GAT, pokec2 is the same as pokec_n in PyG
# python test_ALL.py --dataset pokec2 --model gat --seed 1 \
#     --hidden 128 --dropout 0.6 --lr 0.0001 --weight_decay 0.01 --ap 1000
# python test_ALL.py --dataset pokec2 --model gat --seed 2 \
#     --hidden 128 --dropout 0.3 --lr 0.01 --weight_decay 0.0001 --ap 200
# python test_ALL.py --dataset pokec2 --model gat --seed 3  \
#     --hidden 4 --dropout 0.5 --lr 0.001 --weight_decay 0.0001 --ap 25
# python test_ALL.py --dataset pokec2 --model gat --seed 4 \
#     --hidden 128 --dropout 0.6 --lr 0.0001 --weight_decay 0.01 --ap 100
# python test_ALL.py --dataset pokec2 --model gat --seed 5 \
#     --hidden 128 --dropout 0.4 --lr 0.01 --weight_decay 0.01 --ap 80

# # # Income dataset for BIND_GAT
# python test_ALL.py --dataset income --model gat --seed 1 \
#     --hidden 64 --dropout 0.4 --lr 0.01 --weight_decay 0.00001 --ap 200
# python test_ALL.py --dataset income --model gat --seed 2 \
#     --hidden 16 --dropout 0.7 --lr 0.01 --weight_decay 0.00001 --ap 200
# python test_ALL.py --dataset income --model gat --seed 3  \
#     --hidden 128 --dropout 0.6 --lr 0.01 --weight_decay 0.0001 --ap 1000
# python test_ALL.py --dataset income --model gat --seed 4 \
#     --hidden 64 --dropout 0.6 --lr 0.01 --weight_decay 0.00001 --ap 1000
# python test_ALL.py --dataset income --model gat --seed 5 \
#     --hidden 64 --dropout 0.7 --lr 0.01 --weight_decay 0.001 --ap 10

# # NBA dataset for BIND_GAT
# python test_ALL.py --dataset nba --model gat --seed 1 \
#     --hidden 16 --dropout 0.3 --lr 0.01 --weight_decay 0.01 --ap 10
# python test_ALL.py --dataset nba --model gat --seed 2 \
#     --hidden 4 --dropout 0.3 --lr 0.001 --weight_decay 0.01 --ap 10
# python test_ALL.py --dataset nba --model gat --seed 3  \
#     --hidden 64 --dropout 0.5 --lr 0.01 --weight_decay 0.0001 --ap 1000
# python test_ALL.py --dataset nba --model gat --seed 4 \
#     --hidden 4 --dropout 0.3 --lr 0.01 --weight_decay 0.001 --ap 1000
# python test_ALL.py --dataset nba --model gat --seed 5 \
#     --hidden 16 --dropout 0.3 --lr 0.001 --weight_decay 0.01 --ap 50

#BIND_GCN section
# python test_ALL.py --dataset pokec1 --model gcn --seed 1 \
#     --hidden 4 --dropout 0.7 --lr 0.0001 --weight_decay 0.01 --ap 1000
# python test_ALL.py --dataset pokec1 --model gcn --seed 2 \
#     --hidden 4 --dropout 0.5 --lr 0.0001 --weight_decay 0.01 --ap 80
# python test_ALL.py --dataset pokec1 --model gcn --seed 3  \
#     --hidden 16 --dropout 0.6 --lr 0.0001 --weight_decay 0.00001 --ap 50
# python test_ALL.py --dataset pokec1 --model gcn --seed 4 \
#     --hidden 64 --dropout 0.3 --lr 0.01 --weight_decay 0.001 --ap 80
# python test_ALL.py --dataset pokec1 --model gcn --seed 5 \
#     --hidden 4 --dropout 0.4 --lr 0.0001 --weight_decay 0.00001 --ap 60

# python test_ALL.py --dataset pokec2 --model gcn --seed 1 \
#     --hidden 4 --dropout 0.7 --lr 0.0001 --weight_decay 0.01 --ap 1000
# python test_ALL.py --dataset pokec2 --model gcn --seed 2 \
#     --hidden 4 --dropout 0.5 --lr 0.0001 --weight_decay 0.01 --ap 80
# python test_ALL.py --dataset pokec2 --model gcn --seed 3  \
#     --hidden 16 --dropout 0.6 --lr 0.0001 --weight_decay 0.00001 --ap 50
# python test_ALL.py --dataset pokec2 --model gcn --seed 4 \
#     --hidden 64 --dropout 0.3 --lr 0.01 --weight_decay 0.001 --ap 80
# python test_ALL.py --dataset pokec2 --model gcn --seed 5 \
#     --hidden 4 --dropout 0.4 --lr 0.0001 --weight_decay 0.00001 --ap 60

# python test_ALL.py --dataset income --model gcn --seed 1 \
#     --hidden 128 --dropout 0.3 --lr 0.01 --weight_decay 0.0001 --ap 100
# python test_ALL.py --dataset income --model gcn --seed 2 \
#     --hidden 128 --dropout 0.7 --lr 0.01 --weight_decay 0.0001 --ap 1000
# python test_ALL.py --dataset income --model gcn --seed 3  \
#     --hidden 64 --dropout 0.7 --lr 0.01 --weight_decay 0.0001 --ap 1000
# python test_ALL.py --dataset income --model gcn --seed 4 \
#     --hidden 128 --dropout 0.5 --lr 0.01 --weight_decay 0.01 --ap 80
# python test_ALL.py --dataset income --model gcn --seed 5 \
#     --hidden 4 --dropout 0.3 --lr 0.001 --weight_decay 0.001 --ap 1000

# python test_ALL.py --dataset bail --model gcn --seed 1 \
#     --hidden 128 --dropout 0.3 --lr 0.001 --weight_decay 0.0001 --ap 80
# python test_ALL.py --dataset bail --model gcn --seed 2 \
#     --hidden 128 --dropout 0.5 --lr 0.001 --weight_decay 0.00001 --ap 50
# python test_ALL.py --dataset bail --model gcn --seed 3  \
#     --hidden 128 --dropout 0.5 --lr 0.001 --weight_decay 0.0001 --ap 50
# python test_ALL.py --dataset bail --model gcn --seed 4 \
#     --hidden 4 --dropout 0.7 --lr 0.01 --weight_decay 0.0001 --ap 60
# python test_ALL.py --dataset bail --model gcn --seed 5 \
#     --hidden 4 --dropout 0.4 --lr 0.01 --weight_decay 0.00001 --ap 25

# python test_ALL.py --dataset nba --model gcn --seed 1 \
#     --hidden 16 --dropout 0.7 --lr 0.01 --weight_decay 0.01 --ap 25
# python test_ALL.py --dataset nba --model gcn --seed 2 \
#     --hidden 128 --dropout 0.3 --lr 0.01 --weight_decay 0.01 --ap 200
# python test_ALL.py --dataset nba --model gcn --seed 3  \
#     --hidden 16 --dropout 0.7 --lr 0.01 --weight_decay 0.00001 --ap 100
# python test_ALL.py --dataset nba --model gcn --seed 4 \
#     --hidden 128 --dropout 0.6 --lr 0.0001 --weight_decay 0.01 --ap 80
# python test_ALL.py --dataset nba --model gcn --seed 5 \
#     --hidden 4 --dropout 0.3 --lr 0.01 --weight_decay 0.01 --ap 200

# #BIND_SAGE
python test_ALL.py --dataset pokec1 --model sage --seed 1 \
    --hidden 128 --dropout 0.7 --lr 0.00001 --weight_decay 0.01 --ap 1000
python test_ALL.py --dataset pokec1 --model sage --seed 2 \
    --hidden 16 --dropout 0.6 --lr 0.01 --weight_decay 0.01 --ap 25
python test_ALL.py --dataset pokec1 --model sage --seed 3  \
    --hidden 16 --dropout 0.7 --lr 0.01 --weight_decay 0.001 --ap 200
python test_ALL.py --dataset pokec1 --model sage --seed 4 \
    --hidden 128 --dropout 0.7 --lr 0.0001 --weight_decay 0.01 --ap 50
python test_ALL.py --dataset pokec1 --model sage --seed 5 \
    --hidden 128 --dropout 0.6 --lr 0.00001 --weight_decay 0.0001 --ap 60

python test_ALL.py --dataset pokec2 --model sage --seed 1 \
    --hidden 128 --dropout 0.3 --lr 0.00001 --weight_decay 0.01 --ap 200
python test_ALL.py --dataset pokec2 --model sage --seed 2 \
    --hidden 128 --dropout 0.6 --lr 0.01 --weight_decay 0.00001 --ap 60
python test_ALL.py --dataset pokec2 --model sage --seed 3  \
    --hidden 64 --dropout 0.4 --lr 0.01 --weight_decay 0.01 --ap 1000
python test_ALL.py --dataset pokec2 --model sage --seed 4 \
    --hidden 4 --dropout 0.5 --lr 0.01 --weight_decay 0.01 --ap 1000
python test_ALL.py --dataset pokec2 --model sage --seed 5 \
    --hidden 128 --dropout 0.3 --lr 0.01 --weight_decay 0.001 --ap 1000

python test_ALL.py --dataset income --model sage --seed 1 \
    --hidden 16 --dropout 0.7 --lr 0.01 --weight_decay 0.001 --ap 200
python test_ALL.py --dataset income --model sage --seed 2 \
    --hidden 128 --dropout 0.3 --lr 0.01 --weight_decay 0.01 --ap 1000
python test_ALL.py --dataset income --model sage --seed 3  \
    --hidden 16 --dropout 0.6 --lr 0.001 --weight_decay 0.00001 --ap 80
python test_ALL.py --dataset income --model sage --seed 4 \
    --hidden 4 --dropout 0.7 --lr 0.01 --weight_decay 0.001 --ap 200
python test_ALL.py --dataset income --model sage --seed 5 \
    --hidden 4 --dropout 0.3 --lr 0.01 --weight_decay 0.0001 --ap 1000

python test_ALL.py --dataset bail --model sage --seed 1 \
    --hidden 64 --dropout 0.7 --lr 0.01 --weight_decay 0.001 --ap 1000
python test_ALL.py --dataset bail --model sage --seed 2 \
    --hidden 128 --dropout 0.4 --lr 0.001 --weight_decay 0.001 --ap 1000
python test_ALL.py --dataset bail --model sage --seed 3  \
    --hidden 64 --dropout 0.6 --lr 0.01 --weight_decay 0.0001 --ap 1000
python test_ALL.py --dataset bail --model sage --seed 4 \
    --hidden 128 --dropout 0.5 --lr 0.0001 --weight_decay 0.0001 --ap 200
python test_ALL.py --dataset bail --model sage --seed 5 \
    --hidden 128 --dropout 0.4 --lr 0.0001 --weight_decay 0.01 --ap 200

python test_ALL.py --dataset nba --model sage --seed 1 \
    --hidden 4 --dropout 0.5 --lr 0.01 --weight_decay 0.01 --ap 50
python test_ALL.py --dataset nba --model sage --seed 2 \
    --hidden 128 --dropout 0.5 --lr 0.001 --weight_decay 0.01 --ap 200
python test_ALL.py --dataset nba --model sage --seed 3  \
    --hidden 64 --dropout 0.4 --lr 0.00001 --weight_decay 0.01 --ap 1000
python test_ALL.py --dataset nba --model sage --seed 4 \
    --hidden 16 --dropout 0.4 --lr 0.0001 --weight_decay 0.01 --ap 200
python test_ALL.py --dataset nba --model sage --seed 5 \
    --hidden 128 --dropout 0.5 --lr 0.001 --weight_decay 0.01 --ap 60
