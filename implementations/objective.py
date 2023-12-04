# from pygdebias.debiasing import FairGNN
# from pygdebias.datasets import Bail
# /home/joyce/BIND_2/implementations/1_training.py
from BIND_2.implementations.1_training import train, test
import optuna

# Available choices: 'Credit', 'German', 'Facebook', 'Pokec_z', 'Pokec_n', 'Nba', 'Twitter', 'Google', 'LCC', 'LCC_small', 'Cora', 'Citeseer', 'Amazon', 'Yelp', 'Epinion', 'Ciao', 'Dblp', 'Filmtrust', 'Lastfm', 'Ml-100k', 'Ml-1m', 'Ml-20m', 'Oklahoma', 'UNC', 'Bail'.

print("hi")

# def objective(trial):
#     # Define the hyperparameter search space
#     # num_hidden = trial.suggest_int("num_hidden", 4, 130)
#     # proj_hidden = trial.suggest_int("proj_hidden", 4, 130)
#     epochs = trial.suggest_int("epochs", 200, 3000)
#     lr = trial.suggest_float("lr", 0.00001, 0.01)
#     weight_decay = trial.suggest_float("weight_decay", 0.000001, 0.001)
#     hidden = trial.suugest_int("hidden", 1, 200)
#     dropout = trial.suggest_int("dropout", 0.2, 0.8)
#     # sim_coeff = trial.suggest_float("sim_coeff", 0.3, 0.7)
#     # n_order = trial.suggest_int("n_order", 5, 100)
#     # subgraph_size = trial.suggest_int("subgraph_size", 10, 200)
#     # acc = trial.suggest_float("acc", 0.4, 0.9)
#     # hidden_size = trial.suggest_int("hidden_size", 20, 2000)
#     # alpha = trial.suggest_int("alpha", 1, 2000)
#     # beta = trial.suggest_int("beta", 1, 200)
#     # sens_number = trial.suggest_int("sens_number", 20, 1000)
#     # label_number = trial.suggest_int("label_number", 20, 500)
#     # calling load_nba
#     bail = Bail()
#     adj, features, idx_train, idx_val, idx_test, labels, sens = (
#         bail.adj(),
#         bail.features(),
#         bail.idx_train(),
#         bail.idx_val(),
#         bail.idx_test(),
#         bail.labels(),
#         bail.sens(),
#     )
#     # Create GNN model with suggested hyperparameters
#     model = FairGNN(nfeat=features.shape[1],
#                     lr = lr,
#                     weight_decay = weight_decay,
#                     proj_hidden = proj_hidden,
#                     hidden_size = hidden_size,
#                     num_hidden = num_hidden,
#                     alpha = alpha,
#                     beta = beta,
#                     sim_coeff = sim_coeff, 
#                     n_order = n_order, 
#                     subgraph_size = subgraph_size, 
#                     acc = acc,
#                     epoch = 2000
#                     )
#     model.fit(adj, features, labels, idx_train, idx_val, idx_test, sens, idx_train)
#     # Evaluate the model.

#     (
#         ACC,
#         # AUCROC,
#         F1,
#         recall, 
#         precision,
#         cm,
#         ACC_sens0,
#         # AUCROC_sens0,
#         F1_sens0,
#         ACC_sens1,
#         # AUCROC_sens1,
#         F1_sens1,
#         SP,
#         EO,
#     ) = model.predict(idx_test)

#     return ACC
# # Create an Optuna study object and specify the optimization direction
# study = optuna.create_study(direction='maximize')

# # Run the optimization
# study.optimize(objective, n_trials=100)  # Run 100 trials

# # Get the best hyperparameters and their value
# best_params = study.best_params
# best_value = study.best_value


# # Get the best trial
# best_trial = study.best_trial

# # Print the best trial number and value
# print(f"Best trial number: {best_trial.number}")
# print(f"Best value: {best_trial.value}")

# # Get the best parameters
# best_params = best_trial.params

# # Print the best parameters
# print("Best parameters:")
# for key, value in best_params.items():
#     print(f"{key}: {value}")