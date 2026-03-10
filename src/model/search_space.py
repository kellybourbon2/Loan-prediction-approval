from hyperopt import hp
from hyperopt.pyll.base import scope

search_space = hp.choice(
    "model_type",
    [
        {
            "model_name": "random_forest",
            "n_estimators": scope.int(hp.quniform("rf_n_estimators", 50, 300, 100)),
            "max_depth": scope.int(hp.quniform("rf_max_depth", 3, 20, 6)),
        },
        # {
        #     "model_name": "xgboost",
        #     "max_depth": scope.int(hp.quniform("xgb_max_depth", 3, 10, 1)),
        #     "learning_rate": hp.loguniform("xgb_learning_rate", -3, 0),
        #     "reg_alpha": hp.loguniform("xgb_reg_alpha", -5, -1),
        #     "reg_lambda": hp.loguniform("xgb_reg_lambda", -6, -1),
        #     "min_child_weight": hp.quniform("xgb_min_child_weight", 1, 10, 1),
        #     "seed": scope.int(hp.quniform("xgb_seed", 0, 100, 1)),
        # },
        # {
        #     "model_name": "catboost",
        #     "depth": scope.int(hp.quniform("cat_depth", 3, 10, 1)),
        #     "learning_rate": hp.loguniform("cat_learning_rate", -3, 0),
        #     "l2_leaf_reg": hp.loguniform("cat_l2_leaf_reg", -3, 3),
        #     "iterations": scope.int(hp.quniform("cat_iterations", 100, 500, 100)),
        #     "seed": scope.int(hp.quniform("cat_seed", 0, 100, 1)),
        # },
    ],
)
