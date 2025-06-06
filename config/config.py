import os

config = {
    "base_path": "C:/Users/kom49df/PycharmProjects/flipper/",
    "training_path": "data/train",
    "validation_path": "data/validation",
    "test_path": "data/test",
    "training_amount_percentage": 0.7,
    "validation_amount_percentage": 0.2,
    "test_amount_percentage": 0.1
}

def get_training_path():
    return os.path.join(config["base_path"], config["training_path"])

def get_validation_path():
    return os.path.join(config["base_path"], config["validation_path"])

def get_test_path():
    return os.path.join(config["base_path"], config["test_path"])

def get_training_amount():
    return config["training_amount_percentage"]

def get_validation_amount():
    return config["validation_amount_percentage"]

def get_test_amount():
    return config["test_amount_percentage"]