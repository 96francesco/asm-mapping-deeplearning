import pytest
import os

# import custom modules and classes to be sted
from data.planet_dataset import PlanetDataset

# cover all the different cases
@pytest.fixture(params=[
    {"is_inference": True, "is_fusion": True, "data_dir": 
     '/mnt/guanabana/raid/home/pasan001/thesis/dataset/inference_dataset/images'},
    {"is_inference": True, "is_fusion": False, "data_dir": 
     '/mnt/guanabana/raid/home/pasan001/thesis/dataset/inference_dataset/images/planet'},
    {"is_inference": False, "is_fusion": True, "data_dir": 
     '/mnt/guanabana/raid/home/pasan001/thesis/dataset/asm_dataset_split_0/fusion/training_data'},
    {"is_inference": False, "is_fusion": False, "data_dir": 
     '/mnt/guanabana/raid/home/pasan001/thesis/dataset/asm_dataset_split_0/planet/binary/training_data'}
])
def planet_dataset(request):
    config = request.param
    dataset = PlanetDataset(
        data_dir=config["data_dir"],
        is_inference=config["is_inference"],
        is_fusion=config["is_fusion"]
    )
    return dataset, config

# test dataset initialization
def test_planet_dataset_initialization(planet_dataset):
    dataset, config = planet_dataset
    assert dataset is not None
    assert isinstance(dataset, PlanetDataset)
