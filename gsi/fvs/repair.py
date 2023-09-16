#
# standard imports
#
import time
import sys
import shutil
import os
import socket
import argparse

#
# external/installed packages
#
import numpy
import numpy as np
import pandas as pd
import swagger_client
from swagger_client.models import *
from swagger_client.rest import ApiException

#
# functions
#

def compute_recall(a, b):
    '''Computes the recall metric on query results.'''

    nq, rank = a.shape
    intersect = [ numpy.intersect1d(a[i, :rank], b[i, :rank]).size for i in range(nq) ]
    ninter = sum( intersect )
    return ninter / a.size, intersect

def unload_datasets(args):
    if args.unload:
        # Setup connection to local FVS api
        server = socket.gethostbyname(socket.gethostname())
        port = "7761"
        version = 'v1.0'

        # Create FVS api objects
        config = swagger_client.configuration.Configuration()
        api_config = swagger_client.ApiClient(config)
        gsi_boards_apis = swagger_client.BoardsApi(api_config)
        gsi_datasets_apis = swagger_client.DatasetsApi(api_config)

        # Configure the FVS api
        config.verify_ssl = False
        config.host = f'http://{server}:{port}/{version}'
    
        # Capture the supplied allocation id
        Allocation_id = args.allocation

        # Set default header
        api_config.default_headers["allocationToken"] = Allocation_id

        # Print dataset count
        print("Getting total datasets...")
        dsets = gsi_datasets_apis.controllers_dataset_controller_get_datasets_list(allocation_token=Allocation_id)
        print(f"Number of datasets:{len(dsets.datasets_list)}")

        # if no datasets skip everything
        if len(dsets.datasets_list) > 0:
            # Print loaded dataset count
            print("Getting loaded datasets for allocation token: ", Allocation_id)
            loaded = gsi_boards_apis.controllers_boards_controller_get_allocations_list(Allocation_id)
            print(f"Number of loaded datasets: {len(loaded.allocations_list[Allocation_id]['loadedDatasets'])}")
            # check loaded dataset count
            if len(loaded.allocations_list[Allocation_id]["loadedDatasets"]) > 0:
                # Unloading all datasets
                print("Unloading all loaded datasets...")
                if True:
                    loaded = loaded.allocations_list[Allocation_id]["loadedDatasets"]
                    for data in loaded:
                        dataset_id = data['datasetId']
                        print("Unloading dataset_id=", dataset_id)
                        resp = gsi_datasets_apis.controllers_dataset_controller_unload_dataset(
                                    UnloadDatasetRequest(allocation_id=Allocation_id, dataset_id=dataset_id), 
                                    allocation_token=Allocation_id)
                        if resp.status != 'ok':
                            print(f"error unloading dataset: {dataset_id}")

                    # Getting current number of loaded datasets
                    curr = gsi_boards_apis.controllers_boards_controller_get_allocations_list(Allocation_id)
                    print(f"Unloaded datasets, current loaded dataset count: {len(curr.allocations_list[Allocation_id]['loadedDatasets'])}")
                sys.exit(0)

        # Full wipe: delete all datasets
        if args.wipe == True:
            wipe = input("are you super sure? y/[n]: ")
            if wipe == "y":
                print("removing all datasets...")
                for data in dsets.datasets_list:
                    dataset_id = data['id']
                    resp = gsi_datasets_apis.controllers_dataset_controller_remove_dataset(\
                            dataset_id=dataset_id, allocation_token=Allocation_id)
                    if resp.status != "ok":
                        print(f"Error removing dataset: {dataset_id}")

        else:
            print("Currently no loaded datasets. Done.")
    else: 
        print("Done")


def run_repair(args):
    '''Run a specific benchmark.'''

    # Capture rows for dataframe
    all_results = []
    
    # Capture the human readable date time now
    ts_start = time.ctime()

    # Make sure dataset is under /home/public
    dataset_public_path = os.path.join("/home/public", \
            os.path.basename( args.dataset ) )
    if not os.path.exists(dataset_public_path):
        print("Copy dataset to /home/public/ ...")
        shutil.copyfile( args.dataset, dataset_public_path)    

    # Setup connection to local FVS api
    server = socket.gethostbyname(socket.gethostname())
    port = "7761"
    version = 'v1.0'

    # Create FVS api objects
    config = swagger_client.configuration.Configuration()
    api_config = swagger_client.ApiClient(config)
    gsi_datasets_apis = swagger_client.DatasetsApi(api_config)
    gsi_dataset_apis = swagger_client.DatasetsApi(api_config)
    gsi_search_apis = swagger_client.SearchApi(api_config)
    gsi_utilities_apis = swagger_client.UtilitiesApi(api_config)

    # Configure the FVS api
    config.verify_ssl = False
    config.host = f'http://{server}:{port}/{version}'
  
    # Capture the supplied allocation id
    Allocation_id = args.allocation

    # Set default header
    api_config.default_headers["allocationToken"] = Allocation_id

    # Import dataset
    print("Importing the dataset. Training with searchtype=clusters and nbit=768")
    ts_train_start = time.time()
    response = gsi_datasets_apis.controllers_dataset_controller_import_dataset( \
                    ImportDatasetRequest(records=dataset_public_path, train_ind=True,  search_type="clusters", \
                            nbits=768), allocation_token=Allocation_id)
    dataset_id = response.dataset_id
    print("...got dataset_id=", dataset_id)

    # Wait until training the dataset before loading it
    print("Waiting until training ends...")

    dataset_status = None
    while dataset_status != "completed":

        dataset_status = gsi_datasets_apis.controllers_dataset_controller_get_dataset_status(
            dataset_id=dataset_id, allocation_token=Allocation_id).dataset_status

        datasets = gsi_datasets_apis.controllers_dataset_controller_get_datasets_list(
            allocation_token=Allocation_id).datasets_list

        for dataset in datasets:
            if dataset["id"] == dataset_id and dataset["datasetStatus"] == "error":
                print("ERROR: got dataset error", dataset)
                return

        time.sleep(1)
        print("waiting...")

    ts_train_end = time.time()
    print("..training ended.")

    # Load dataset
    print("Loading the dataset...")
    gsi_datasets_apis.controllers_dataset_controller_load_dataset(
                    LoadDatasetRequest(allocation_id=Allocation_id, dataset_id=dataset_id ), 
                    allocation_token=Allocation_id)

    # Set in focusa
    print("Setting focus...")
    gsi_datasets_apis.controllers_dataset_controller_focus_dataset(
        FocusDatasetRequest(allocation_id=Allocation_id, dataset_id=dataset_id), 
        allocation_token=Allocation_id)

    # Unload dataset
    print("Unloading dataset...")
    gsi_datasets_apis.controllers_dataset_controller_unload_dataset(
                    UnloadDatasetRequest(allocation_id=Allocation_id, dataset_id=dataset_id), 
                    allocation_token=Allocation_id)

    # Removing dataset...
    print("Removing dataset...")
    gsi_datasets_apis.controllers_dataset_controller_remove_dataset(dataset_id=dataset_id, 
            allocation_token=Allocation_id)


def init_args():
    '''Initialize benchmark parameters.'''
    
    parser = argparse.ArgumentParser(
                    prog = sys.argv[0],
                    description = 'Gemini FVS Benchmark Script')
    parser.add_argument('-a','--allocation', required=True)
    parser.add_argument('-d','--dataset', required=True)
    parser.add_argument('-u', '--unload', required=False, default=True, action='store_false')
    parser.add_argument('-w', '--wipe', required=False, default=False, action='store_true')
    args = parser.parse_args()

    if not os.path.exists( args.dataset ):
        raise Exception("The datset path does not exist->", args.dataset)

    return args

if __name__ == "__main__":

    args = init_args()
    unload_datasets(args) 
    run_repair(args)

    print("Done.")
