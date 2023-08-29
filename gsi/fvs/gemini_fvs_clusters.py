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


def run_benchmark(args):
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

    # Make sure queries is under /home/public
    queries_public_path = os.path.join("/home/public", \
            os.path.basename( args.queries ) )
    if not os.path.exists(queries_public_path):
        print("Copy queries to /home/public/ ...")
        shutil.copyfile( args.queries, queries_public_path)    

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
    print("Importing the dataset. Training with searchtype=clusters and nbit=%d ..." % args.bits)
    ts_train_start = time.time()
    response = gsi_datasets_apis.controllers_dataset_controller_import_dataset( \
                    ImportDatasetRequest(records=dataset_public_path, train_ind=True,  search_type="clusters", \
                            nbits=args.bits), allocation_token=Allocation_id)
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
                print("ERROR: got dataset error")
                return

        time.sleep(1)
        print("waiting...")

    ts_train_end = time.time()
    print("..training ended.")

    # Store train results
    new_row = {
        "type": "train",
        "allocationid": Allocation_id, \
        "datasetid": dataset_id, \
        "dataset_path": args.dataset, \
        "queries_path": "", \
        "bits" : args.bits, \
        "ts_start": ts_start, \
        "ts_train_start": ts_train_start, \
        "ts_train_end": ts_train_end, \
        "ts_train_walltime": ts_train_end - ts_train_start, \
        "ts_query_start": "", \
        "ts_query_end": "", \
        "ts_query_walltime": "",\
        "response": "", \
        "recall": "" }
    all_results.append(new_row)

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

    if args.qbq:

        # query-by-query search
        print("Q-BY-Q Search...")

        # Load queries
        queries = np.load(queries_public_path)

        # Load ground truth
        print("Loading ground truth...")
        gt = numpy.load(args.groundtruth)
        if gt.shape[1]<args.neighbors:
            raise Exception("Invalid ground truth array shape->" + str(gt.shape))
        if gt.shape[0] != queries.shape[0]:
            raise Exception("Invalid ground truth array shape->" + str(gt.shape))
        gt = gt[:, 0:args.neighbors] # we only care about the first 'neighbors'

        for i, q in enumerate(queries):

            # choose the i'th ground truth
            g = np.reshape( gt[i,:], (1, gt[i,:].shape[0]) )

            # extract query to a tmp file
            base_name = os.path.basename( queries_public_path ).split(".")[0]
            tmp_qpath = os.path.join( "/home/public", "%s__%d.npy" % (base_name, i) )
            f = open(tmp_qpath, "wb")
            qrs = np.reshape(q, (1, q.shape[0]) )
            np.save( f, qrs )
            #print("Saved", tmp_qpath, qrs.shape)
            f.close()

            #print("Loading single query...")
            response = gsi_utilities_apis.controllers_utilities_controller_import_queries(\
                ImportQueriesRequest(queries_file_path=tmp_qpath),
                allocation_token=Allocation_id)
            queries_id = response.added_query["id"]
            queries_path = response.added_query["queriesFilePath"]

            # perform search - get walltime and inner response time
            ts_query_start = time.time()
            response = gsi_search_apis.controllers_search_controller_search(
                            SearchRequest(allocation_id=Allocation_id, dataset_id=dataset_id, 
                                queries_file_path=queries_path, topk=args.neighbors), allocation_token=Allocation_id)
            ts_query_end = time.time()
            inds = numpy.array(response.indices)

            # Compute recall...
            recall, intersection = compute_recall( g, inds )
            print("qbq search %d/%d, result=" % (i, queries.shape[0]), ts_query_end-ts_query_start, response.search, recall)

            # Store and commit the results
            new_row = {   
                "type": "search",
                "allocationid": Allocation_id, \
                "datasetid": dataset_id, \
                "dataset_path": "", \
                "queries_path": queries_path, \
                "bits" : "", \
                "ts_start": ts_start, \
                "ts_train_start": "", \
                "ts_train_end": "", \
                "ts_train_walltime": "", \
                "ts_query_start": ts_query_start, \
                "ts_query_end": ts_query_end, \
                "ts_query_walltime": ts_query_end-ts_query_start,\
                "response": response.search, \
                "recall": recall }
            all_results.append(new_row)

            # Remove queries...
            #print("Removing queries...")
            gsi_utilities_apis.controllers_utilities_controller_remove_query(query_id=queries_id,
                    allocation_token=Allocation_id)

    else:

        # Load the entire query set
        print("Loading the queries...")
        response = gsi_utilities_apis.controllers_utilities_controller_import_queries(\
            ImportQueriesRequest(queries_file_path=queries_public_path),
            allocation_token=Allocation_id)
        queries_id = response.added_query["id"]
        queries_path = response.added_query["queriesFilePath"]

        # Batch Search
        print("Batch search...")
        ts_query_start = time.time()
        response = gsi_search_apis.controllers_search_controller_search(
                        SearchRequest(allocation_id=Allocation_id, dataset_id=dataset_id, 
                            queries_file_path=queries_path, topk=args.neighbors), allocation_token=Allocation_id)
        ts_query_end = time.time()
        print("search result=", response.search)
        inds = numpy.array(response.indices)
        if args.save:
            # Save results alongside the csv
            path = os.path.dirname(args.output)
            fname = os.path.join(path, "query_inds.npy")
            numpy.save(fname, inds)
            print("Saved query results indices to", fname)
            dist = numpy.array(response.distance)
            fname = os.path.join(path, "query_dists.npy")
            numpy.save(fname, dist)
            print("Saved query results distances to", fname)

        # Removing queries...
        print("Removing queries...")
        gsi_utilities_apis.controllers_utilities_controller_remove_query(query_id=queries_id,
                allocation_token=Allocation_id)

        # Load ground truth
        print("Loading ground truth...")
        gt = numpy.load(args.groundtruth)
        if gt.shape[1]<args.neighbors:
            raise Exception("Invalid ground truth array shape->" + str(gt.shape))
        if gt.shape[0] != inds.shape[0]:
            raise Exception("Invalid ground truth array shape->" + str(gt.shape))
        gt = gt[:, 0:args.neighbors] # we only care about the first 'neighbors'

        # Compute recall...
        recall, intersection = compute_recall( gt, inds )

        # Store and commit the results
        new_row = {   
            "type": "search",
            "allocationid": Allocation_id, \
            "datasetid": dataset_id, \
            "dataset_path": "", \
            "queries_path": queries_path, \
            "bits" : "", \
            "ts_start": ts_start, \
            "ts_train_start": "", \
            "ts_train_end": "", \
            "ts_train_walltime": "", \
            "ts_query_start": ts_query_start, \
            "ts_query_end": ts_query_end, \
            "ts_query_walltime": ts_query_end - ts_query_start, \
            "response": response.search, \
            "recall": recall }
        all_results.append( new_row )

    # Write the results
    benchmark_df = pd.DataFrame( all_results )
    benchmark_df.to_csv( args.output, index=False)
    print("Committed benchmark result to %s" % args.output)

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
    parser.add_argument('-o','--output', required=True)
    parser.add_argument('-d','--dataset', required=True)
    parser.add_argument('-q','--queries', required=True)
    parser.add_argument('-g','--groundtruth', required=True)
    parser.add_argument('-b','--bits', required=False, default=768, type=int)
    parser.add_argument('-n','--neighbors', required=False, default=10, type=int)
    parser.add_argument('-s','--save', required=False, default=False, action='store_true')
    #parser.add_argument('-n','--tries', required=False, default=5)
    parser.add_argument('-u', '--unload', required=False, default=True, action='store_false')
    parser.add_argument('-w', '--wipe', required=False, default=False, action='store_true')
    parser.add_argument('-z', '--qbq', required=False, default=False, action='store_true') # query-by-query instead of batch
    args = parser.parse_args()

    if not os.path.exists( args.dataset ):
        raise Exception("The datset path does not exist->", args.dataset)

    if not os.path.exists( args.queries ):
        raise Exception("The queries path does not exist->", args.queries)
    
    if not os.path.exists( args.groundtruth ):
        raise Exception("The groundtruth path does not exist->", args.groundtruth)

    return args

if __name__ == "__main__":

    args = init_args()
    unload_datasets(args) 
    run_benchmark(args)

    print("Done.")
