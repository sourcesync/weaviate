# coding: utf-8

# flake8: noqa
"""
    GSI Floating-Point 32 API

    **Introduction**<br> GSI Technology’s floating-point similarity search API provides an accessible gateway to running searches on GSI’s Gemini® Associative Processing Unit (APU).<br> It works in conjunction with the GSI system management solution which enables users to work with multiple APU boards simultaneously for improved performance.<br><br> **Dataset and Query Format**<br> Dataset embeddings can be in 32- or 64-bit floating point format, and any number of features, e.g. 256 or 512 (there is no upper limit).<br> Query embeddings must have the same floating-point format and number of features as used in the dataset.<br> GSI performs the search and delivers the top-k most similar results.  # noqa: E501

    OpenAPI spec version: 1.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

from __future__ import absolute_import

# import models into model package
from swagger_client.models.add_data_request import AddDataRequest
from swagger_client.models.add_data_response import AddDataResponse
from swagger_client.models.add_neural_matrix_request import AddNeuralMatrixRequest
from swagger_client.models.add_neural_matrix_response import AddNeuralMatrixResponse
from swagger_client.models.alive_response import AliveResponse
from swagger_client.models.allocate_request import AllocateRequest
from swagger_client.models.allocate_response import AllocateResponse
from swagger_client.models.bad_validation_response import BadValidationResponse
from swagger_client.models.commit_transactions_request import CommitTransactionsRequest
from swagger_client.models.context_request import ContextRequest
from swagger_client.models.context_response import ContextResponse
from swagger_client.models.convert_txt_to_bin_request import ConvertTxtToBinRequest
from swagger_client.models.convert_txt_to_bin_response import ConvertTxtToBinResponse
from swagger_client.models.create_gt_request import CreateGTRequest
from swagger_client.models.create_gt_response import CreateGTResponse
from swagger_client.models.create_random_queries_file_request import CreateRandomQueriesFileRequest
from swagger_client.models.create_random_queries_file_response import CreateRandomQueriesFileResponse
from swagger_client.models.deallocate_request import DeallocateRequest
from swagger_client.models.focus_dataset_request import FocusDatasetRequest
from swagger_client.models.generate_metadata_request import GenerateMetadataRequest
from swagger_client.models.generate_queries_request import GenerateQueriesRequest
from swagger_client.models.generate_queries_response import GenerateQueriesResponse
from swagger_client.models.get_allocations_list_response import GetAllocationsListResponse
from swagger_client.models.get_datasets_list_response import GetDatasetsListResponse
from swagger_client.models.get_eligible_dataset_clusters_by_num_boards_response import GetEligibleDatasetClustersByNumBoardsResponse
from swagger_client.models.get_files_from_s3_response import GetFilesFromS3Response
from swagger_client.models.get_focused_dataset_response import GetFocusedDatasetResponse
from swagger_client.models.get_neural_matrix_list_response import GetNeuralMatrixListResponse
from swagger_client.models.get_num_of_boards_response import GetNumOfBoardsResponse
from swagger_client.models.get_queries_list_response import GetQueriesListResponse
from swagger_client.models.get_train_status_response import GetTrainStatusResponse
from swagger_client.models.get_transactions_response import GetTransactionsResponse
from swagger_client.models.import_clusters_request import ImportClustersRequest
from swagger_client.models.import_clusters_response import ImportClustersResponse
from swagger_client.models.import_dataset_request import ImportDatasetRequest
from swagger_client.models.import_dataset_response import ImportDatasetResponse
from swagger_client.models.import_metadata_request import ImportMetadataRequest
from swagger_client.models.import_neural_matrix_request import ImportNeuralMatrixRequest
from swagger_client.models.import_neural_matrix_response import ImportNeuralMatrixResponse
from swagger_client.models.import_queries_request import ImportQueriesRequest
from swagger_client.models.import_queries_response import ImportQueriesResponse
from swagger_client.models.is_valid_request import IsValidRequest
from swagger_client.models.is_valid_response import IsValidResponse
from swagger_client.models.load_dataset_request import LoadDatasetRequest
from swagger_client.models.one_of_add_data_request_data_to_add import OneOfAddDataRequestDataToAdd
from swagger_client.models.one_of_add_data_request_metadata_to_add import OneOfAddDataRequestMetadataToAdd
from swagger_client.models.one_of_search_request_queries_file_path import OneOfSearchRequestQueriesFilePath
from swagger_client.models.page_not_found_response import PageNotFoundResponse
from swagger_client.models.predefined_allocation_request import PredefinedAllocationRequest
from swagger_client.models.presigned_upload_request import PresignedUploadRequest
from swagger_client.models.queries_file_response import QueriesFileResponse
from swagger_client.models.reload_data_request import ReloadDataRequest
from swagger_client.models.remove_data_request import RemoveDataRequest
from swagger_client.models.remove_data_response import RemoveDataResponse
from swagger_client.models.remove_meta_data_request import RemoveMetaDataRequest
from swagger_client.models.rollback_transactions_request import RollbackTransactionsRequest
from swagger_client.models.save_results_request import SaveResultsRequest
from swagger_client.models.save_results_response import SaveResultsResponse
from swagger_client.models.search_accuracy_request import SearchAccuracyRequest
from swagger_client.models.search_accuracy_response import SearchAccuracyResponse
from swagger_client.models.search_request import SearchRequest
from swagger_client.models.search_request_prefilters import SearchRequestPrefilters
from swagger_client.models.search_response import SearchResponse
from swagger_client.models.set_clusters_active_request import SetClustersActiveRequest
from swagger_client.models.set_clusters_active_response import SetClustersActiveResponse
from swagger_client.models.set_neural_matrix_active_request import SetNeuralMatrixActiveRequest
from swagger_client.models.set_neural_matrix_active_response import SetNeuralMatrixActiveResponse
from swagger_client.models.status_ok_response import StatusOkResponse
from swagger_client.models.train_dataset_request import TrainDatasetRequest
from swagger_client.models.unload_dataset_request import UnloadDatasetRequest
from swagger_client.models.upload_file_request import UploadFileRequest
from swagger_client.models.upload_file_response import UploadFileResponse
from swagger_client.models.validate_request import ValidateRequest
from swagger_client.models.validate_response import ValidateResponse
