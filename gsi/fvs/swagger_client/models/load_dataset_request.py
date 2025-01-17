# coding: utf-8

"""
    GSI Floating-Point 32 API

    **Introduction**<br> GSI Technology’s floating-point similarity search API provides an accessible gateway to running searches on GSI’s Gemini® Associative Processing Unit (APU).<br> It works in conjunction with the GSI system management solution which enables users to work with multiple APU boards simultaneously for improved performance.<br><br> **Dataset and Query Format**<br> Dataset embeddings can be in 32- or 64-bit floating point format, and any number of features, e.g. 256 or 512 (there is no upper limit).<br> Query embeddings must have the same floating-point format and number of features as used in the dataset.<br> GSI performs the search and delivers the top-k most similar results.  # noqa: E501

    OpenAPI spec version: 1.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""

import pprint
import re  # noqa: F401

import six

class LoadDatasetRequest(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """
    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        'allocation_id': 'str',
        'dataset_id': 'str',
        'typical_n_queries': 'int',
        'max_n_queries': 'int',
        'normalize': 'bool',
        'centroids_hamming_k': 'int',
        'centroids_rerank': 'int',
        'hamming_k': 'int',
        'topk': 'int',
        'neural_matrix_id': 'str',
        'algorithm': 'str',
        'bitmasks_ind': 'bool',
        'async_load': 'bool'
    }

    attribute_map = {
        'allocation_id': 'allocationId',
        'dataset_id': 'datasetId',
        'typical_n_queries': 'typicalNQueries',
        'max_n_queries': 'maxNQueries',
        'normalize': 'normalize',
        'centroids_hamming_k': 'centroidsHammingK',
        'centroids_rerank': 'centroidsRerank',
        'hamming_k': 'hammingK',
        'topk': 'topk',
        'neural_matrix_id': 'neuralMatrixId',
        'algorithm': 'algorithm',
        'bitmasks_ind': 'bitmasksInd',
        'async_load': 'asyncLoad'
    }

    def __init__(self, allocation_id=None, dataset_id=None, typical_n_queries=10, max_n_queries=3100, normalize=False, centroids_hamming_k=5000, centroids_rerank=4000, hamming_k=3200, topk=1000, neural_matrix_id=None, algorithm=None, bitmasks_ind=False, async_load=False):  # noqa: E501
        """LoadDatasetRequest - a model defined in Swagger"""  # noqa: E501
        self._allocation_id = None
        self._dataset_id = None
        self._typical_n_queries = None
        self._max_n_queries = None
        self._normalize = None
        self._centroids_hamming_k = None
        self._centroids_rerank = None
        self._hamming_k = None
        self._topk = None
        self._neural_matrix_id = None
        self._algorithm = None
        self._bitmasks_ind = None
        self._async_load = None
        self.discriminator = None
        self.allocation_id = allocation_id
        self.dataset_id = dataset_id
        if typical_n_queries is not None:
            self.typical_n_queries = typical_n_queries
        if max_n_queries is not None:
            self.max_n_queries = max_n_queries
        if normalize is not None:
            self.normalize = normalize
        if centroids_hamming_k is not None:
            self.centroids_hamming_k = centroids_hamming_k
        if centroids_rerank is not None:
            self.centroids_rerank = centroids_rerank
        if hamming_k is not None:
            self.hamming_k = hamming_k
        if topk is not None:
            self.topk = topk
        if neural_matrix_id is not None:
            self.neural_matrix_id = neural_matrix_id
        if algorithm is not None:
            self.algorithm = algorithm
        if bitmasks_ind is not None:
            self.bitmasks_ind = bitmasks_ind
        if async_load is not None:
            self.async_load = async_load

    @property
    def allocation_id(self):
        """Gets the allocation_id of this LoadDatasetRequest.  # noqa: E501

        The UID representing an allocation of a specific number of APU boards. It is generated using the /allocate endpoint.  # noqa: E501

        :return: The allocation_id of this LoadDatasetRequest.  # noqa: E501
        :rtype: str
        """
        return self._allocation_id

    @allocation_id.setter
    def allocation_id(self, allocation_id):
        """Sets the allocation_id of this LoadDatasetRequest.

        The UID representing an allocation of a specific number of APU boards. It is generated using the /allocate endpoint.  # noqa: E501

        :param allocation_id: The allocation_id of this LoadDatasetRequest.  # noqa: E501
        :type: str
        """
        if allocation_id is None:
            raise ValueError("Invalid value for `allocation_id`, must not be `None`")  # noqa: E501

        self._allocation_id = allocation_id

    @property
    def dataset_id(self):
        """Gets the dataset_id of this LoadDatasetRequest.  # noqa: E501

        Dataset UID identifies specific dataset to load. It is generated using the /import/dataset endpoint.  # noqa: E501

        :return: The dataset_id of this LoadDatasetRequest.  # noqa: E501
        :rtype: str
        """
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id):
        """Sets the dataset_id of this LoadDatasetRequest.

        Dataset UID identifies specific dataset to load. It is generated using the /import/dataset endpoint.  # noqa: E501

        :param dataset_id: The dataset_id of this LoadDatasetRequest.  # noqa: E501
        :type: str
        """
        if dataset_id is None:
            raise ValueError("Invalid value for `dataset_id`, must not be `None`")  # noqa: E501

        self._dataset_id = dataset_id

    @property
    def typical_n_queries(self):
        """Gets the typical_n_queries of this LoadDatasetRequest.  # noqa: E501

        Typical number of queries in a search (not more than maxNQueries).  # noqa: E501

        :return: The typical_n_queries of this LoadDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._typical_n_queries

    @typical_n_queries.setter
    def typical_n_queries(self, typical_n_queries):
        """Sets the typical_n_queries of this LoadDatasetRequest.

        Typical number of queries in a search (not more than maxNQueries).  # noqa: E501

        :param typical_n_queries: The typical_n_queries of this LoadDatasetRequest.  # noqa: E501
        :type: int
        """

        self._typical_n_queries = typical_n_queries

    @property
    def max_n_queries(self):
        """Gets the max_n_queries of this LoadDatasetRequest.  # noqa: E501

        Maximum number of queries in a search (lower or equal to typicalNQueries).  # noqa: E501

        :return: The max_n_queries of this LoadDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._max_n_queries

    @max_n_queries.setter
    def max_n_queries(self, max_n_queries):
        """Sets the max_n_queries of this LoadDatasetRequest.

        Maximum number of queries in a search (lower or equal to typicalNQueries).  # noqa: E501

        :param max_n_queries: The max_n_queries of this LoadDatasetRequest.  # noqa: E501
        :type: int
        """

        self._max_n_queries = max_n_queries

    @property
    def normalize(self):
        """Gets the normalize of this LoadDatasetRequest.  # noqa: E501

        Indicates whether a dataset should be normalized (of values between 0 to 1).  # noqa: E501

        :return: The normalize of this LoadDatasetRequest.  # noqa: E501
        :rtype: bool
        """
        return self._normalize

    @normalize.setter
    def normalize(self, normalize):
        """Sets the normalize of this LoadDatasetRequest.

        Indicates whether a dataset should be normalized (of values between 0 to 1).  # noqa: E501

        :param normalize: The normalize of this LoadDatasetRequest.  # noqa: E501
        :type: bool
        """

        self._normalize = normalize

    @property
    def centroids_hamming_k(self):
        """Gets the centroids_hamming_k of this LoadDatasetRequest.  # noqa: E501

        Number of results from APU HAMMING search function over centroids (per dataset or system-default).  # noqa: E501

        :return: The centroids_hamming_k of this LoadDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._centroids_hamming_k

    @centroids_hamming_k.setter
    def centroids_hamming_k(self, centroids_hamming_k):
        """Sets the centroids_hamming_k of this LoadDatasetRequest.

        Number of results from APU HAMMING search function over centroids (per dataset or system-default).  # noqa: E501

        :param centroids_hamming_k: The centroids_hamming_k of this LoadDatasetRequest.  # noqa: E501
        :type: int
        """

        self._centroids_hamming_k = centroids_hamming_k

    @property
    def centroids_rerank(self):
        """Gets the centroids_rerank of this LoadDatasetRequest.  # noqa: E501

        Array size keeping the number of results to re-rank from APU HAMMING search function over centroids.  # noqa: E501

        :return: The centroids_rerank of this LoadDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._centroids_rerank

    @centroids_rerank.setter
    def centroids_rerank(self, centroids_rerank):
        """Sets the centroids_rerank of this LoadDatasetRequest.

        Array size keeping the number of results to re-rank from APU HAMMING search function over centroids.  # noqa: E501

        :param centroids_rerank: The centroids_rerank of this LoadDatasetRequest.  # noqa: E501
        :type: int
        """

        self._centroids_rerank = centroids_rerank

    @property
    def hamming_k(self):
        """Gets the hamming_k of this LoadDatasetRequest.  # noqa: E501

        Number of results from APU HAMMING search function (per dataset or system-default).  # noqa: E501

        :return: The hamming_k of this LoadDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._hamming_k

    @hamming_k.setter
    def hamming_k(self, hamming_k):
        """Sets the hamming_k of this LoadDatasetRequest.

        Number of results from APU HAMMING search function (per dataset or system-default).  # noqa: E501

        :param hamming_k: The hamming_k of this LoadDatasetRequest.  # noqa: E501
        :type: int
        """

        self._hamming_k = hamming_k

    @property
    def topk(self):
        """Gets the topk of this LoadDatasetRequest.  # noqa: E501

        Array size keeping the number of results to re-rank from APU HAMMING search function.<br> value as 0 will send back entire results defined by HAMMING function  # noqa: E501

        :return: The topk of this LoadDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._topk

    @topk.setter
    def topk(self, topk):
        """Sets the topk of this LoadDatasetRequest.

        Array size keeping the number of results to re-rank from APU HAMMING search function.<br> value as 0 will send back entire results defined by HAMMING function  # noqa: E501

        :param topk: The topk of this LoadDatasetRequest.  # noqa: E501
        :type: int
        """

        self._topk = topk

    @property
    def neural_matrix_id(self):
        """Gets the neural_matrix_id of this LoadDatasetRequest.  # noqa: E501

        Neural matrix UID. It is generated using the /import/dataset or /import/neuralMatrix endpoints (using default active neural matrix UID if null passed).  # noqa: E501

        :return: The neural_matrix_id of this LoadDatasetRequest.  # noqa: E501
        :rtype: str
        """
        return self._neural_matrix_id

    @neural_matrix_id.setter
    def neural_matrix_id(self, neural_matrix_id):
        """Sets the neural_matrix_id of this LoadDatasetRequest.

        Neural matrix UID. It is generated using the /import/dataset or /import/neuralMatrix endpoints (using default active neural matrix UID if null passed).  # noqa: E501

        :param neural_matrix_id: The neural_matrix_id of this LoadDatasetRequest.  # noqa: E501
        :type: str
        """

        self._neural_matrix_id = neural_matrix_id

    @property
    def algorithm(self):
        """Gets the algorithm of this LoadDatasetRequest.  # noqa: E501

        Search algorithm to be used if using a binary dataset search.  # noqa: E501

        :return: The algorithm of this LoadDatasetRequest.  # noqa: E501
        :rtype: str
        """
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        """Sets the algorithm of this LoadDatasetRequest.

        Search algorithm to be used if using a binary dataset search.  # noqa: E501

        :param algorithm: The algorithm of this LoadDatasetRequest.  # noqa: E501
        :type: str
        """
        allowed_values = ["hamming", "tanimoto"]  # noqa: E501
        if algorithm not in allowed_values:
            raise ValueError(
                "Invalid value for `algorithm` ({0}), must be one of {1}"  # noqa: E501
                .format(algorithm, allowed_values)
            )

        self._algorithm = algorithm

    @property
    def bitmasks_ind(self):
        """Gets the bitmasks_ind of this LoadDatasetRequest.  # noqa: E501

        Indicates whether the search will be done with bitmasks  # noqa: E501

        :return: The bitmasks_ind of this LoadDatasetRequest.  # noqa: E501
        :rtype: bool
        """
        return self._bitmasks_ind

    @bitmasks_ind.setter
    def bitmasks_ind(self, bitmasks_ind):
        """Sets the bitmasks_ind of this LoadDatasetRequest.

        Indicates whether the search will be done with bitmasks  # noqa: E501

        :param bitmasks_ind: The bitmasks_ind of this LoadDatasetRequest.  # noqa: E501
        :type: bool
        """

        self._bitmasks_ind = bitmasks_ind

    @property
    def async_load(self):
        """Gets the async_load of this LoadDatasetRequest.  # noqa: E501

        Indicates whether the load will be asynchronously or not  # noqa: E501

        :return: The async_load of this LoadDatasetRequest.  # noqa: E501
        :rtype: bool
        """
        return self._async_load

    @async_load.setter
    def async_load(self, async_load):
        """Sets the async_load of this LoadDatasetRequest.

        Indicates whether the load will be asynchronously or not  # noqa: E501

        :param async_load: The async_load of this LoadDatasetRequest.  # noqa: E501
        :type: bool
        """

        self._async_load = async_load

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(map(
                    lambda x: x.to_dict() if hasattr(x, "to_dict") else x,
                    value
                ))
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(map(
                    lambda item: (item[0], item[1].to_dict())
                    if hasattr(item[1], "to_dict") else item,
                    value.items()
                ))
            else:
                result[attr] = value
        if issubclass(LoadDatasetRequest, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, LoadDatasetRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
