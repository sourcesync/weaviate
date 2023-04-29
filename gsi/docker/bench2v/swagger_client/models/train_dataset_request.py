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

class TrainDatasetRequest(object):
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
        'dataset_id': 'str',
        'grid_train': 'bool',
        'nbits': 'int',
        'qbits': 'int',
        'num_of_boards': 'int',
        'num_of_clusters': 'int'
    }

    attribute_map = {
        'dataset_id': 'datasetId',
        'grid_train': 'gridTrain',
        'nbits': 'nbits',
        'qbits': 'qbits',
        'num_of_boards': 'numOfBoards',
        'num_of_clusters': 'numOfClusters'
    }

    def __init__(self, dataset_id=None, grid_train=False, nbits=768, qbits=768, num_of_boards=None, num_of_clusters=None):  # noqa: E501
        """TrainDatasetRequest - a model defined in Swagger"""  # noqa: E501
        self._dataset_id = None
        self._grid_train = None
        self._nbits = None
        self._qbits = None
        self._num_of_boards = None
        self._num_of_clusters = None
        self.discriminator = None
        self.dataset_id = dataset_id
        if grid_train is not None:
            self.grid_train = grid_train
        if nbits is not None:
            self.nbits = nbits
        if qbits is not None:
            self.qbits = qbits
        if num_of_boards is not None:
            self.num_of_boards = num_of_boards
        if num_of_clusters is not None:
            self.num_of_clusters = num_of_clusters

    @property
    def dataset_id(self):
        """Gets the dataset_id of this TrainDatasetRequest.  # noqa: E501

        The datasetId identifies the specific dataset to search. It is generated using the /import/dataset endpoint.  # noqa: E501

        :return: The dataset_id of this TrainDatasetRequest.  # noqa: E501
        :rtype: str
        """
        return self._dataset_id

    @dataset_id.setter
    def dataset_id(self, dataset_id):
        """Sets the dataset_id of this TrainDatasetRequest.

        The datasetId identifies the specific dataset to search. It is generated using the /import/dataset endpoint.  # noqa: E501

        :param dataset_id: The dataset_id of this TrainDatasetRequest.  # noqa: E501
        :type: str
        """
        if dataset_id is None:
            raise ValueError("Invalid value for `dataset_id`, must not be `None`")  # noqa: E501

        self._dataset_id = dataset_id

    @property
    def grid_train(self):
        """Gets the grid_train of this TrainDatasetRequest.  # noqa: E501

        Flag that indicates whether the train should be optimized. Grid train is taking longer time than default train.  # noqa: E501

        :return: The grid_train of this TrainDatasetRequest.  # noqa: E501
        :rtype: bool
        """
        return self._grid_train

    @grid_train.setter
    def grid_train(self, grid_train):
        """Sets the grid_train of this TrainDatasetRequest.

        Flag that indicates whether the train should be optimized. Grid train is taking longer time than default train.  # noqa: E501

        :param grid_train: The grid_train of this TrainDatasetRequest.  # noqa: E501
        :type: bool
        """

        self._grid_train = grid_train

    @property
    def nbits(self):
        """Gets the nbits of this TrainDatasetRequest.  # noqa: E501

        If dataset is float, nbits is the value to convert the float dataset into  # noqa: E501

        :return: The nbits of this TrainDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._nbits

    @nbits.setter
    def nbits(self, nbits):
        """Sets the nbits of this TrainDatasetRequest.

        If dataset is float, nbits is the value to convert the float dataset into  # noqa: E501

        :param nbits: The nbits of this TrainDatasetRequest.  # noqa: E501
        :type: int
        """

        self._nbits = nbits

    @property
    def qbits(self):
        """Gets the qbits of this TrainDatasetRequest.  # noqa: E501

        If dataset is float and search type is clusters, qbits is the value to convert the float centroids into  # noqa: E501

        :return: The qbits of this TrainDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._qbits

    @qbits.setter
    def qbits(self, qbits):
        """Sets the qbits of this TrainDatasetRequest.

        If dataset is float and search type is clusters, qbits is the value to convert the float centroids into  # noqa: E501

        :param qbits: The qbits of this TrainDatasetRequest.  # noqa: E501
        :type: int
        """

        self._qbits = qbits

    @property
    def num_of_boards(self):
        """Gets the num_of_boards of this TrainDatasetRequest.  # noqa: E501

        Used to for cluster search only. numOfBoards will define the num of clusters to create.  # noqa: E501

        :return: The num_of_boards of this TrainDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._num_of_boards

    @num_of_boards.setter
    def num_of_boards(self, num_of_boards):
        """Sets the num_of_boards of this TrainDatasetRequest.

        Used to for cluster search only. numOfBoards will define the num of clusters to create.  # noqa: E501

        :param num_of_boards: The num_of_boards of this TrainDatasetRequest.  # noqa: E501
        :type: int
        """

        self._num_of_boards = num_of_boards

    @property
    def num_of_clusters(self):
        """Gets the num_of_clusters of this TrainDatasetRequest.  # noqa: E501

        If numOfBoards is empty, numOfClusters will be used to define the num of clusters to create.  # noqa: E501

        :return: The num_of_clusters of this TrainDatasetRequest.  # noqa: E501
        :rtype: int
        """
        return self._num_of_clusters

    @num_of_clusters.setter
    def num_of_clusters(self, num_of_clusters):
        """Sets the num_of_clusters of this TrainDatasetRequest.

        If numOfBoards is empty, numOfClusters will be used to define the num of clusters to create.  # noqa: E501

        :param num_of_clusters: The num_of_clusters of this TrainDatasetRequest.  # noqa: E501
        :type: int
        """

        self._num_of_clusters = num_of_clusters

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
        if issubclass(TrainDatasetRequest, dict):
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
        if not isinstance(other, TrainDatasetRequest):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
