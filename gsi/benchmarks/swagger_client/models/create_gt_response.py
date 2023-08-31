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

class CreateGTResponse(object):
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
        'existing_query_size': 'str',
        'random_queries_ind': 'str',
        'indices_file': 'str',
        'distances_file': 'str'
    }

    attribute_map = {
        'existing_query_size': 'existingQuerySize',
        'random_queries_ind': 'randomQueriesInd',
        'indices_file': 'indicesFile',
        'distances_file': 'distancesFile'
    }

    def __init__(self, existing_query_size=None, random_queries_ind=None, indices_file=None, distances_file=None):  # noqa: E501
        """CreateGTResponse - a model defined in Swagger"""  # noqa: E501
        self._existing_query_size = None
        self._random_queries_ind = None
        self._indices_file = None
        self._distances_file = None
        self.discriminator = None
        if existing_query_size is not None:
            self.existing_query_size = existing_query_size
        if random_queries_ind is not None:
            self.random_queries_ind = random_queries_ind
        if indices_file is not None:
            self.indices_file = indices_file
        if distances_file is not None:
            self.distances_file = distances_file

    @property
    def existing_query_size(self):
        """Gets the existing_query_size of this CreateGTResponse.  # noqa: E501


        :return: The existing_query_size of this CreateGTResponse.  # noqa: E501
        :rtype: str
        """
        return self._existing_query_size

    @existing_query_size.setter
    def existing_query_size(self, existing_query_size):
        """Sets the existing_query_size of this CreateGTResponse.


        :param existing_query_size: The existing_query_size of this CreateGTResponse.  # noqa: E501
        :type: str
        """

        self._existing_query_size = existing_query_size

    @property
    def random_queries_ind(self):
        """Gets the random_queries_ind of this CreateGTResponse.  # noqa: E501


        :return: The random_queries_ind of this CreateGTResponse.  # noqa: E501
        :rtype: str
        """
        return self._random_queries_ind

    @random_queries_ind.setter
    def random_queries_ind(self, random_queries_ind):
        """Sets the random_queries_ind of this CreateGTResponse.


        :param random_queries_ind: The random_queries_ind of this CreateGTResponse.  # noqa: E501
        :type: str
        """

        self._random_queries_ind = random_queries_ind

    @property
    def indices_file(self):
        """Gets the indices_file of this CreateGTResponse.  # noqa: E501


        :return: The indices_file of this CreateGTResponse.  # noqa: E501
        :rtype: str
        """
        return self._indices_file

    @indices_file.setter
    def indices_file(self, indices_file):
        """Sets the indices_file of this CreateGTResponse.


        :param indices_file: The indices_file of this CreateGTResponse.  # noqa: E501
        :type: str
        """

        self._indices_file = indices_file

    @property
    def distances_file(self):
        """Gets the distances_file of this CreateGTResponse.  # noqa: E501


        :return: The distances_file of this CreateGTResponse.  # noqa: E501
        :rtype: str
        """
        return self._distances_file

    @distances_file.setter
    def distances_file(self, distances_file):
        """Sets the distances_file of this CreateGTResponse.


        :param distances_file: The distances_file of this CreateGTResponse.  # noqa: E501
        :type: str
        """

        self._distances_file = distances_file

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
        if issubclass(CreateGTResponse, dict):
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
        if not isinstance(other, CreateGTResponse):
            return False

        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        return not self == other
