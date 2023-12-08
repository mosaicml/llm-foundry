# This file is a monkey patch on top of the DB Connect package that allows
# the client to fetch the results in different formats from the server. To be
# able to use the code make sure to first import this module before importing
# the DB Connect classes.
from typing import Tuple, List

from pyspark.sql.connect.dataframe import DataFrame
from pyspark.sql.connect.client.core import SparkConnectClient
from pyspark.sql.connect.client.reattach import ExecutePlanResponseReattachableIterator

# PB2 stuff
import pyspark.sql.connect.proto.cloud_pb2 as cloud_pb2
import pyspark.sql.connect.proto as pb2
import google.protobuf.any_pb2 as any_pb2
from collections import namedtuple

Result = namedtuple("Result", ["url", "row_count", "compressed_size", "uncompressed_size"])

# Monkey Patching for SparkConnectClient
def to_cf(self: SparkConnectClient, plan: pb2.Plan, type: str = "json"):
    """
    Executes a given plan object and returns the results as cloud fetch
    presigned URLS. It can handle the current outptu formats that are
    supported by the server.

    In contrast to the regular API methods of the client, this method
    does not return the schema and drops all other responses.
    """
    req = self._execute_plan_request_with_metadata()
    req.plan.CopyFrom(plan)

    # Add the request options
    if type == "json":
        format = cloud_pb2.ResultOptions.CloudOptions.FORMAT_JSON
    elif type == "csv":
        format = cloud_pb2.ResultOptions.CloudOptions.FORMAT_CSV
    elif type == "arrow":
        format = cloud_pb2.ResultOptions.CloudOptions.FORMAT_ARROW
    else:
        raise Exception("Invalid type")

    ro = cloud_pb2.ResultOptions(
        type=cloud_pb2.ResultOptions.TYPE_CLOUD,
        cloudOptions=cloud_pb2.ResultOptions.CloudOptions(
            format=format,
            useCompression=False,
        ))
    cloud_option = any_pb2.Any()
    cloud_option.Pack(ro)
    req.request_options.append(pb2.ExecutePlanRequest.RequestOption(extension=cloud_option))

    # Create the iterator
    iterator = ExecutePlanResponseReattachableIterator(req, self._stub,
                                                       self._retry_policy,
                                                       self._builder.metadata())
    # Iterate over the response

    result = []
    row_count = 0
    is_overflow = False

    for response in iterator:
        if response.HasField("extension") and response.extension.Is(
                cloud_pb2.CloudResultBatch.DESCRIPTOR):
            batch = cloud_pb2.CloudResultBatch()
            assert response.extension.Is(cloud_pb2.CloudResultBatch.DESCRIPTOR)
            response.extension.Unpack(batch)
            result += [Result(b.url, b.row_count, b.compressed_size, b.uncompressed_size) for b in batch.results]
            row_count += sum(result.row_count for result in batch.results)
            is_overflow |= batch.truncated
    return result, row_count, is_overflow


SparkConnectClient.to_cf = to_cf


# Monkey Patching for DataFrame

def collect_as_cf(self: DataFrame, type: str = "json") -> Tuple[List[Result], int, bool]:
    query = self._plan.to_proto(self._session.client)
    results, row_count, is_overflow = self._session.client.to_cf(
        query, type)
    return results, row_count, is_overflow


DataFrame.collect_cf = collect_as_cf
