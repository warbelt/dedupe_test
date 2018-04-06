import csv
import dedupe
import time
import cProfile
from collections import OrderedDict

from pyspark import SparkContext as sc

#def blocker_call_spark(self, records, target=False):
def blocker_call_spark(p, records, target=False):

    predicates = [(':' + str(i), predicate)
                  for i, predicate
#                 in enumerate(self.predicates)]
                  in enumerate(p)]

    records_rdd = sc.parallelize(records)

    blocked_records_rdd = records_rdd.map()

    blocked_records_rdd.collect()


def create_record_blocks(record, predicates):

    record_id, instance = record
    record_block = []

    for pred_id, predicate in predicates:
        block_keys = predicate(instance, target=target)
        for block_key in block_keys:
            record_block(append) block_key + pred_id, record_id