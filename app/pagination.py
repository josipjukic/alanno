from rest_framework import pagination
from rest_framework.response import Response
from collections import OrderedDict
import math


class InfoPagination(pagination.LimitOffsetPagination):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)
        self.allow_empty_first_page = True

    def get_paginated_response(self, data):
        num_pages = math.ceil(self.count / self.limit)
        current_page = int(self.offset / self.limit)

        return Response(
            OrderedDict(
                [
                    ("count", self.count),
                    ("next", self.get_next_link()),
                    ("previous", self.get_previous_link()),
                    ("start_index", self.offset),
                    ("num_pages", num_pages),
                    ("current_page", current_page),
                    ("results", data),
                ]
            )
        )
