from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from vectordb_bench.backend.filter import non_filter
from vectordb_bench.backend.payload import PayloadProfile
from vectordb_bench.backend.runner.mp_runner import MultiProcessingSearchRunner
from vectordb_bench.backend.runner.serial_runner import SerialSearchRunner
from vectordb_bench.backend.task_runner import CaseRunner
from vectordb_bench.models import TaskStage


class QueryProbeDB:
    name = "query-probe"
    supports_batch_search = True

    def __init__(self):
        self.queries = []

    def supports_payload_profile(self, profile):
        return True

    @contextmanager
    def init(self):
        yield

    def prepare_filter(self, filters):
        pass

    def search_embedding(self, query, k):
        assert isinstance(query, list)
        assert all(isinstance(value, float) for value in query)
        self.queries.append(query)
        return [int(np.argmax(query))]


def _spawned_query_summary(runner, queries):
    # Exercise the same bound-method/argument serialization as the search pool.
    return (
        runner.test_data.dtype.name,
        queries.shape,
        queries.nbytes,
        runner.test_data is queries,
        runner._query_at(queries, len(queries) - 1),
    )


@pytest.mark.parametrize("nq", [1, 2, 5])
def test_array_queries_convert_only_current_request_and_wrap(nq):
    queries = np.arange(12, dtype=np.float32).reshape(3, 4)
    runner = MultiProcessingSearchRunner(QueryProbeDB(), queries, nq=nq)

    assert runner.test_data is queries
    query = runner._query_at(runner.test_data, 2)
    expected = queries[2].tolist() if nq == 1 else [queries[(2 + i) % 3].tolist() for i in range(nq)]
    assert query == expected


def test_serial_array_queries_keep_recall_and_packed_storage():
    queries = np.eye(3, dtype=np.float32)
    db = QueryProbeDB()
    runner = SerialSearchRunner(db, queries, ground_truth=[[0], [1], [2]], k=1)

    assert runner.test_data is queries
    result = runner.search((runner.test_data, runner.ground_truth))

    assert result[0] == 1.0
    assert result[1] == 1.0
    assert db.queries == queries.tolist()


def test_spawn_keeps_query_array_float32():
    queries = np.ones((1000, 4096), dtype=np.float32)
    runner = MultiProcessingSearchRunner(QueryProbeDB(), queries)
    with ProcessPoolExecutor(max_workers=1, mp_context=runner.get_mp_context()) as pool:
        dtype, shape, nbytes, same_object, last_query = pool.submit(
            _spawned_query_summary, runner, runner.test_data
        ).result(timeout=30)

    assert dtype == "float32"
    assert shape == (1000, 4096)
    assert nbytes == 1000 * 4096 * 4
    assert same_object
    assert last_query == queries[-1].tolist()


@pytest.mark.parametrize("normalize", [False, True])
def test_case_runner_keeps_packed_queries_for_both_search_stages(monkeypatch, normalize):
    queries = np.array([[3.0, 4.0], [4.0, 3.0]], dtype=np.float32)
    case = SimpleNamespace(
        dataset=SimpleNamespace(test_data=queries, gt_data=[[0], [1]]),
        is_multitenant=False,
        filters=non_filter,
        payload_profile=PayloadProfile.IDS_ONLY,
    )
    config = SimpleNamespace(
        stages=[TaskStage.SEARCH_SERIAL, TaskStage.SEARCH_CONCURRENT],
        case_config=SimpleNamespace(
            k=1,
            nq=1,
            concurrency_search_config=SimpleNamespace(
                num_concurrency=[1], concurrency_duration=1, concurrency_timeout=30
            ),
        ),
    )
    monkeypatch.setattr(CaseRunner, "normalize", property(lambda self: normalize))
    runner = CaseRunner.model_construct(ca=case, config=config, db=QueryProbeDB())
    runner._init_search_runner()

    for search_runner in (runner.serial_search_runner, runner.search_runner):
        assert search_runner.test_data.dtype == np.float32
        assert search_runner.test_data is runner.test_emb
        np.testing.assert_allclose(search_runner.test_data, queries / 5 if normalize else queries)
