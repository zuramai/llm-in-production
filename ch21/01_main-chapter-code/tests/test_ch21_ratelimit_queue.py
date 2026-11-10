import sys
import threading
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch21_ratelimit import RedisTokenBucket, TokenBucket

REDIS_URL = "redis://localhost:6386/0"


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self) -> float:
        return self.t


def test_bucket_burst_then_refill():
    clock = Clock()
    b = TokenBucket(rate=2, capacity=4, clock=clock)
    assert [b.acquire()[0] for _ in range(5)] == [True] * 4 + [False]
    allowed, wait = b.acquire()
    assert not allowed and wait == pytest.approx(0.5)  # 1 token at 2 tokens/s
    clock.t += 0.5
    assert b.acquire() == (True, 0.0)
    clock.t += 1000  # long idle: refills only up to capacity
    assert [b.acquire()[0] for _ in range(5)] == [True] * 4 + [False]


def test_bucket_cost_weights_and_validation():
    clock = Clock()
    b = TokenBucket(rate=1000, capacity=3000, clock=clock)  # e.g. a tokens-per-second budget
    assert b.acquire(2500)[0]
    allowed, wait = b.acquire(1000)
    assert not allowed and wait == pytest.approx(0.5)
    with pytest.raises(ValueError):
        b.acquire(5000)  # could never succeed: reject loudly instead of waiting forever
    with pytest.raises(ValueError):
        TokenBucket(rate=0, capacity=1)


@pytest.fixture
def redis_client():
    redis = pytest.importorskip("redis")
    client = redis.Redis.from_url(REDIS_URL)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis on :6386 not running (docker start ch21-redis)")
    yield client
    client.close()


@pytest.mark.docker
def test_redis_bucket_is_shared_and_atomic(redis_client):
    """Many threads, two limiter objects (think: two servers), one key: exactly `capacity` pass."""
    key = f"test-{uuid.uuid4().hex}"
    a = RedisTokenBucket(redis_client, rate=0.001, capacity=50)
    b = RedisTokenBucket(redis_client, rate=0.001, capacity=50)
    results: list[bool] = []
    lock = threading.Lock()

    def hammer(limiter: RedisTokenBucket) -> None:
        for _ in range(20):
            ok, _ = limiter.acquire(key)
            with lock:
                results.append(ok)

    threads = [threading.Thread(target=hammer, args=(a if i % 2 else b,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 200 and sum(results) == 50
    ok, wait = a.acquire(key)
    assert not ok and wait > 900  # 1 token at 0.001 tokens/s
    assert b.acquire(f"other-{key}")[0]  # keys are independent


@pytest.fixture
def db():
    import ch21_jobqueue as q

    from llmprod import connect

    try:
        conn = connect()
    except Exception as exc:  # noqa: BLE001 - environment without the DB
        pytest.skip(f"ShopFlow DB not reachable: {exc}")
    q.setup(conn, reset=True)
    yield conn
    q.setup(conn, reset=True)
    conn.close()


@pytest.mark.docker
@pytest.mark.parametrize("variant", ["lease", "in_transaction"])
def test_queue_concurrent_workers_run_each_job_once(db, variant):
    import time

    import ch21_jobqueue as q

    from llmprod import connect

    ids = q.enqueue(db, "echo", [{"i": i} for i in range(60)])
    assert len(ids) == 60
    results: dict[str, dict] = {}
    run = q.run_worker if variant == "lease" else q.run_worker_in_transaction

    def handler(job: q.Job) -> dict:
        time.sleep(0.005)  # long enough for workers to overlap
        return {"echo": job.payload["i"]}

    def work(name: str) -> None:
        results[name] = run(connect, name, handler)

    threads = [threading.Thread(target=work, args=(f"w{i}",)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r["done"] for r in results.values()) == 60
    runs = db.execute("SELECT job_id, count(*) FROM ch21.job_runs GROUP BY 1 "
                      "HAVING count(*) > 1").fetchall()
    assert runs == []
    status = dict(db.execute("SELECT status, count(*) FROM ch21.jobs GROUP BY 1").fetchall())
    assert status == {"done": 60}


@pytest.mark.docker
def test_queue_expired_lease_is_reclaimed_and_old_owner_is_fenced(db):
    import ch21_jobqueue as q

    q.enqueue(db, "echo", [{"i": 1}])
    [job] = q.claim(db, "crashy", lease_s=0.05)
    assert q.claim(db, "other") == []  # leased: invisible to others
    db.execute("SELECT pg_sleep(0.1)")
    [again] = q.claim(db, "rescuer")
    assert again.id == job.id and again.attempts == 2
    assert not q.complete(db, job, "crashy", {"late": True})  # stale owner can't overwrite
    assert q.complete(db, again, "rescuer", {"ok": True})


@pytest.mark.docker
def test_queue_failures_back_off_then_give_up(db):
    import ch21_jobqueue as q

    q.enqueue(db, "echo", [{"i": 1}], max_attempts=2)
    [job] = q.claim(db, "w")
    assert q.fail(db, job, "w", "boom", base_delay_s=0) == "queued"
    [job] = q.claim(db, "w")
    assert job.attempts == 2
    assert q.fail(db, job, "w", "boom again", base_delay_s=0) == "failed"
    assert q.claim(db, "w") == []
