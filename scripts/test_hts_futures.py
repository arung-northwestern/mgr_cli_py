#!/usr/bin/env python3
"""Test that Dask wait loop exits when some futures error (status != 'finished').

Run: python scripts/test_hts_futures.py

Expect: Loop exits when done == total (using .done() or status in finished/error),
        then we collect results; errored futures raise when calling .result().
"""
import time

def ok_task(x):
    return ("ok", f"/out/{x}", True, None)

def fail_task(x):
    raise RuntimeError(f"Faulty CIF {x}")

def main():
    try:
        from dask.distributed import Client, LocalCluster
        from dask import delayed
    except ImportError:
        print("Install dask[distributed] to run this test.")
        return

    cluster = LocalCluster(n_workers=2, threads_per_worker=1)
    client = Client(cluster)
    try:
        # 4 ok, 2 fail (like 214 ok + 2 faulty CIFs)
        delayed_list = (
            [delayed(ok_task)(i) for i in range(4)]
            + [delayed(fail_task)(i) for i in (99, 100)]
        )
        futures = client.compute(delayed_list)
        total = len(futures)

        print("Waiting for futures (use .done() so errored count as done)...")
        t0 = time.time()
        while True:
            done = sum(1 for f in futures if (hasattr(f, "done") and f.done()) or (getattr(f, "status", None) in ("finished", "error", "cancelled", "lost")))
            print(f"  Progress: {done}/{total}  (statuses: {[getattr(f, 'status', None) for f in futures]})")
            if done >= total:
                break
            time.sleep(0.5)
        print(f"Wait loop exited in {time.time() - t0:.1f}s")

        results_list = []
        for i, f in enumerate(futures):
            try:
                r = f.result(timeout=10)
                results_list.append(r if isinstance(r, tuple) and len(r) == 4 else ("?", None, False, str(r)))
            except Exception as e:
                results_list.append(("?", None, False, str(e)))

        n_ok = sum(1 for r in results_list if isinstance(r, tuple) and len(r) >= 3 and r[2])
        n_err = total - n_ok
        print(f"Summary: completed={n_ok} errored={n_err} total={total}")
        assert n_ok == 4 and n_err == 2, f"Expected 4 ok, 2 err; got {n_ok} ok, {n_err} err"
        print("OK: test passed.")
    finally:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()
