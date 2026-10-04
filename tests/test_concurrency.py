from concurrent.futures import ThreadPoolExecutor


def test_many_simultaneous_requests_are_all_served_correctly(client):
    """The browser fires several API calls at once; they must not corrupt the shared Teradata connection."""
    top = client.get("/api/alerts", params={"limit": 12}).json()
    ids = [a["txn_id"] for a in top]

    def detail(txn_id):
        r = client.get(f"/api/alerts/{txn_id}")
        return r.status_code, r.json().get("txn", {}).get("txn_id")

    def mixed(i):
        # a mix of cheap cached reads and fresh database reads, as a busy dashboard would send
        if i % 3 == 0:
            return client.get("/api/kpis").status_code, None
        return detail(ids[i % len(ids)])

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(mixed, range(24)))
    assert all(status == 200 for status, _ in results), results
    for i, (_, got) in enumerate(results):
        if i % 3 != 0:
            assert got == ids[i % len(ids)], "each response must belong to its own request"
