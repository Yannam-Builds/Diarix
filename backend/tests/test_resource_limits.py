from backend.services.resource_limits import allowed_cpu_count


def test_runtime_guard_uses_eighty_percent_of_logical_processors() -> None:
    assert allowed_cpu_count(20, 80) == 16
    assert allowed_cpu_count(8, 80) == 7


def test_runtime_guard_never_selects_zero_or_more_than_available() -> None:
    assert allowed_cpu_count(1, 10) == 1
    assert allowed_cpu_count(4, 100) == 4
