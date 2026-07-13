import chrono24


def test_package_imports():
    assert chrono24.__version__


def test_load_watches_is_exposed():
    from chrono24 import load_watches

    assert callable(load_watches)
