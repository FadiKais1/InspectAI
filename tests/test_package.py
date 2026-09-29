import inspectai


def test_package_version() -> None:
    assert inspectai.__version__ == "0.1.0"
