import inspectai


def test_package_version() -> None:
    assert inspectai.__version__ == "1.0.0"
