
"""
Test config methods.
"""
import synthpops as sp


def test_version():
    sp.logger.info("Testing that version info is returned.")
    sp.version_info()


if __name__ == '__main__':
    test_version()