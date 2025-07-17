import os


class set_env_var:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self._old_value = None
        self._had_old = False

    def __enter__(self):
        self._had_old = self.key in os.environ
        if self._had_old:
            self._old_value = os.environ[self.key]
        os.environ[self.key] = self.value

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._had_old:
            os.environ[self.key] = self._old_value
        else:
            del os.environ[self.key]
