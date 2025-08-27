from hurricore.utils import Environment, Dummy


def test_environment():
    env_a = Environment()
    env_a.num = 1
    assert type(env_a.char) is Dummy
    env_b = Environment()
    assert env_b.num == 1


if __name__ == '__main__':
    test_environment()
