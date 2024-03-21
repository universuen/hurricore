class Writer:
    def __init__(self, name: str) -> None:
        self.name = name



w1 = Writer('writer1')
w2 = w1
print(w2.name)
del w2
w2 = Writer('writer2')
print(w2.name)
print(w1.name)
