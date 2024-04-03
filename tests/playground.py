class A:
    a = []


class B(A):
    pass


A.a.append(1)
print(B.a)
