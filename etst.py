class Vin:
    def __init__(self, y):
        self.x = y


ob = Vin(3)
print(ob.x)

l = []
l.append(ob)
ob = Vin(9)
l.append(Vin(7))

print(l[0].x)
print(l[1].x)
print(ob.x)
