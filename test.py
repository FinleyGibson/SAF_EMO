class Class1:
    def __init__(self, a):
        self.a = a
    
    def increment(func):
        self.a+=1
        return func

    @increment
    def print_a(self):
        print(self.a)

x = Class1(10)
x.print_a()
