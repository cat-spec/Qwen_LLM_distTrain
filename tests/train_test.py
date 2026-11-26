class Test():
    def __init__(self,args):
        self.args=args
    def __call__(self):
        print('test')
test=Test('test')

test()