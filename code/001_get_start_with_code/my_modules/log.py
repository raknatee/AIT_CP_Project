log_global_config ={
    "do_print":True,'n':30
}

class Limit:
    def __init__(self,n=10) -> None:
        self.n=n
        self.counter = 0
    def __call__(self, *args, **kwargs):
        if(self.counter<self.n):
            log(*args,**kwargs)
            self.counter+=1
        
class CounterBreak:
    c = None

    @classmethod
    def set_count(cls,n):
        if(cls.c is None):
            cls.c = 0
            cls.n = n 
        return cls
    
    @classmethod
    def count(cls):
        if(cls.c > cls.n):
            cls.c = None
            return True
        else:
            cls.c+=1
            return False
class CounterMod:
    i = None
    @classmethod
    def init(cls):
        if(cls.i is None):
            cls.i = 1
        return cls
    @classmethod
    def set_mod(cls,n):
        cls.n = n
        return cls
    @classmethod
    def count(cls):
        if(cls.i % cls.n ==0):
            cls.i= None
            return True
        else:
            cls.i+=1
            return False


def log(*args,**kwargs):
    


    default_args = {
        'do_print':log_global_config['do_print'],'title':"",'n':log_global_config['n']
    }
    

    for k in default_args:
        if(k in kwargs):
            default_args[k] = kwargs[k]
            del kwargs[k]
    
    if(default_args['do_print']):
        if(default_args['title'] != ""):
            print("="*default_args['n'],default_args['title'],"="*default_args['n'],sep="")
        if(len(args)>0):
            print(*args,**kwargs)