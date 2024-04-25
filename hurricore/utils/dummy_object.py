class DummyObject:
    def __getattr__(self, name: str):
        def do_nothing(*args, **kwargs):
            pass 
        
        return do_nothing
