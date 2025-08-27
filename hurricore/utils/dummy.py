class Dummy:
    """
    Dummy to be used as a placeholder for missing attributes or methods.
    """
    def __getattr__(self, name: str):
        def do_nothing(*args, **kwargs):
            return self 
        
        return do_nothing
    
    def __bool__(self):
        return False
