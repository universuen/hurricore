class AttributeDict(dict):
    """
    Savable dictionary class that allows dot notation access to keys.
    """
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        return self.get(key, None)

    def state_dict(self):
        return dict(self)

    def load_state_dict(self, state_dict):
        self.update(state_dict)

    def __reduce__(self):
        return (AttributeDict, (dict(self),))
