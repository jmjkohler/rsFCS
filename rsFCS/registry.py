class Registry:
    def __init__(self):
        self._objects = {}

    def register_object(self, name, object):
        self._objects[name] = object

    def get_object(self, name):
        returned_object = self._objects.get(name)
        if not returned_object:
            raise ValueError('No item with that name')
        return returned_object

    def info(self):
        return self._objects

    def __str__(self):
        desc = ["{} : {}\n".format(key, value) for key, value in self._objects.items()]
        return "\n".join(desc)
