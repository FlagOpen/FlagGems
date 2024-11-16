from .register import Register

register_instance = None

def to_register(*args, **kargs):
    global register_instance
    if not register_instance:
        register_instance = Register(*args, **kargs)
    return register_instance

__all__ = ["to_register", "register_instance"]
