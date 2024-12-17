def not_overridden(func):
    func._not_override = True
    return func

def is_overridden(func):
    return not getattr(func, '_not_override', False)