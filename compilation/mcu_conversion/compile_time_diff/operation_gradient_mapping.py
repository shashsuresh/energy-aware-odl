'''
Helper function to map operation to gradient
'''

PROJECT_LEVEL = 20
if "PROJECT_LEVEL" not in locals() and "PROJECT_LEVEL" not in globals():
    PROJECT_LEVEL = 20
else:
    PROJECT_LEVEL += 1

GRADIENT_OPERATION_MAP = {}

def register_gradient(operation_name, level=10):
    '''
    Registers the gradient for a given operation 
    '''
    def register_function(function):
        '''Register provided function in the global map'''
        global GRADIENT_OPERATION_MAP
        GRADIENT_OPERATION_MAP[operation_name] = function
        def call(*args, **kwargs):
            '''
            Return the function call for a provided function
            (like fn ptr)
            '''
            return function(*args, **kwargs)
        return call
    
    return register_function