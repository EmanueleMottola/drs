import struct
import redis
import numpy as np



def toRedis(r, a, n):
    """Store a given Numpy array 'a' in Redis under key 'n'"""
    encoded = a.tobytes()

    # Store encoded data in Redis
    r.set(n, encoded)

    return