# -*- coding: utf-8 -*-
import os
from collections import deque
from pysodium import randombytes
from datetime import datetime, timezone
import time

def toposort(nodes, parents):
    seen = {}
    def visit(u):
        if u in seen:
            if seen[u] == 0:
                raise ValueError('not a DAG')
        elif u in nodes:
            seen[u] = 0
            for v in parents(u):
                yield from visit(v)
            seen[u] = 1
            yield u
    for u in nodes:
        yield from visit(u)


def bfs(s, succ):
    s = tuple(s)
    seen = set(s)
    q = deque(s)
    while q:
        u = q.popleft()
        yield u
        for v in succ(u):
            if not v in seen:
                seen.add(v)
                q.append(v)


def dfs(s, succ):
    seen = set()
    q = [s]
    while q:
        u = q.pop()
        yield u
        seen.add(u)
        for v in succ(u):
            if v not in seen:
                q.append(v)


def randrange(n):
    a = (n.bit_length() + 7) // 8  # number of bytes to store n
    b = 8 * a - n.bit_length()     # number of shifts to have good bit number
    r = int.from_bytes(randombytes(a), byteorder='big') >> b
    while r >= n:
        r = int.from_bytes(randombytes(a), byteorder='big') >> b
    return r


def get_location():
    return os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def binstr(string):
    '''
    :param string:
    :return: byte representation of the string
    '''
    return bytes(string, 'utf-8')

def test_utc():
    t1 = int(time.time())
    print(t1)
    d = datetime.utcfromtimestamp(t1)
    print(d)
    t2 = int(d.replace(tzinfo=timezone.utc).timestamp())
    assert t1 == t2
    print(t2)


def get_ts(y,m,d,h=0,mm=0,s=0):
    return int(datetime(y,m,d,h,mm,s).replace(tzinfo=timezone.utc).timestamp())