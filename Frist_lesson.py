import sys
import math

print(sys.platform)
print(2 ** 100)
x = 'Hack!'
print(x * 8)

print(math.pi)
print(math.sqrt(81))

S = bytearray(b'deep')
S.extend(b' learning')
print(S.decode())

line = 'aaa, bbb, ccc, dd'

new_line = line.split(',')

print(new_line)

tool = 'python'
major = 3
minor = 3
print(f'Using {tool} version {major}.{minor + 9}')

list_L = [1, 2, 3, 4]
list_L = list_L + [5, 6, 7]  # or L.extend([5,6,7])
print(list_L)

D = {'name': 'Pat', 'job': 'dev', 'age':40}
print(D['name'])

C = {}
C['name'] = 'Pat'
C['job'] = 'dev'
C['age'] = 40
print(C)

set_A = {'a', 'b', 'c'}
set_B = set('hack')
print(set_A & set_B, set_A | set_B)

print(type(set_A))

tuple_A = 1,2,3,4
print(tuple_A)

