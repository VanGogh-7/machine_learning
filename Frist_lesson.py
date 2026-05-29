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

D = {'name': 'Pat', 'job': 'dev', 'age': 40}
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

tuple_A = 1, 2, 3, 4
print(tuple_A)

# while True:
# reply = input('Enter text:')
# if reply == 'stop' : break
# print(reply.upper())

L = [1, 2, 3, 4]
while L:
    front, L = L[0], L[1:]
    print(front, L)

seq = [1, 2, 3, 4]
a, *b = seq
print(b)

x = 10
while x:
    x -= 1
    if x % 2 != 0: continue
    print(x)

#num0 = 1
#while True:
#    tool = input(f'{num0} What\'s your favorite language? please enter:')
#    if tool == 'stop': break
#    print('Bravo!' if tool == 'Python' else 'Try again...')
#    num0 += 1

#num1 = 1
#while (tool := input(f'{num1}) What\'s your favorite language? ')) != 'stop':
#    print('Bravo!' if tool == 'Python' else 'Try again...')
#    num1 += 1

for x in [1,2,3,4]:
    print(x**2, end=' ')

def times0(x, y):
    z = x * y
    return z

times1 = lambda x, y : x * y

def intersect(seq1, seq2):
    res = []
    for x in seq1:
        if x in seq2:
            res.append(x)
    return res


