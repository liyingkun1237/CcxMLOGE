a = [1,2,3,4,5]
b = [2,5,6,7,8]
''.join(b)
list(set(a).difference(set(b)))


a = "何博睿"
for i in a:
    print(chr(ord(i) ^ 5537),ord(i) ^ 5537)


def encrypt(key, string):
    ans = []
    for i in string:
        ans.append(chr(ord(i) ^ key))
    return ''.join(ans)

c = encrypt(5537,'110108199110034811')

encrypt(5537,encrypt(5537,'110108199110034811'))