import os, base64, zlib

def ce(fp):
    with open(fp, 'rb') as f:
        data = f.read()
    return base64.b85encode(zlib.compress(data, 9)).decode('ascii')

lines = []
for fname in ['auction_net.pt', 'range_net.pt', 'opp_profiles.pkl']:
    fpath = './models/' + fname
    if os.path.exists(fpath):
        lines.append('        ' + repr(fname) + ': ' + repr(ce(fpath)) + ',')
        print('Embedded: ' + fname)

block = '\n'.join(lines)
loader = '\nimport base64,zlib,os,tempfile\ndef _extract_models():\n    d=os.path.join(tempfile.gettempdir(),"ghostbot")\n    os.makedirs(d,exist_ok=True)\n    E={\n' + block + '\n    }\n    for fn,data in E.items():\n        p=os.path.join(d,fn)\n        if not os.path.exists(p):\n            with open(p,"wb") as f: f.write(zlib.decompress(base64.b85decode(data.encode())))\n    return d\nMODELS_DIR=_extract_models()\n\n'

bot = './bot_v3.py' if os.path.exists('./bot_v3.py') else './bot_v2.py'
with open(bot, 'r', encoding='utf-8', errors='ignore') as f:
    src = f.read()
src = src.replace('STARTING_STACK = 5000', loader + 'STARTING_STACK = 5000', 1)
with open('./bot_final.py', 'w', encoding='utf-8') as f:
    f.write(src)
print('SUCCESS! Size: ' + str(os.path.getsize('./bot_final.py')//1024) + ' KB')