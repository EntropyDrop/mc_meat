"""Optional human fine labels. Unknown is not a negative class."""
import json
from uv_visibility import encode_rgba
SCHEMA='mc-meat-semantics-v2'
LEGACY_CLASSES=[dict(id=1,name='skin',label='裸露皮肤',meat=True,color='#26bb80'),dict(id=2,name='eyes',label='眼睛（眼白/瞳孔）',meat=True,color='#00a9e8'),dict(id=3,name='mouth',label='嘴巴',meat=True,color='#f17b9d'),dict(id=4,name='hair',label='头发',meat=False,color='#9a79df'),dict(id=5,name='eyebrows',label='眉毛',meat=False,color='#ea9a27'),dict(id=6,name='beard',label='胡子',meat=False,color='#c77c47'),dict(id=7,name='glasses',label='眼镜',meat=False,color='#e25ee1'),dict(id=8,name='other',label='衣物/帽子/其他饰品',meat=False,color='#6985a3')]
# Stable PNG ids; removed classes never get silently reassigned.
CLASSES=[c for c in LEGACY_CLASSES if c['id'] in {1,2,3,5,6}]
CLASS_IDS=[c['id'] for c in CLASSES]
PARENT={c['id']:c['meat'] for c in CLASSES}
def validate_labels(labels,valid,selected):
    if not isinstance(labels,list) or len(labels)!=4096:raise ValueError('细分类需要 4096 个 UV 格')
    chosen=set(selected)
    for i,c in enumerate(labels):
        if type(c) is not int or c not in [0,*CLASS_IDS]:raise ValueError('细分类编号无效')
        if c and (not valid[i] or PARENT[c]!=(i in chosen)):raise ValueError('细分类与有效区或 meat 标签冲突；请先修正并保存 meat')
    return labels

def encode_labels(labels,valid):
    return encode_rgba(b''.join(bytes([int(c)]*3+[255 if valid[i] else 0]) for i,c in enumerate(labels)))
