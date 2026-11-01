"""Exact face-normal orthographic coverage, independent of RGB and labels.

Only the same body's outward outer face can exclude an inner texel. Another
limb occluding it in one pose is not permanent occlusion. Each inner rectangle
must be fully covered by the union of alpha=255 outer rectangles. Positive-area
slivers count as visible. Shell enlargement matches skin_parts / Three.js.
"""
from fractions import Fraction
from functools import lru_cache
import struct
import zlib

POLICY = 'same-part-face-orthographic-full-opaque-v1'
SCHEMA = 'mc-meat-v2'


@lru_cache(maxsize=32)
def axis_cover(count, inflate):
    n=Fraction(count); extra=Fraction(str(inflate))
    inner=[Fraction(k)-n/2 for k in range(count+1)]
    outer=[(Fraction(k)/n-Fraction(1,2))*(n+extra) for k in range(count+1)]
    return tuple(tuple(j for j in range(count)
                       if max(inner[i],outer[j]) < min(inner[i+1],outer[j+1]))
                 for i in range(count))


def visible_skin(rgba, parts, occupied):
    """Return cleared RGBA, trainable cells and full-occlusion flags (4096 bytes)."""
    assert len(rgba)==16384 and len(occupied)==4096
    hidden=bytearray(4096)
    outer={p['name']:p for p in parts if p['layer']==1}
    for p in parts:
        if p['layer']!=0:continue
        shell=outer[p['name']]
        for face,(u,v,w,h,*flags) in p['faces'].items():
            ou,ov,ow,oh,*oflags=shell['faces'][face]
            assert (w,h,flags)==(ow,oh,oflags), 'Inner/outer face conventions differ'
            xs=axis_cover(w,shell['inflate']);ys=axis_cover(h,shell['inflate'])
            for y in range(h):
                for x in range(w):
                    i=(v+y)*64+u+x
                    if occupied[i] and all(rgba[((ov+oy)*64+ou+ox)*4+3]==255
                                           for oy in ys[y] for ox in xs[x]):hidden[i]=1
    valid=bytes(int(bool(a) and not b) for a,b in zip(occupied,hidden))
    out=bytearray(rgba)
    for i,ok in enumerate(valid):
        if not ok:out[i*4:i*4+4]=b'\0'*4
    return bytes(out),valid,bytes(hidden)


def encode_rgba(rgba):
    assert len(rgba)==16384
    def chunk(tag,data):
        return struct.pack('>I',len(data))+tag+data+struct.pack('>I',zlib.crc32(tag+data))
    raw=b''.join(b'\0'+rgba[y*256:(y+1)*256] for y in range(64))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,64,8,6,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')


def annotation_mask(selected,valid):
    selected=set(selected)
    assert len(valid)==4096 and all(type(i) is int and 0<=i<4096 and valid[i] for i in selected)
    raw=bytearray(16384)
    for i,ok in enumerate(valid):
        if ok:raw[i*4:i*4+4]=bytes([255 if i in selected else 0]*3+[255])
    return encode_rgba(raw)
