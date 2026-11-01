"""Read-only UV format helpers vendored from DDJ best_annotator (2026-10-29).

No service or annotation-writing API. Geometry/layer is context, not a class label.
"""
import hashlib
import struct
import zlib


def digest(data):
    return hashlib.sha256(data).hexdigest()


def decode_rgba(data):
    """Read bounded 64x64, 8-bit RGB/RGBA PNG, including all five row filters."""
    if len(data) > 131072 or data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('meat 标注需使用 64×64 RGB/RGBA PNG 皮肤')
    offset, compressed, channels, ended = 8, bytearray(), None, False
    while offset < len(data):
        size = struct.unpack('>I', data[offset:offset+4])[0]
        tag = data[offset+4:offset+8]; payload = data[offset+8:offset+8+size]
        crc = struct.unpack('>I', data[offset+8+size:offset+12+size])[0]
        if zlib.crc32(tag+payload) != crc:
            raise ValueError('PNG 校验失败')
        if offset == 8:
            width, height, depth, color, comp, filt, interlace = struct.unpack('>IIBBBBB', payload)
            if tag != b'IHDR' or (width,height,depth,comp,filt,interlace) != (64,64,8,0,0,0) or color not in (2,6):
                raise ValueError('meat 标注需使用 64×64、8 位 RGB/RGBA PNG 皮肤')
            channels = 4 if color == 6 else 3
        elif tag == b'IHDR':
            raise ValueError('PNG 重复图像头')
        if tag == b'IDAT': compressed.extend(payload)
        offset += size+12
        if tag == b'IEND':
            if size or offset != len(data): raise ValueError('PNG 文件尾无效')
            ended=True; break
    dec=zlib.decompressobj(); raw=dec.decompress(bytes(compressed),16449)
    if not ended or not dec.eof or dec.unused_data or len(raw) != 64*(64*channels+1):
        raise ValueError('PNG 像素长度无效')
    stride=64*channels; prev=bytearray(stride); rgba=bytearray()
    for y in range(64):
        filt=raw[y*(stride+1)]; row=bytearray(raw[y*(stride+1)+1:(y+1)*(stride+1)])
        if filt > 4: raise ValueError('PNG 滤波类型无效')
        for x in range(stride):
            a=row[x-channels] if x>=channels else 0; b=prev[x]; c=prev[x-channels] if x>=channels else 0
            if filt==1: add=a
            elif filt==2: add=b
            elif filt==3: add=(a+b)//2
            elif filt==4:
                p=a+b-c; pa,pb,pc=abs(p-a),abs(p-b),abs(p-c)
                add=a if pa<=pb and pa<=pc else b if pb<=pc else c
            else: add=0
            row[x]=(row[x]+add)&255
        for x in range(64): rgba.extend(row[x*channels:x*channels+3]+bytes([row[x*channels+3] if channels==4 else 255]))
        prev=row
    return bytes(rgba)


def encode_mask(selected):
    pixels=bytearray(4096)
    for i in selected:pixels[i]=255
    def chunk(tag,payload):
        return struct.pack('>I',len(payload))+tag+payload+struct.pack('>I',zlib.crc32(tag+payload))
    raw=b''.join(b'\0'+pixels[y*64:(y+1)*64] for y in range(64))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',64,64,8,0,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')


def skin_parts(rgba):
    """Match the existing editor's classic/slim UV convention; layer is context only."""
    slim=rgba[(20*64+55)*4+3]==0; aw=3 if slim else 4
    head={'right':[0,8,8,8],'left':[16,8,8,8],'top':[8,0,8,8],'bottom':[16,0,8,8,0,1],'front':[8,8,8,8],'back':[24,8,8,8]}
    body={'right':[16,20,4,12],'left':[28,20,4,12],'top':[20,16,8,4],'bottom':[28,16,8,4,0,1],'front':[20,20,8,12],'back':[32,20,8,12]}
    arm={'right':[40,20,4,12],'front':[44,20,aw,12],'left':[44+aw,20,4,12],'top':[44,16,aw,4],'bottom':[44+aw,16,aw,4,0,1],'back':[48+aw,20,aw,12]}
    leg={'right':[0,20,4,12],'left':[8,20,4,12],'top':[4,16,4,4],'bottom':[8,16,4,4,0,1],'front':[4,20,4,12],'back':[12,20,4,12]}
    def shift(face,dx,dy):return {k:[v[0]+dx,v[1]+dy,*v[2:]] for k,v in face.items()}
    bases=[('head',head,[8,8,8],[0,10,0],32,0),('body',body,[8,12,4],[0,0,0],0,16),
           ('rightArm',arm,[aw,12,4],[-4-aw/2,0,0],0,16),('leftArm',shift(arm,-8,32),[aw,12,4],[4+aw/2,0,0],16,0),
           ('rightLeg',leg,[4,12,4],[-2,-12,0],0,16),('leftLeg',shift(leg,16,32),[4,12,4],[2,-12,0],-16,0)]
    parts=[];valid=bytearray(4096)
    for name,faces,size,center,dx,dy in bases:
        for layer in [0,1]:
            fm=shift(faces,dx,dy) if layer else faces
            parts.append(dict(name=name,layer=layer,faces=fm,size=size,center=center,inflate=(1 if name=='head' else .5) if layer else 0))
            for u,v,w,h,*_ in fm.values():
                for y in range(v,v+h):
                    for x in range(u,u+w):
                        i=y*64+x;valid[i]=int(rgba[i*4+3]>0)
    return parts,bytes(valid),'slim' if slim else 'classic'
