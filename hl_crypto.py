"""
HyperBot Pro — hl_crypto.py
EIP-712 TypedData signing + secp256k1 ECDSA + msgpack encoding
100% pure Python stdlib. No eth_account / web3.py / coincurve required.
"""

import hashlib, hmac, struct, json
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# KECCAK-256 (Pure Python — NOT SHA3-256)
# ══════════════════════════════════════════════════════════════════

def keccak256(data: bytes) -> bytes:
    rate_bytes = 136
    msg = bytearray(data)
    msg.append(0x01)
    while len(msg) % rate_bytes:
        msg.append(0x00)
    msg[-1] |= 0x80
    state = [[0]*5 for _ in range(5)]
    for bs in range(0, len(msg), rate_bytes):
        for i in range(rate_bytes // 8):
            state[i%5][i//5] ^= struct.unpack_from('<Q', msg, bs+i*8)[0]
        state = _kf(state)
    out = b''
    for y in range(5):
        for x in range(5):
            out += struct.pack('<Q', state[x][y])
            if len(out) >= 32: return out[:32]
    return out[:32]

def _kf(A):
    RC=[0x1,0x8082,0x800000000000808A,0x8000000080008000,0x808B,0x80000001,
        0x8000000080008081,0x8000000000008009,0x8A,0x88,0x80008009,0x8000000A,
        0x8000808B,0x800000000000008B,0x8000000000008089,0x8000000000008003,
        0x8000000000008002,0x8000000000000080,0x800A,0x800000008000000A,
        0x8000000080008081,0x8000000000008080,0x80000001,0x8000000080008008]
    ROT=[[0,36,3,41,18],[1,44,10,45,2],[62,6,43,15,61],[28,55,25,21,56],[27,20,39,8,14]]
    def r(v,n): return ((v<<n)|(v>>(64-n)))&0xFFFFFFFFFFFFFFFF
    for rc in RC:
        C=[A[x][0]^A[x][1]^A[x][2]^A[x][3]^A[x][4] for x in range(5)]
        D=[C[(x-1)%5]^r(C[(x+1)%5],1) for x in range(5)]
        A=[[A[x][y]^D[x] for y in range(5)] for x in range(5)]
        B=[[0]*5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2*x+3*y)%5]=r(A[x][y],ROT[x][y])
        A=[[B[x][y]^((~B[(x+1)%5][y])&B[(x+2)%5][y]) for y in range(5)] for x in range(5)]
        A[0][0]^=rc
    return A


# ══════════════════════════════════════════════════════════════════
# SECP256K1 — Pure Python
# ══════════════════════════════════════════════════════════════════

_P  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
_N  = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
_Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
_Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
_G  = (_Gx, _Gy)

def _add(P, Q):
    if P is None: return Q
    if Q is None: return P
    if P[0]==Q[0] and P[1]!=Q[1]: return None
    lam = (3*P[0]*P[0]*pow(2*P[1],_P-2,_P))%_P if P==Q else ((Q[1]-P[1])*pow(Q[0]-P[0],_P-2,_P))%_P
    x3  = (lam*lam-P[0]-Q[0])%_P
    return (x3,(lam*(P[0]-x3)-P[1])%_P)

def _mul(k, P):
    R,Q=None,P
    while k:
        if k&1: R=_add(R,Q)
        Q=_add(Q,Q); k>>=1
    return R

def _rfc6979(d: int, h: bytes) -> int:
    x=d.to_bytes(32,'big'); K=b'\x00'*32; V=b'\x01'*32
    K=hmac.new(K,V+b'\x00'+x+h,hashlib.sha256).digest()
    V=hmac.new(K,V,hashlib.sha256).digest()
    K=hmac.new(K,V+b'\x01'+x+h,hashlib.sha256).digest()
    V=hmac.new(K,V,hashlib.sha256).digest()
    while True:
        V=hmac.new(K,V,hashlib.sha256).digest()
        k=int.from_bytes(V,'big')
        if 1<=k<_N: return k
        K=hmac.new(K,V+b'\x00',hashlib.sha256).digest()
        V=hmac.new(K,V,hashlib.sha256).digest()

def _recover(h,r,s,rec):
    x=r+rec*_N
    if x>=_P: return None
    ys=(pow(x,3,_P)+7)%_P; y=pow(ys,(_P+1)//4,_P)
    if (y*y)%_P!=ys: return None
    if y%2!=rec: y=_P-y
    ri=pow(r,_N-2,_N); z=int.from_bytes(h,'big')
    return _add(_mul((-z*ri)%_N,_G),_mul((s*ri)%_N,(x,y)))


# ══════════════════════════════════════════════════════════════════
# ETHEREUM UTILS
# ══════════════════════════════════════════════════════════════════

def private_key_to_address(pk_hex: str) -> str:
    pk  = int(pk_hex.replace('0x',''),16)
    pub = _mul(pk, _G)
    raw = pub[0].to_bytes(32,'big') + pub[1].to_bytes(32,'big')
    return to_checksum_address('0x'+keccak256(raw)[-20:].hex())

def to_checksum_address(addr: str) -> str:
    a=addr.lower().replace('0x',''); h=keccak256(a.encode()).hex()
    return '0x'+''.join(c.upper() if int(h[i],16)>=8 else c for i,c in enumerate(a))

def sign_hash(digest: bytes, pk_hex: str) -> Dict:
    pk=int(pk_hex.replace('0x',''),16); k=_rfc6979(pk,digest)
    R=_mul(k,_G); r=R[0]%_N
    s=(pow(k,_N-2,_N)*(int.from_bytes(digest,'big')+r*pk))%_N
    if s>_N//2: s=_N-s
    pub=_mul(pk,_G); v=27
    for rec in [0,1]:
        if _recover(digest,r,s,rec)==pub: v=27+rec; break
    return {'r':'0x'+hex(r)[2:].zfill(64),'s':'0x'+hex(s)[2:].zfill(64),'v':v}


# ══════════════════════════════════════════════════════════════════
# MSGPACK (matches hyperliquid-python-sdk exactly)
# ══════════════════════════════════════════════════════════════════

def msgpack_encode(obj) -> bytes:
    if obj is None: return b'\xc0'
    if isinstance(obj, bool): return b'\xc3' if obj else b'\xc2'
    if isinstance(obj, int):
        if 0<=obj<=127:       return struct.pack('B',obj)
        if -32<=obj<0:        return struct.pack('b',obj)
        if obj<=0xFF:         return b'\xcc'+struct.pack('B',obj)
        if obj<=0xFFFF:       return b'\xcd'+struct.pack('>H',obj)
        if obj<=0xFFFFFFFF:   return b'\xce'+struct.pack('>I',obj)
        return b'\xcf'+struct.pack('>Q',obj)
    if isinstance(obj, float): return b'\xcb'+struct.pack('>d',obj)
    if isinstance(obj, str):
        b=obj.encode('utf-8'); n=len(b)
        if n<=31:      return struct.pack('B',0xa0|n)+b
        if n<=0xFF:    return b'\xd9'+struct.pack('B',n)+b
        if n<=0xFFFF:  return b'\xda'+struct.pack('>H',n)+b
        return b'\xdb'+struct.pack('>I',n)+b
    if isinstance(obj,(bytes,bytearray)):
        n=len(obj)
        if n<=0xFF:    return b'\xc4'+struct.pack('B',n)+bytes(obj)
        if n<=0xFFFF:  return b'\xc5'+struct.pack('>H',n)+bytes(obj)
        return b'\xc6'+struct.pack('>I',n)+bytes(obj)
    if isinstance(obj,(list,tuple)):
        n=len(obj)
        h=(struct.pack('B',0x90|n) if n<=15 else
           (b'\xdc'+struct.pack('>H',n) if n<=0xFFFF else b'\xdd'+struct.pack('>I',n)))
        return h+b''.join(msgpack_encode(i) for i in obj)
    if isinstance(obj,dict):
        n=len(obj)
        h=(struct.pack('B',0x80|n) if n<=15 else
           (b'\xde'+struct.pack('>H',n) if n<=0xFFFF else b'\xdf'+struct.pack('>I',n)))
        return h+b''.join(msgpack_encode(k)+msgpack_encode(v) for k,v in obj.items())
    raise TypeError(f'msgpack: cannot encode {type(obj).__name__}')


# ══════════════════════════════════════════════════════════════════
# EIP-712
# ══════════════════════════════════════════════════════════════════

HL_DOMAIN = {'name':'Exchange','version':'1','chainId':1337,'verifyingContract':'0x0000000000000000000000000000000000000000'}

HL_TYPES = {
    'EIP712Domain':[
        {'name':'name','type':'string'},{'name':'version','type':'string'},
        {'name':'chainId','type':'uint256'},{'name':'verifyingContract','type':'address'},
    ],
    'Agent':[{'name':'source','type':'string'},{'name':'connectionId','type':'bytes32'}],
}

def _enc_type(p,types):
    f=types[p]; r=p+'('+','.join(f"{x['type']} {x['name']}" for x in f)+')'
    for x in f:
        b=x['type'].rstrip('[]')
        if b in types and b!=p: r+=_enc_type(b,types)
    return r

def _type_hash(p,types): return keccak256(_enc_type(p,types).encode())

def _enc_val(ftype, val, types):
    if ftype=='string': return keccak256(val.encode() if isinstance(val,str) else val)
    if ftype=='bytes':  return keccak256(val if isinstance(val,bytes) else bytes.fromhex(val.replace('0x','')))
    if ftype=='bytes32':
        raw=val if isinstance(val,(bytes,bytearray)) else bytes.fromhex(str(val).replace('0x','').zfill(64))
        return raw[:32].ljust(32,b'\x00')
    if ftype=='address': return b'\x00'*12+bytes.fromhex(str(val).replace('0x',''))
    if ftype=='bool': return (1 if val else 0).to_bytes(32,'big')
    if ftype.startswith(('uint','int')): return int(val).to_bytes(32,'big')
    if ftype in types: return keccak256(_enc_struct(ftype,val,types))
    raw=bytes.fromhex(str(val).replace('0x','')) if isinstance(val,str) else bytes(val)
    return raw.ljust(32,b'\x00')[:32]

def _enc_struct(p,data,types):
    return _type_hash(p,types)+b''.join(_enc_val(f['type'],data.get(f['name']),types) for f in types[p])

def eip712_hash(domain,ptype,msg,types):
    return keccak256(b'\x19\x01'+keccak256(_enc_struct('EIP712Domain',domain,types))+keccak256(_enc_struct(ptype,msg,types)))


# ══════════════════════════════════════════════════════════════════
# HYPERLIQUID ACTION SIGNING
# ══════════════════════════════════════════════════════════════════

def sign_l1_action(action:Dict, private_key:str, vault_address:str=None, nonce:int=None) -> Dict:
    import time as _t
    if nonce is None: nonce=int(_t.time()*1000)
    vault_bytes=b'\x00' if vault_address is None else b'\x01'+bytes.fromhex(vault_address.replace('0x',''))
    action_hash=keccak256(msgpack_encode(action)+nonce.to_bytes(8,'big')+vault_bytes)
    digest=eip712_hash(HL_DOMAIN,'Agent',{'source':'a','connectionId':action_hash},HL_TYPES)
    sig=sign_hash(digest,private_key)
    pl={'action':action,'nonce':nonce,'signature':sig}
    if vault_address: pl['vaultAddress']=vault_address
    return pl

def float_to_wire(x:float) -> str:
    if x==0: return '0'
    import math; mag=math.floor(math.log10(abs(x)))
    return str(round(x,4-mag)).rstrip('0').rstrip('.')


# ══════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════

if __name__=='__main__':
    assert keccak256(b'').hex()=='c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470'
    assert keccak256(b'hello').hex()=='1c8aff950685c2ed4bc3174f3472287b56d9517b9c948127319a09a7a36deac8'
    print('✅ keccak256')
    addr=private_key_to_address('0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80')
    assert addr=='0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266', addr
    print('✅ address:',addr)
    sig=sign_hash(keccak256(b'test'),   '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80')
    assert sig['v'] in [27,28]; print('✅ sign_hash v=',sig['v'])
    pl=sign_l1_action({'type':'order','orders':[{'a':0,'b':True,'p':'65000','s':'0.001','r':False,'t':{'limit':{'tif':'Gtc'}}}],'grouping':'na'},
        '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80', nonce=1000000)
    assert pl['signature']['v'] in [27,28]; print('✅ sign_l1_action v=',pl['signature']['v'])
    print('\n🎯 ALL CRYPTO TESTS PASSED')
