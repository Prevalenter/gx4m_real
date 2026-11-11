"""Upload/download model artifacts using locally authorized Baidu Netdisk.

Requires requests. Credentials live outside the repository in ~/.config/gx4m-model-storage.
Tokens use Windows DPAPI on Windows and owner-only JSON files on other platforms.
Never prints tokens, credentials or authenticated URLs.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import tempfile
import requests

CONFIG = Path.home()/'.config/gx4m-model-storage'
API = 'https://pan.baidu.com/rest/2.0/xpan/'


def protect(raw, decrypt=False):
    class Blob(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]
    buf = ctypes.create_string_buffer(raw)
    src = Blob(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    dst = Blob()
    fn = ctypes.windll.crypt32.CryptUnprotectData if decrypt else ctypes.windll.crypt32.CryptProtectData
    if not fn(ctypes.byref(src), None, None, None, None, 1, ctypes.byref(dst)):
        raise RuntimeError('Windows DPAPI operation failed')
    try:
        return ctypes.string_at(dst.data, dst.size)
    finally:
        ctypes.windll.kernel32.LocalFree(dst.data)


class Drive:
    def __init__(self):
        self.app = json.loads((CONFIG/'baidu-app.json').read_text(encoding='utf-8-sig'))
        windows = os.name == 'nt'
        path = CONFIG/('baidu-token.dpapi' if windows else 'baidu-token.json')
        raw = path.read_bytes()
        self.token = json.loads(protect(raw, True) if windows else raw.decode('utf-8-sig'))
        if not self.token.get('refresh_token'):
            raise RuntimeError('Missing refresh_token; complete account authorization first')
        # Refresh on startup; refresh tokens can rotate, so persist the full response immediately.
        r = requests.post('https://openapi.baidu.com/oauth/2.0/token', data={
            'grant_type': 'refresh_token', 'refresh_token': self.token['refresh_token'],
            'client_id': self.app['app_key'], 'client_secret': self.app['secret_key']}, timeout=30).json()
        if 'access_token' not in r:
            raise RuntimeError('OAuth refresh failed: '+str(r.get('error', 'unknown')))
        self.token.update(r)
        payload = json.dumps(self.token).encode()
        with tempfile.NamedTemporaryFile(dir=CONFIG, prefix='.baidu-token-', delete=False) as output:
            temporary = Path(output.name)
            try:
                output.write(protect(payload) if windows else payload)
            except BaseException:
                output.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def api(self, endpoint, method, data=None, **params):
        params.update(method=method, access_token=self.token['access_token'])
        r = requests.post(API+endpoint, params=params, data=data, timeout=60) if data is not None else requests.get(API+endpoint, params=params, timeout=60)
        result = r.json()
        if result.get('errno', 0) != 0:
            raise RuntimeError(f'{method}: Baidu errno {result.get("errno")}')
        return result

    def upload(self, local, remote):
        local = Path(local)
        size = local.stat().st_size
        block_size = 4*1024*1024
        with local.open('rb') as f:
            hashes = [hashlib.md5(chunk).hexdigest() for chunk in iter(lambda:f.read(block_size), b'')]
        pre = self.api('file', 'precreate', data={'path':remote, 'size':size, 'isdir':0,
                       'autoinit':1, 'rtype':0, 'block_list':json.dumps(hashes)})
        upload_id = pre.get('uploadid')
        if pre.get('return_type') != 2:
            server = 'https://d.pcs.baidu.com'
            for index in pre.get('block_list', range(len(hashes))):
                index = int(index)
                with local.open('rb') as f:
                    f.seek(index*block_size)
                    chunk = f.read(block_size)
                r = requests.post(server+'/rest/2.0/pcs/superfile2', params={
                    'method':'upload', 'access_token':self.token['access_token'], 'type':'tmpfile',
                    'path':remote, 'uploadid':upload_id, 'partseq':index},
                    files={'file':('chunk',chunk,'application/octet-stream')}, timeout=180).json()
                if r.get('md5') != hashes[index]:
                    raise RuntimeError('Block upload checksum failed')
            result = self.api('file','create',data={'path':remote,'size':size,'isdir':0,
                       'rtype':0,'uploadid':upload_id,'block_list':json.dumps(hashes)})
        else:
            result = pre.get('info', pre)
        if 'fs_id' not in result:
            listing = self.api('file','list',dir=str(Path(remote).parent).replace('\\','/'),limit=1000)
            result = next(x for x in listing['list'] if x['path']==remote)
        return {'name':local.name, 'path':remote, 'fs_id':result['fs_id'], 'size':size,
                'sha256':hashlib.sha256(local.read_bytes()).hexdigest()}

    def download(self, entry, target):
        meta = self.api('multimedia','filemetas',fsids=json.dumps([entry['fs_id']]),dlink=1)
        link = meta['list'][0]['dlink']
        temporary = Path(str(target)+'.part')
        digest = hashlib.sha256()
        count = 0
        with requests.get(link,params={'access_token':self.token['access_token']},
                          headers={'User-Agent':'pan.baidu.com'},stream=True,timeout=180) as r:
            if r.status_code != 200:
                raise RuntimeError('Download HTTP status '+str(r.status_code))
            with temporary.open('wb') as f:
                for chunk in r.iter_content(1024*1024):
                    f.write(chunk);digest.update(chunk);count+=len(chunk)
        if count != entry['size'] or digest.hexdigest()!=entry['sha256']:
            temporary.unlink(missing_ok=True)
            raise RuntimeError('Download size/SHA256 mismatch')
        temporary.replace(target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    up=sub.add_parser('upload');up.add_argument('--folder',required=True);up.add_argument('--manifest',type=Path,required=True);up.add_argument('files',nargs='+',type=Path)
    down=sub.add_parser('download');down.add_argument('--manifest',type=Path,required=True);down.add_argument('--out',type=Path,required=True)
    a=p.parse_args();d=Drive()
    if a.command=='upload':
        entries=[]
        for file in a.files:
            remote=a.folder.rstrip('/')+'/'+file.name
            entry=d.upload(file,remote);entries.append(entry)
            a.manifest.parent.mkdir(parents=True,exist_ok=True)
            a.manifest.write_text(json.dumps({'files':entries},indent=2),encoding='utf-8')
            print('Uploaded:',file.name,flush=True)
    else:
        a.out.mkdir(parents=True,exist_ok=True)
        for entry in json.loads(a.manifest.read_text())['files']:
            d.download(entry,a.out/entry['name']);print('Downloaded and SHA256 verified:',entry['name'],flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        # Network exception strings may contain authenticated URLs; never print them.
        print('Operation failed:',type(e).__name__)
        if isinstance(e,RuntimeError):print(str(e))
        raise SystemExit(1)
