"""Remote archive guards. Only finalized, job-owned trees may be removed."""
# Executed by python3 on the login node; does not import our GPU environments.
MANIFEST_SCRIPT = r'''
import hashlib,json,os,pathlib,sys
root=pathlib.Path(sys.argv[1])
if root.is_symlink() or not root.is_dir(): raise RuntimeError('Missing or symlinked run directory')
if not (root/'exit.json').is_file(): raise RuntimeError('Launcher exit is unconfirmed; remote files retained')
records=[]
for directory, dirs, files in os.walk(root,followlinks=False):
 parent=pathlib.Path(directory)
 for name in list(dirs):
  p=parent/name
  if p.parent==root and name in ('model','predefined_trajectories'):
   dirs.remove(name); continue
  if p.is_symlink(): files.append(name); dirs.remove(name)
 for name in sorted(files):
  p=parent/name; relative=str(p.relative_to(root))
  if relative in ('.archive-manifest.json','.archive-manifest.tmp'): continue
  if p.is_symlink():
   target=os.readlink(p)
   if pathlib.Path(target).is_absolute() or not p.resolve().is_relative_to(root.resolve()):
    raise RuntimeError('External result link: '+relative)
   records.append(dict(path=relative,link=target)); continue
  h=hashlib.sha256()
  with p.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''): h.update(chunk)
  stat=p.stat()
  records.append(dict(path=relative,size=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=h.hexdigest()))
body=json.dumps(dict(files=records),sort_keys=True).encode()
temporary=root/'.archive-manifest.tmp'; temporary.write_bytes(body)
temporary.replace(root/'.archive-manifest.json')
print(hashlib.sha256(body).hexdigest())
'''

DELETE_SCRIPT = r'''
import hashlib,json,os,pathlib,shutil,sys
root=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
if root.is_symlink() or root.parent.name!='worldexplorer-runs' or not root.name.startswith('we_'):
 raise RuntimeError('Unsafe cleanup path')
if not (root/'exit.json').is_file(): raise RuntimeError('Launcher exit unconfirmed')
manifest=root/'.archive-manifest.json'; raw=manifest.read_bytes()
if hashlib.sha256(raw).hexdigest()!=expected: raise RuntimeError('Archive manifest changed')
body=json.loads(raw)
for record in body['files']:
 p=root/record['path']
 if not p.resolve().is_relative_to(root.resolve()): raise RuntimeError('Path escaped run')
 if 'link' in record:
  if not p.is_symlink() or os.readlink(p)!=record['link']: raise RuntimeError('Result link changed')
 else:
  stat=p.stat()
  if stat.st_size!=record['size'] or stat.st_mtime_ns!=record['mtime_ns']: raise RuntimeError('Result file changed')
# rmtree unlinks the shared model/trajectory symlinks, never their targets.
shutil.rmtree(root)
print('removed')
'''
