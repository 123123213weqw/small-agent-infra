"""Read-only mmap votes, mathematically identical to all 64 SQLite postings votes.

No anchor approximation, Bloom pruning, threshold change, API or ledger writes.
Immutable arrays are derived from one read transaction and checksum verified.
"""
import json,math,os,sqlite3,struct,time
from pathlib import Path
import numpy as np
from small_agent.curation.supply_control import digest,file_sha
from small_agent.curation.release_gates_v5 import write

def prepare(root,index_manifest=None):
    root=Path(root)
    with sqlite3.connect('file:'+str(root/'postings.sqlite')+'?mode=ro',uri=True) as db:
        db.execute('BEGIN')
        ledger=json.loads(db.execute("SELECT v FROM meta WHERE k='ledger'").fetchone()[0])
        n,maximum=db.execute('SELECT COUNT(*),COALESCE(MAX(id),0) FROM docs').fetchone()
        count=db.execute('SELECT COUNT(*) FROM postings').fetchone()[0]
        identity=dict(ledger=ledger,documents=n,maximum_id=maximum,postings=count,representation='native_le_u64_blob_order_v1')
        if index_manifest and json.loads(Path(index_manifest).read_text())['ledger']!=ledger:raise ValueError('Stale vector index')
        directory=root/'vector_v22'/digest(identity);directory.mkdir(parents=True,exist_ok=True)
        marker=directory/'READY.json'
        if marker.exists():
            value=json.loads(marker.read_text())
            if value['identity']!=identity:raise ValueError('Vector binding changed')
            for name,h in value['files'].items():
                if file_sha(directory/name)!=h:raise ValueError('Vector cache changed')
        else:
            started=time.monotonic()
            # SQLite BLOB ordering is byte lexicographic; >u8 retains precisely that order.
            keys=np.lib.format.open_memmap(directory/'keys.npy',mode='w+',dtype='<u8',shape=(count,))
            ids=np.lib.format.open_memmap(directory/'ids.npy',mode='w+',dtype='<u4',shape=(count,))
            if maximum>=2**32:raise ValueError('Vector document IDs overflow')
            sizes=np.lib.format.open_memmap(directory/'sizes.npy',mode='w+',dtype='<i8',shape=(maximum+1,))
            ks=np.lib.format.open_memmap(directory/'ks.npy',mode='w+',dtype='<i8',shape=(maximum+1,))
            sizes[:]=-1;ks[:]=-1
            for i,size,k in db.execute('SELECT id,n,k FROM docs'):
                if i<0 or size<0 or not 0<=k<=64:raise ValueError('Invalid document shape')
                sizes[i]=size;ks[i]=k
            cursor=db.execute('SELECT h,d FROM postings ORDER BY h,d');offset=0
            while True:
                rows=cursor.fetchmany(65536)
                if not rows:break
                if any(len(h)!=8 or d<0 or d>maximum for h,d in rows):raise ValueError('Invalid posting')
                values=np.frombuffer(b''.join(h for h,d in rows),dtype='>u8')
                docids=np.fromiter((d for h,d in rows),dtype='<u4',count=len(rows))
                if np.any(sizes[docids]<0):raise ValueError('Posting lacks document')
                keys[offset:offset+len(rows)]=values;ids[offset:offset+len(rows)]=docids;offset+=len(rows)
            if offset!=count or (count and np.any(keys[1:]<keys[:-1])):raise ValueError('Incomplete/unsorted postings')
            for array in (keys,ids,sizes,ks):array.flush()
            names=['keys.npy','ids.npy','sizes.npy','ks.npy']
            for name in names:
                with (directory/name).open('rb') as f:os.fsync(f.fileno())
            write(marker,dict(identity=identity,files={name:file_sha(directory/name) for name in names},build_seconds=time.monotonic()-started))
        current=root/'vector_v22/CURRENT';tmp=current.with_suffix('.tmp');tmp.write_text(str(marker)+'\n');tmp.replace(current)
    return marker

class VectorVotes:
    def __init__(self,root):
        self.marker=Path((Path(root)/'vector_v22/CURRENT').read_text().strip())
        self.binding=json.loads(self.marker.read_text());self.root=self.marker.parent
        for name in ('keys','ids','sizes','ks'):setattr(self,name,np.load(self.root/(name+'.npy'),mmap_mode='r'))
    def verify_files(self):
        if json.loads(self.marker.read_text())!=self.binding:raise ValueError('Vector binding changed')
        for name,h in self.binding['files'].items():
            if file_sha(self.root/name)!=h:raise ValueError('Vector cache changed')
    def query(self,db,sketch,n,policy):
        if not sketch:return []
        hashes=np.frombuffer(b''.join(struct.pack('<Q',h) for h in sorted(set(sketch))),dtype='>u8').astype('<u8')
        lower=np.searchsorted(self.keys,hashes,side='left');upper=np.searchsorted(self.keys,hashes,side='right')
        lo=math.ceil(n*policy['threshold']);hi=math.floor(n/policy['threshold'])
        chunks=[]
        for a,b in zip(lower,upper):
            ds=self.ids[a:b];sizes=self.sizes[ds];chunks.append(ds[(sizes>=lo)&(sizes<=hi)])
        if not any(len(x) for x in chunks):return []
        found,votes=np.unique(np.concatenate(chunks),return_counts=True)
        # Match SQLite CAST(...+0.999999 AS INTEGER), including short sketches.
        needed=np.maximum(1,(np.minimum(self.ks[found],len(sketch))*policy['minimum_shared_fraction']+0.999999).astype(np.int64))
        qualified=found[votes>=needed][:policy['maximum_pair_checks_per_document']+1]
        selected=[];counts=dict(zip(found.tolist(),votes.tolist()))
        for start in range(0,len(qualified),512):
            batch=qualified[start:start+512].tolist()
            query='SELECT id,body,split,source,rowno,n,k FROM docs WHERE id IN ('+','.join('?' for _ in batch)+') ORDER BY id'
            selected.extend(tuple(r)+(int(counts[r[0]]),) for r in db.execute(query,batch))
        if len(selected)!=len(qualified):raise ValueError('Vector index/document mismatch')
        return selected
