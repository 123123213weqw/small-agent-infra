"""v18 index writer: exact original coverage and SHA gates, FULL-durable batched postings."""
from pathlib import Path
import json,heapq,struct
from small_agent.curation.stock_v6 import DiskNearIndex
from small_agent.curation.release_gates_v5 import inventory_files,ledger_snapshot,lexical,sha,write
from small_agent.curation.supply_control import file_sha,digest

class BatchedDiskNearIndex(DiskNearIndex):
    def update(self,database,legacy_receipt,sketch_root):
        import pyarrow.parquet as pq
        # FULL durability is retained; fewer commits and sorted posting writes.
        # Caller must hold production-release.lock, just like the old builder.
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA cache_size=-131072')
        files=inventory_files(database,legacy_receipt);snapshot=ledger_snapshot(database)
        for e in files:
            source=Path(e['path'])
            if file_sha(source)!=e['sha256']:raise ValueError('Inventory source changed')
            old=self.db.execute('SELECT id,sha,complete FROM sources WHERE path=?',(str(source),)).fetchone()
            if old and old[1]!=e['sha256']:raise ValueError('Append-only reference source changed')
            if old and old[2]:continue
            if not old:
                with self.db:self.db.execute('INSERT INTO sources(path,sha) VALUES (?,?)',(str(source),e['sha256']))
            sid=self.db.execute('SELECT id FROM sources WHERE path=?',(str(source),)).fetchone()[0]
            key=digest(e);cached=Path(sketch_root)/(key+'.sketch.parquet');marker=Path(sketch_root)/(key+'.done.json')
            reuse=False
            if marker.exists():
                m=json.loads(marker.read_text())
                reuse=m['input']==e and file_sha(cached)==m['sha256']
            rowno=0
            pf=pq.ParquetFile(cached if reuse else source)
            columns=['body_sha256','split','row','shingle_count','sketch'] if reuse else ['text','normalized_sha256','split']
            for b in pf.iter_batches(batch_size=4096,columns=columns):
                with self.db:
                    postings=[]
                    for r in b.to_pylist():
                        if reuse:h=r['body_sha256'];n=r['shingle_count'];sketch=r['sketch'];pos=r['row']
                        else:
                            h=sha(r['text'])
                            if h!=r['normalized_sha256']:raise ValueError('Reference body differs')
                            values=lexical(r['text']);n=len(values);sketch=heapq.nsmallest(64,values);pos=rowno
                        body=bytes.fromhex(h);old=self.db.execute('SELECT id,split,n,k FROM docs WHERE body=?',(body,)).fetchone()
                        if old:
                            if old[1:]!=(r['split'],n,len(sketch)):raise ValueError('Index body/split/sketch count conflict')
                        else:
                            cur=self.db.execute('INSERT INTO docs(body,split,source,rowno,n,k) VALUES (?,?,?,?,?,?)',(body,r['split'],sid,pos,n,len(sketch)))
                            postings.extend((struct.pack('<Q',v),cur.lastrowid) for v in sketch)
                        rowno+=1
                    postings.sort()
                    self.db.executemany('INSERT INTO postings VALUES (?,?)',postings)
                if rowno%16384<4096:print('INDEX_PROGRESS',source.name,rowno,flush=True)
            if file_sha(source)!=e['sha256']:raise ValueError('Source changed during index update')
            with self.db:self.db.execute('UPDATE sources SET complete=1 WHERE id=?',(sid,))
        self.db.execute('ATTACH DATABASE ? AS supply',(str(Path(database).resolve()),))
        try:
            missing=self.db.execute('SELECT COUNT(*) FROM supply.documents d LEFT JOIN docs n ON n.body=unhex(d.body_sha) WHERE n.id IS NULL OR n.split!=d.split').fetchone()[0]
            extra=self.db.execute('SELECT COUNT(*) FROM docs n LEFT JOIN supply.documents d ON d.body_sha=lower(hex(n.body)) WHERE d.body_sha IS NULL').fetchone()[0]
        finally:self.db.execute('DETACH DATABASE supply')
        count=self.db.execute('SELECT COUNT(*) FROM docs').fetchone()[0]
        if missing or extra or count!=snapshot['documents'] or ledger_snapshot(database)!=snapshot:raise ValueError('Index ledger coverage differs/stale')
        state=dict(version=18,stage='complete_registered_disk_index',database=str((self.root/'postings.sqlite').resolve()),ledger=snapshot,inputs=files,
            documents=count,missing_documents=0,extra_documents=0,algorithm='same casefold lexical5 bottom64 overlap>=half then exact Jaccard>=0.90; append-only indexed retrieval',
            checker_sha256=file_sha(Path(__file__)),mutable_append_only_index=True)
        generation=self.root/digest(state);generation.mkdir(exist_ok=True);write(generation/'manifest.json',state)
        with self.db:self.db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)',('ledger',json.dumps(snapshot,sort_keys=True)))
        pointer=self.root/'.CURRENT.tmp';pointer.write_text(str((generation/'manifest.json').resolve())+'\n');pointer.replace(self.root/'CURRENT')
        self.db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return generation/'manifest.json'
