"""All-inventory adapters and resumable pointer store; no quality/rights approval by counting."""
from collections import Counter,OrderedDict
from pathlib import Path
import hashlib,json,re,sqlite3
from urllib.parse import urlsplit,urlunsplit
from .supply_control import SupplyControl,digest,file_sha
from .release_gates_v5 import sha,write

LANG={'c++':'cpp','c#':'csharp','shell':'bash','sh':'bash','shellscript':'bash','objective-c':'objc','js':'javascript','ts':'typescript','cmn':'zh','chinese':'zh','english':'en','markdown':'markdown','python':'python'}
EXT={'.py':'python','.js':'javascript','.jsx':'javascript','.ts':'typescript','.tsx':'tsx','.c':'c','.h':'c','.cc':'cpp','.cpp':'cpp','.java':'java','.go':'go','.rs':'rust','.sh':'bash','.rb':'ruby','.php':'php','.cs':'csharp','.html':'html','.css':'css','.sql':'sql','.json':'json','.md':'markdown','.kt':'kotlin','.swift':'swift','.lua':'lua','.pl':'perl','.r':'r','.scala':'scala','.hs':'haskell'}
LICENSES={'mit','apache-2.0','bsd-2-clause','bsd-3-clause','isc','0bsd','unlicense','cc0-1.0'}
FIELDS=('text','content','code','document_text','url','source_locator','repo_name','path','relative_path','language','in_language','license','record_license','detected_licenses','family_id','split','domain','own_text_tokens','normalized_sha256','source_revision','document_id')

def adapter(row,source,file,pos,byte_offset=0):
    text=next((row[k] for k in ('text','content','code','document_text') if isinstance(row.get(k),str)),None)
    if text is None:raise ValueError('missing_text_field')
    # No whitespace collapse or hidden chat conversion.
    h=sha(text);domain=source['domain'];lang=source['language']
    if source.get('normalized'):
        if row['normalized_sha256']!=h:raise ValueError('normalized_body_changed')
        domain=row['domain'];lang=row['language']
    elif domain=='code':lang=LANG.get(str(row.get('language',lang)).lower(),str(row.get('language',lang)).lower())
    else:lang=LANG.get(str(row.get('in_language',row.get('language',lang))).lower(),lang)
    lang=LANG.get(str(lang).lower(),str(lang).lower())
    if domain=='code' and lang in ('unknown',''):lang=EXT.get(Path(str(row.get('path',row.get('relative_path','')))).suffix.lower(),'unknown')
    if domain=='code' and lang=='markdown':domain='technical';lang='en'
    locator=row.get('source_locator',row.get('url'))
    if not isinstance(locator,str) or not locator.startswith(('http://','https://','github:')):
        repo=row.get('repo_name');path=row.get('path',row.get('relative_path'))
        locator='github:'+str(repo)+':'+str(path) if repo and path else 'file-sha256:'+file['sha256']+':row:'+str(pos)
    if domain in ('general_en','general_zh') and locator.startswith(('http://','https://')):
        host=(urlsplit(locator).hostname or '').lower()
        if host in ('docs.python.org','developer.mozilla.org','docs.pytorch.org','numpy.org','doc.rust-lang.org','docs.oracle.com') or host.startswith('docs.') and any(k in text[:4000].lower() for k in ('api','parameter','function','参数','函数')):domain='technical'
    if source.get('normalized'):family=row['family_id'];split=row['split']
    else:
        if row.get('repo_name'):key='repo:'+str(row['repo_name'])
        elif locator.startswith(('http://','https://')):
            u=urlsplit(locator);key=urlunsplit((u.scheme,u.netloc,u.path.rstrip('/'),'',''))
        else:key='body:'+h
        family=sha(key);v=int(sha('2026100211|'+family)[:8],16)%1000
        split='validation' if v<10 else 'test' if v<20 else 'train'
    license=str(row.get('record_license',row.get('license',''))).lower().strip()
    if not license and isinstance(row.get('detected_licenses'),list):
        declared={str(x).lower() for x in row['detected_licenses']}
        if declared and declared<=LICENSES:license=sorted(declared)[0]
    n=row.get('own_text_tokens') if source.get('normalized') else None
    return dict(document_id=h,normalized_sha256=h,source_id=source['id'],source_revision=source['revision'],source_locator=locator,
                domain=domain,language=lang,family_id=family,split=split,text=text,own_text_tokens=n,record_license=license,
                policy_bucket='record_license_whitelisted' if license in LICENSES else source['terms_scope'],
                origin_file=str(file['path']),origin_sha256=file['sha256'],origin_row=pos,origin_byte_offset=byte_offset)

def rule(row):
    text=row['text'];domain=row['domain'];size=len(text.encode())
    if not text.strip():return 'DROP','empty'
    if '\x00' in text or '\ufffd' in text:return 'DROP','null_or_replacement'
    if size>100000:return 'HOLD','whole_body_context_limit'
    if len(text.strip())<(24 if domain=='code' else 80):return 'DROP','tiny'
    if re.search(r'unlock (?:the |this |full |complete )*(?:answer|solution)|answer preview|subscribe to (?:view|unlock|access)',text,re.I):return 'DROP','explicit_locked_preview'
    if domain=='code' and any(p in row['source_locator'].lower() for p in ('/node_modules/','/vendor/','/third_party/','/dist/','/min.js')):return 'DROP','vendored_or_generated_path'
    paragraphs=[x.strip() for x in text.split('\n\n') if len(x.strip())>=80];counts=Counter(paragraphs)
    if sum(len(x)*(n-1) for x,n in counts.items())/max(1,len(text))>.35:return 'DROP','repeated_paragraphs'
    if domain=='code' and row['record_license'] not in LICENSES:return 'HOLD','record_license_unresolved'
    if row['split'] not in ('train','validation','test'):return 'HOLD','split_unassigned'
    return 'CANDIDATE','rule_pass_not_quality_approval'

def layer(row):
    n=row['own_text_tokens'];return ('short' if n<=512 else 'medium' if n<=4096 else 'long')+'/'+row['domain']+'/'+row['language']

def open_store(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);d=sqlite3.connect(root/'stock.sqlite',timeout=60);d.row_factory=sqlite3.Row
    d.execute('PRAGMA journal_mode=WAL');d.execute('PRAGMA cache_size=-65536')
    d.executescript('''
    CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY,path TEXT UNIQUE,sha TEXT,source TEXT,next_pos INTEGER DEFAULT 0,next_byte INTEGER DEFAULT 0,state TEXT DEFAULT 'pending',stats TEXT DEFAULT '{}');
    CREATE TABLE IF NOT EXISTS records(h TEXT PRIMARY KEY,fid INTEGER,pos INTEGER,byte INTEGER,domain TEXT,lang TEXT,family TEXT,split TEXT,n INTEGER,rank TEXT,stratum TEXT,gate TEXT,job TEXT,initial_gate TEXT);
    CREATE INDEX IF NOT EXISTS records_queue ON records(gate,domain,stratum,rank);
    CREATE TABLE IF NOT EXISTS jobs(name TEXT PRIMARY KEY,source TEXT,mode TEXT,state TEXT DEFAULT 'planned',reason TEXT);
    CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT);
    ''');d.commit();return d

class RawReader:
    def __init__(self):self.cache=OrderedDict()
    def row(self,file,pos,offset):
        path=Path(file['path'])
        if path.suffix=='.jsonl':
            with path.open('rb') as f:f.seek(offset);return json.loads(f.readline())
        import pyarrow.parquet as pq
        if str(path) not in self.cache:
            pf=pq.ParquetFile(path);starts=[];total=0
            for i in range(pf.num_row_groups):starts.append(total);total+=pf.metadata.row_group(i).num_rows
            self.cache[str(path)]=(pf,starts,None,None)
            if len(self.cache)>2:self.cache.popitem(last=False)
        from bisect import bisect_right
        pf,starts,g,rows=self.cache[str(path)];want=bisect_right(starts,pos)-1
        if g!=want:
            rows=pf.read_row_group(want,columns=[k for k in FIELDS if k in pf.schema_arrow.names]).to_pylist();g=want
            self.cache[str(path)]=(pf,starts,g,rows)
        return rows[pos-starts[want]]

class AllStockControl(SupplyControl):
    """Register verified reserve batches after domain quota; original stage/release gates unchanged."""
    def register(self,domain,input_binding):
        if domain not in self.recipe['targets_by_domain'] or not input_binding:raise ValueError('known domain/frozen binding required')
        import time
        bid=digest(dict(domain=domain,input_binding=input_binding,recipe=self.recipe_sha))
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            if self.db.execute('SELECT 1 FROM batches WHERE batch_id=?',(bid,)).fetchone():return bid
            pending=self.db.execute("SELECT COUNT(*) FROM batches WHERE status NOT IN ('ready','quarantined','legacy_internal')").fetchone()[0]
            if pending>=self.recipe['maximum_unreleased_batches']:raise ValueError('backpressure')
            self.db.execute('INSERT INTO batches(batch_id,domain,input_binding,created) VALUES (?,?,?,?)',(bid,domain,json.dumps(input_binding,sort_keys=True),time.time()))
        return bid
    def status(self):
        s=super().status();s['all_stock_cleaning_not_stopped_by_goal_quota']=True
        targets=self.recipe['targets_by_domain'];s['recipe_admitted_train_body_tokens']=sum(min(targets[d],s['accounted_by_domain'][d]) for d in targets)
        s['reserve_by_domain']={d:max(0,s['accounted_by_domain'][d]-targets[d]) for d in targets}
        s['balanced_training_manifest_frozen']=False
        s['recipe_credit_scope']='per-domain goal credit cap; not a newly frozen balanced 20B training manifest'
        return s
