"""Validate distribution hashes, local documentation links and obvious secret/path patterns."""
import hashlib,json,re
from pathlib import Path
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'evidence/bundle_manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        p=ROOT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected: raise ValueError('Changed/missing bound file: '+name)
    patterns=[r'(?i)[A-Z]:[\\/]+Users[\\/]',r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'-----BEGIN [A-Z ]*PRIVATE KEY-----',r'AKIA[A-Z0-9]{16}']
    links=0
    for name in manifest['files']:
        p=ROOT/name
        if p.suffix in ('.json','.md','.py','.txt','.yml'):
            body=p.read_text(encoding='utf-8')
            if any(re.search(pattern,body) for pattern in patterns): raise ValueError('Sensitive pattern in '+name)
        if p.suffix=='.md':
            text=re.sub(r'```.*?```','',body,flags=re.S)
            for target in re.findall(r'!?\[[^\]\n]*\]\((<[^>]+>|[^)]+)\)',text):
                target=target.strip('<>')
                if target.startswith(('https://','http://','#','mailto:')):continue
                target=unquote(target.split('#')[0]);target=re.sub(r':\d+$','',target)
                if not (p.parent/target).resolve().is_file(): raise ValueError('Broken link in '+name+': '+target)
                links+=1
    print(json.dumps({'status':'passed_distribution_integrity','bound_files':len(manifest['files']),'local_links_checked':links,'secret_scan':'listed known-token and absolute-user-path patterns only, not an exhaustive security audit','scope':'portable distribution, not historical scientific confirmation'},indent=2))
if __name__=='__main__':main()
