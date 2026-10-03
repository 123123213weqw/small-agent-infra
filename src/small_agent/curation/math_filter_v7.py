"""Conservative structural candidate filter; not a semantic correctness judge."""
import re,collections
from urllib.parse import urlsplit
TERMS=re.compile(r'\b(?:equation|theorem|proof|algebra|geometry|geometric|calculus|derivative|integral|matrix|matrices|vector|probability|polynomial|function|fraction|number|integer|triangle|circle|angle|limit|series|sum|product|distribution|variance|mean|statistic|logarithm|ratio|factor|prime|root|coefficient|solution|solve|value|variable)\b',re.I)
BOILER=re.compile(r'(?:log in|sign in|register now|posting permissions|all rights reserved|cookies policy|privacy policy|find all school-related|join chat room|share on facebook|browse all|view all .*papers|related questions|sponsored links|skip to navigation|advertisement)',re.I)
LOCKED=re.compile(r'(?:unlock (?:the |this |full |complete )*(?:answer|solution)|subscribe to (?:view|unlock|access)|purchase (?:the |this |full )*(?:answer|solution)|answer preview|full solution (?:is |only )?available (?:to|for) subscribers)',re.I)

def structural_filter(row,config):
    try:u=urlsplit(row['source_locator']);host=(u.hostname or '').lower()
    except ValueError:return 'malformed_source_locator_hold'
    if u.scheme not in ('http','https') or host not in config['allowed_exact_hosts']:return 'host_outside_conservative_subset_hold'
    if '/search' in u.path.lower() or '/tagged/' in u.path.lower() or '/tags/' in u.path.lower():return 'directory_search_page_hold'
    if row['language'].lower()!=config['required_language']:return 'language_hold'
    if row['own_text_tokens']<config['minimum_body_tokens']:return 'tiny_fragment_hold'
    text=row['text']
    if len(text.encode('utf-8'))>config['maximum_body_bytes']:return 'oversized_body_hold'
    if '\ufffd' in text or '\x00' in text:return 'replacement_or_null_character_hold'
    if LOCKED.search(text):return 'explicit_locked_answer_preview_hold'
    paragraphs=[x.strip() for x in re.split(r'\n\s*\n',text) if x.strip()]
    boiler=sum(len(x) for x in paragraphs if BOILER.search(x))
    if boiler/max(1,len(text))>config['maximum_boilerplate_fraction']:return 'boilerplate_dominant_hold'
    counts=collections.Counter(x for x in paragraphs if len(x)>=80)
    repeated=sum(len(x)*(n-1) for x,n in counts.items() if n>1)
    if repeated/max(1,len(text))>config['maximum_repeated_paragraph_fraction']:return 'repeated_paragraphs_hold'
    if len(set(x.lower() for x in TERMS.findall(text)))<config['minimum_math_terms']:return 'insufficient_math_content_signals_hold'
    return 'structural_candidate_pass_not_quality_approval'
