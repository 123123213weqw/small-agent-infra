"""Full-stock routing and READY-only quotas; neither rules nor a ledger approve quality."""
import collections
import re
from urllib.parse import urlsplit
from .supply_control import SupplyControl
from .math_filter_v7 import structural_filter as existing_math_filter

BOILER = re.compile(r'(?:log in|sign in|register now|posting permissions|all rights reserved|cookies policy|privacy policy|sponsored links|advertisement|skip to navigation)', re.I)
LOCKED = re.compile(r'(?:unlock (?:the |this |full |complete )*(?:answer|solution)|subscribe to (?:view|unlock|access)|purchase (?:the |this |full )*(?:answer|solution)|answer preview|full solution (?:is |only )?available (?:to|for) subscribers)', re.I)
MATH_BUCKETS = ('math_academic', 'math_reference', 'math_discussion', 'math_web')


def length_layer(n, language):
    return ('short' if n <= 512 else 'medium' if n <= 4096 else 'long') + '/' + language.lower()


def route(row, config):
    """Never rewrite a body, run untrusted code, or make KEEP from an unknown scope."""
    text = row['text']; domain = row['domain']; language = row['language'].lower()
    if row['split'] not in ('train', 'validation', 'test'):
        return 'QUARANTINE', 'unassigned_split', domain
    if not text.strip() or row['own_text_tokens'] < config['minimum_body_tokens_by_domain'].get(domain,128):
        return 'DROP', 'empty_or_tiny', domain
    if '\x00' in text or '\ufffd' in text:
        return 'DROP', 'null_or_replacement', domain
    if len(text.encode('utf-8')) > config['maximum_body_bytes']:
        return 'QUARANTINE', 'whole_body_teacher_context_limit', domain
    if domain == 'code':
        if row['record_license'].lower() not in config['code_licenses'] or row['policy_bucket'] != 'record_license_whitelisted':
            return 'QUARANTINE', 'record_license_unresolved', 'code_' + language
        # A parse failure is not automatically bad pretraining text. No code execution.
        if language == 'python':
            return 'DELEGATED', 'python_existing_calibrated_processor', 'code_python'
        return 'QUARANTINE', 'independent_language_reference_required', 'code_' + language
    if domain != 'math' or language != 'en':
        return 'QUARANTINE', 'independent_domain_reference_required', domain + '_' + language
    try:
        u = urlsplit(row['source_locator']); host = (u.hostname or '').lower()
    except ValueError:
        return 'QUARANTINE', 'malformed_locator', 'math_web'
    if u.scheme not in ('http', 'https') or not host:
        return 'QUARANTINE', 'malformed_locator', 'math_web'
    if re.search(r'/(?:search|tags|tagged)(?:/|$)', u.path, re.I):
        return 'DROP', 'directory_or_search', 'math_web'
    if LOCKED.search(text):
        return 'DROP', 'explicit_locked_preview', 'math_web'
    parts = [s.strip() for s in re.split(r'\n\s*\n', text) if s.strip()]
    if sum(len(s) for s in parts if BOILER.search(s)) / max(1, len(text)) > config['maximum_boilerplate_fraction']:
        return 'DROP', 'boilerplate_dominant', 'math_web'
    repeated = collections.Counter(s for s in parts if len(s) >= 80)
    if sum(len(s) * (n-1) for s, n in repeated.items() if n > 1) / max(1, len(text)) > config['maximum_repeated_paragraph_fraction']:
        return 'DROP', 'repeated_paragraphs', 'math_web'
    if host in config['existing_v8_exact_hosts'] and existing_math_filter(row,config['existing_v8_filter'])=='structural_candidate_pass_not_quality_approval':
        return 'DELEGATED', 'existing_v8_scope_only_not_site_approval', 'math_existing_v8'
    if host.endswith(('.edu', '.edu.cn', '.edu.au', '.ac.uk', '.ac.jp')):
        bucket = 'math_academic'
    elif host == 'wikipedia.org' or host.endswith('.wikipedia.org') or host == 'wikibooks.org' or host.endswith('.wikibooks.org'):
        bucket = 'math_reference'
    elif host == 'stackexchange.com' or host.endswith('.stackexchange.com') or '/forum/' in u.path.lower():
        bucket = 'math_discussion'
    else:
        bucket = 'math_web'
    return 'PENDING_CALIBRATION', 'whole_body_rule_candidate_not_quality_approval', bucket


def quota_status(recipe, ready, legacy, candidates=None):
    targets = recipe['targets_by_domain']
    if sum(targets.values()) != 20000000000:
        raise ValueError('Frozen 20B mixture required')
    if any(v < 0 for v in list(ready.values()) + list(legacy.values())):
        raise ValueError('Negative counters')
    gaps = {d: max(0, n-ready.get(d, 0)) for d, n in targets.items()}
    # Keep oversupply in a reserve bank. Neither legacy nor replay counts as new READY.
    admitted = {d: min(n, ready.get(d, 0)) for d, n in targets.items()}
    return dict(target_train_body_tokens=sum(targets.values()), target_by_domain=targets,
                target_weights={d: n/sum(targets.values()) for d, n in targets.items()},
                ready_by_domain={d: ready.get(d, 0) for d in targets},
                legacy_uncertified_by_domain={d: legacy.get(d, 0) for d in targets},
                candidate_by_domain=candidates or {}, remaining_ready_by_domain=gaps,
                recipe_admitted_unique_train_body_tokens=sum(admitted.values()),
                reserve_by_domain={d: max(0, ready.get(d, 0)-n) for d, n in targets.items()},
                target_met=not any(gaps.values()), missing_domains=[d for d in targets if not ready.get(d, 0)],
                missing_domain_action='WAIT_REPLENISHMENT_NEVER_SILENT_RENORMALIZE',
                counted_scope='unique_READY_train_body_tokens_without_EOS_or_legacy_or_replay')


def priorities(recipe, ready, available):
    """Least-filled quota first, deterministic ties; math abundance never raises its weight."""
    return sorted((d for d in available if d in recipe['targets_by_domain'] and ready.get(d, 0) < recipe['targets_by_domain'][d]),
                  key=lambda d: (ready.get(d, 0)/recipe['targets_by_domain'][d], d))


def loss_corrected_sample_weights(targets,loss_tokens,samples):
    """Expected loss-token mixture, not document-count or observed-stock mixture."""
    if any(loss_tokens.get(d,0)<=0 or samples.get(d,0)<=0 for d in targets):
        raise ValueError('Missing target domain; never silently renormalize')
    means={d:loss_tokens[d]/samples[d] for d in targets}
    return {d:targets[d]/means[d] for d in targets},means


class ReadyQuotaControl(SupplyControl):
    """Same shared ledger identity, but inherited legacy is not new goal completion."""
    def status(self):
        s=super().status()
        s['remaining_by_domain']={d:max(0,n-s['ready_by_domain'].get(d,0)) for d,n in self.recipe['targets_by_domain'].items()}
        s['target_met']=not any(s['remaining_by_domain'].values())
        s['quota_scope']='new_READY_only_legacy_still_preserved_separately'
        return s
