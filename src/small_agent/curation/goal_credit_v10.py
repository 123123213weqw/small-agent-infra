"""Explicit goal-credit policy: inherited eligible internal data + new READY, not raw stock."""
from .supply_control import SupplyControl


def goal_balance(recipe, ready, legacy, candidates=None):
    targets=recipe['targets_by_domain']
    if sum(targets.values())!=20000000000 or any(v<0 for v in [*ready.values(),*legacy.values()]):
        raise ValueError('Positive 20B goal and nonnegative counters required')
    accounted={d:ready.get(d,0)+legacy.get(d,0) for d in targets}
    gaps={d:max(0,n-accounted[d]) for d,n in targets.items()}
    return dict(target_train_body_tokens=sum(targets.values()),target_by_domain=targets,
                target_weights={d:n/sum(targets.values()) for d,n in targets.items()},
                ready_by_domain={d:ready.get(d,0) for d in targets},
                legacy_inherited_internal_by_domain={d:legacy.get(d,0) for d in targets},
                accounted_by_domain=accounted,remaining_goal_by_domain=gaps,
                unique_goal_credited_train_body_tokens=sum(accounted.values()),
                recipe_admitted_unique_train_body_tokens=sum(min(n,accounted[d]) for d,n in targets.items()),
                reserve_by_domain={d:max(0,accounted[d]-n) for d,n in targets.items()},
                target_met=not any(gaps.values()),candidate_by_domain=candidates or {},
                missing_goal_domains=[d for d in targets if not accounted[d]],
                missing_new_READY_domains=[d for d in targets if not ready.get(d,0)],
                missing_domain_action='WAIT_REPLENISHMENT_NEVER_SILENT_RENORMALIZE',
                counted_scope='unique_train_body_inherited_internal_plus_new_READY_without_EOS_heldout_raw_or_replay',
                legacy_quality_recertified=False,legacy_distribution_eligible=False,
                legacy_and_READY_are_disjoint_ledger_categories=True)


class GoalQuotaControl(SupplyControl):
    """Original verified ledger quota semantics, preserving inherited quality labels."""
    def status(self):
        s=super().status()
        s['quota_scope']='inherited_internal_plus_new_READY_per_explicit_user_policy_v10'
        s['legacy_quality_recertified']=False
        return s
