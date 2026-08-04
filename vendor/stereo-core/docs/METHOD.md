# Method

## Motivation

RGB-D addresses local geometric observability, while a decoder MoE increases capacity for
heterogeneous local action roles. Local-ARCA showed that unconstrained role routing can become
nearly uniform and semantically unstable. Stereo-CoRE therefore couples routing decisions to the
experts' measured action capability instead of task labels or inaccessible team state.

## Capability coupling

For the same ground-truth action chunk, each expert is evaluated counterfactually. If expert `e`
has action error `E_e`, the capability teacher forms a soft target `q_cap(e)` favouring low-error
experts. The router distribution `p_router(e|o_i,q_i,query_k)` is optimized with
`KL(q_cap || p_router)`. Inputs remain strictly local at training and deployment.

## Information boundary

No task/agent identity, language, communication, global view, peer view/action, or privileged
team state is used by Stereo-CoRE. Synchronized-team relation and capability-anchor variants are
released only as ablations.
