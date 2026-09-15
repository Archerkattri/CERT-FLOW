# CF1–CF2 witness and sensing packet

## Implemented seam

`certflow.witness` adds three deliberately narrow pieces:

- `RouteWitness` binds the incumbent upper bound, optimistic lower bound,
  route, epoch, selection rule, observation identities and risk scope.
- `RiskLedger` is an append-only finite alpha-spending record with deterministic
  digest and hard over-spend rejection.
- `select_witness_observations` is a deterministic cost-aware acquisition
  heuristic with an explicit challenger reserve.

These objects make selection history and risk expenditure inspectable. They do
not turn a heuristic allocator into a conformal theorem. A production CF1
guarantee still needs a proof for the declared sequential construction and a
lower-bound witness covering every competing route (or a valid exclusion rule).

## Verification

The unit tests cover finite validation, lower/upper ordering, risk over-spend
rejection, deterministic selection and challenger reservation. The packet is
therefore a method-contract pass, not a new end-to-end efficiency result.

## Research decision

Keep the current planner as the sound reference. Integrate this seam only after
the independent calibration block and held-out route-optimum evaluation are
locked. If the restricted graph theorem cannot be completed, publish the
explicit witness/risk ledger as an auditability improvement and retain the
planner's existing marginal guarantee without claiming post-selection coverage.
