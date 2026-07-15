"""Small exact-layout demo. Run from the repository root after installing requirements."""
from mscl import And, Default, Obj, Relation, SampleSearch, Spec, TypePred


spec = Spec(
    objects=[
        Obj("table", "existing", "dining table", box=(400, 350, 250, 180)),
        Obj("plant", "new", "potted plant"),
    ],
    formula=And([
        TypePred("plant", "potted plant"),
        Default("plant"),
        Relation("cleft", ["plant", "table"], 20),
    ]),
)

result = SampleSearch().sample(spec, seed=7)
print("layout:", result.layout)
print(f"solver checking: {result.stats.solver_time_s:.4f}s "
      f"({result.stats.solver_checks} checks)")
print(f"preference:      {result.stats.preference_time_s:.4f}s")
print(f"verification:    {result.stats.verification_time_s:.4f}s")
print(f"total:           {result.stats.total_time_s:.4f}s")
print(f"rejected/backtracks: {result.stats.rejected_proposals}/{result.stats.backtracks}")
