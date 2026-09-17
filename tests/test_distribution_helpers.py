import numpy as np
import pytest

from gl_dq.checks.distribution import LogScale, VarSpec, inverse_transform, percentile_probs, psi


def test_percentile_probs_int_and_custom():
    p = percentile_probs(20)
    assert p[0] == 0 and p[-1] == 1 and len(p) == 22 and 0.99 in p  # 21-point grid + p99
    p10 = percentile_probs(10)
    assert 0.99 in p10 and len(p10) == 12
    assert percentile_probs([0.9, 0.5, 0.5]) == [0.0, 0.5, 0.9, 0.99, 1.0]


def test_invalid_bins():
    with pytest.raises(ValueError):
        VarSpec(name="x", percentile_bins=1)
    with pytest.raises(ValueError):
        VarSpec(name="x", percentile_bins=[1.5])


@pytest.mark.parametrize("method,values", [("log1p", [0.0, 5.0, 1e6]), ("signed_log", [-1e5, -1.0, 0.0, 3.0]),
                                           ("log10", [0.1, 10.0]), ("none", [-3.0, 3.0])])
def test_inverse_transform_roundtrip(method, values):
    x = np.array(values)
    fwd = {"log1p": np.log1p, "log10": np.log10, "signed_log": lambda v: np.sign(v) * np.log1p(np.abs(v)),
           "none": lambda v: v}[method](x)
    np.testing.assert_allclose(inverse_transform(fwd, method), x, rtol=1e-9, atol=1e-9)


def test_log_histogram_handles_zero_and_negative(ctx_injected):
    chk = ctx_injected.make_check("distribution")
    spec = VarSpec(name="expo_amt", group_by=["src"], log_scale={"method": "log10"}, hist_bins=20)
    h = chk.histogram(spec)
    assert h.attrs["excluded"] > 0  # zeros and negatives are outside log10's domain
    assert h["bucket"].between(0, 19).all()
    h2 = chk.histogram(spec.model_copy(update={"log_scale": LogScale(method="signed_log")}))
    assert h2.attrs["excluded"] == 0


def test_psi():
    assert psi(np.array([10, 10]), np.array([10, 10])) == pytest.approx(0)
    assert psi(np.array([90, 10]), np.array([10, 90])) > 1
