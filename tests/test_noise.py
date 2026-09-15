"""The noise model is the scientific core: its statistics must be right."""

import numpy as np
import pytest

from denoiseraw.noise.calibrate import calibrate_from_bias, calibrate_from_flats, fit_tukey_lambda
from denoiseraw.noise.estimate import estimate_profile
from denoiseraw.noise.profile import NoiseProfile, ProfileBank, tukey_lambda_std
from denoiseraw.noise.synth import sample_profile, sample_tukey_lambda, synthesize_noise
from denoiseraw.rawio.loader import RawImage
from denoiseraw.rawio.packing import unpack_bayer

MODELS = ["gaussian", "poisson_gaussian", "eld"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("level", [0.0, 0.01, 0.1, 0.5])
def test_synthesised_noise_has_the_requested_variance(model, level):
    """Every model must reproduce Var = a*x + b to within sampling error."""
    profile = NoiseProfile(a=2e-4, b=9e-6, sigma_row=1.5e-3, tukey_lambda=0.1,
                           quantization_step=1 / 16383)
    clean = np.full((4, 256, 256), level, np.float32)
    noisy = synthesize_noise(clean, profile, model=model, rng=np.random.default_rng(0))
    expected = np.sqrt(profile.a * level + profile.b)
    assert noisy.std() == pytest.approx(expected, rel=0.06)
    assert noisy.mean() == pytest.approx(level, abs=5e-4)


def test_row_noise_is_shared_by_the_planes_of_one_sensor_row():
    """R and G1 come from the same physical row and must share the offset;
    R and G2 come from different rows and must not."""
    profile = NoiseProfile(a=0.0, b=1e-4, sigma_row=9e-3, tukey_lambda=0.14)
    noisy = synthesize_noise(np.zeros((4, 128, 128), np.float32), profile,
                             model="eld", rng=np.random.default_rng(0))
    row_means = noisy.mean(axis=2)
    assert np.corrcoef(row_means[0], row_means[1])[0, 1] > 0.9     # R vs G1
    assert abs(np.corrcoef(row_means[0], row_means[2])[0, 1]) < 0.2  # R vs G2


def test_below_black_values_are_not_clamped():
    """Real RAW goes below black; clamping bakes a colour cast into shadows."""
    profile = NoiseProfile(a=3e-4, b=0.0, sigma_row=0.0, quantization_step=0.0)
    clean = np.full((4, 8, 8), -0.05, np.float32)
    noisy = synthesize_noise(clean, profile, model="poisson_gaussian",
                             rng=np.random.default_rng(0))
    assert np.allclose(noisy, -0.05)


@pytest.mark.parametrize("lam", [-0.3, 0.0, 0.14, 0.5])
def test_tukey_lambda_scale_is_the_actual_standard_deviation(lam):
    samples = sample_tukey_lambda((400_000,), lam, 0.01, rng=np.random.default_rng(0))
    assert samples.std() == pytest.approx(0.01, rel=0.05)
    assert samples.mean() == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("lam", [-0.3, 0.0, 0.14, 0.5])
def test_tukey_lambda_std_matches_monte_carlo(lam):
    from scipy.stats import tukeylambda
    mc = tukeylambda.rvs(lam, size=200_000, random_state=0).std()
    assert tukey_lambda_std(lam) == pytest.approx(mc, rel=0.05)


def _packed_as_raw(packed, white=16383.0):
    return RawImage(data=unpack_bayer(packed, "RGGB"), pattern="RGGB",
                    black_level=0.0, white_level=white)


def test_calibration_recovers_the_parameters():
    truth = NoiseProfile(a=3e-4, b=1.6e-5, sigma_row=2.5e-3, tukey_lambda=0.0,
                         quantization_step=1 / 16383)
    rng = np.random.default_rng(7)

    def frame(level):
        clean = np.full((4, 512, 512), level, np.float32)
        return _packed_as_raw(synthesize_noise(clean, truth, model="eld", rng=rng))

    b, sigma_row, _ = calibrate_from_bias([frame(0.0) for _ in range(3)])
    assert b == pytest.approx(truth.b, rel=0.2)
    assert sigma_row == pytest.approx(truth.sigma_row, rel=0.2)

    pairs = [(frame(v), frame(v)) for v in (0.02, 0.06, 0.15, 0.3, 0.55, 0.85)]
    assert calibrate_from_flats(pairs, b=b) == pytest.approx(truth.a, rel=0.1)


def test_fit_tukey_lambda_identifies_the_shape():
    for lam in (-0.2, 0.0, 0.3):
        samples = sample_tukey_lambda((100_000,), lam, 1.0, rng=np.random.default_rng(1))
        assert fit_tukey_lambda(samples) == pytest.approx(lam, abs=0.12)


@pytest.mark.parametrize("a,b", [(3e-4, 1.6e-5), (1e-3, 6e-5), (5e-5, 2e-6)])
def test_blind_estimation_is_in_the_right_ballpark(a, b):
    """Blind estimation is approximate by nature; what matters is that the
    sigma it reports at a typical signal level is close enough to drive the
    denoiser, and that it errs towards *over*-estimating rather than under."""
    from fixtures import make_raw_image

    profile = NoiseProfile(a=a, b=b, sigma_row=1e-3, quantization_step=1 / 15871)
    img, _, _ = make_raw_image(h=512, w=512, profile=profile, seed=3)
    est = estimate_profile(img)
    true_sigma = np.sqrt(a * 0.1 + b)
    assert est.sigma_at(0.1) == pytest.approx(true_sigma, rel=0.35)
    assert est.a > 0 and est.b > 0


@pytest.mark.parametrize("row", [0.0, 1.2e-3, 3e-3])
def test_blind_estimation_recovers_row_noise(row):
    """Banding is recovered to roughly 25%, with a small positive floor from the
    scene's own horizontal structure. Over-estimating is the safe direction: it
    costs a little smoothing, whereas under-estimating leaves visible stripes."""
    from fixtures import make_raw_image

    profile = NoiseProfile(a=3e-4, b=1.6e-5, sigma_row=row, quantization_step=1 / 15871)
    img, _, _ = make_raw_image(h=512, w=512, profile=profile, seed=3)
    estimated = estimate_profile(img).sigma_row
    assert estimated == pytest.approx(row, abs=6e-4, rel=0.25)
    assert estimated >= row - 3e-4


def test_profile_scaling_and_serialisation(tmp_path):
    profile = NoiseProfile(a=1e-4, b=4e-6, sigma_row=1e-4, iso=1600, camera_model="X")
    scaled = profile.scaled(4.0)
    assert scaled.a == pytest.approx(4e-4)      # shot noise scales linearly
    assert scaled.b == pytest.approx(6.4e-5)    # read noise quadratically

    path = tmp_path / "p.json"
    profile.save(str(path))
    assert NoiseProfile.load(str(path)).to_dict() == profile.to_dict()


def test_profile_bank_interpolates_by_iso(tmp_path):
    bank = ProfileBank("X")
    bank.add(NoiseProfile(a=1e-4, b=4e-6, iso=1600))
    bank.add(NoiseProfile(a=2e-4, b=1e-5, iso=3200))
    at_6400 = bank.at_iso(6400)
    assert at_6400.a == pytest.approx(4e-4)     # 2x the 3200 profile
    assert at_6400.iso == 6400

    path = tmp_path / "bank.json"
    bank.save(str(path))
    assert set(ProfileBank.load(str(path)).profiles) == set(bank.profiles)


def test_sample_profile_stays_physical():
    base = NoiseProfile(a=2e-4, b=9e-6, sigma_row=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = sample_profile(base, (100, 25600), rng=rng)
        assert p.a > 0 and p.b > 0 and -0.4 <= p.tukey_lambda <= 0.6
