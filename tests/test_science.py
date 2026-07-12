from __future__ import annotations

import copy
import inspect
import math
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import conformal as C
import dataset as D
from interfaces import ForecastInputs, inputs_from_bundle
from mechanistic_ode import MechanisticODE, MeasurementModel, integrate
import priors as P
import synthetic
import tcn_baseline as TCN
import train as GB


class GuardrailTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = D.load()
        cls.roles = D.grouped_role_split(cls.data, seed=0)

    def test_g1_training_dispatches_continuous_adjoint(self):
        cfg = GB.TrainConfig(epochs=1, patience=1, dynamics="physics_only")
        with mock.patch.object(GB, "odeint_adjoint", wraps=GB.odeint_adjoint) as spy:
            result = GB.train(self.roles["train"], self.roles["validation"], cfg)
        self.assertGreater(spy.call_count, 0)
        self.assertTrue(result["grads_all_finite"])

    def test_g1_adjoint_gradient_matches_direct(self):
        torch.manual_seed(5)
        core_direct = MechanisticODE.identifiable(dynamics="graybox")
        core_adjoint = copy.deepcopy(core_direct)
        optics_direct = MeasurementModel()
        optics_adjoint = copy.deepcopy(optics_direct)
        z0 = torch.tensor([[5.0, 5.0, 0.0], [5.0, 10.0, 0.0]], requires_grad=True)
        time = torch.tensor([0.0, 30.0, 60.0, 900.0])

        _, direct = integrate(
            core_direct, optics_direct, z0, time,
            use_adjoint=False, rtol=1e-6, atol=1e-8,
        )
        direct[-1].sum().backward()
        direct_grad = torch.cat([
            p.grad.flatten() for p in core_direct.residual.parameters() if p.grad is not None
        ])

        z0_adjoint = z0.detach().clone().requires_grad_(True)
        _, adjoint = integrate(
            core_adjoint, optics_adjoint, z0_adjoint, time,
            use_adjoint=True, rtol=1e-6, atol=1e-8,
        )
        adjoint[-1].sum().backward()
        adjoint_grad = torch.cat([
            p.grad.flatten() for p in core_adjoint.residual.parameters() if p.grad is not None
        ])
        relative = float(torch.linalg.vector_norm(direct_grad - adjoint_grad) /
                         torch.linalg.vector_norm(direct_grad).clamp_min(1e-12))
        cosine = float(torch.nn.functional.cosine_similarity(direct_grad, adjoint_grad, dim=0))
        self.assertLess(relative, 0.01)
        self.assertGreater(cosine, 0.999)

    def test_g2_washburn_is_inverse_L_and_analytic(self):
        core = MechanisticODE.identifiable(dynamics="physics_only")
        z0 = torch.tensor([5.0, 5.0, 0.0])
        time = torch.linspace(0.0, 900.0, 20)
        state, _ = integrate(core, MeasurementModel(), z0, time, use_adjoint=False)
        expected = z0[0] ** 2 + 2.0 * core.k_wash.detach() * time
        self.assertTrue(torch.allclose(state[:, 0] ** 2, expected, rtol=2e-5, atol=2e-5))
        derivative = core(torch.tensor(0.0), z0)[0]
        self.assertAlmostEqual(float(z0[0] * derivative), float(core.k_wash), places=6)

    def test_g3_positive_L0_enforced(self):
        core = MechanisticODE.identifiable(dynamics="physics_only")
        with self.assertRaises(ValueError):
            integrate(core, MeasurementModel(), torch.tensor([0.0, 1.0, 0.0]),
                      torch.tensor([0.0, 1.0]), use_adjoint=False)
        model = GB.LatentODEForecaster(GB.TrainConfig())
        inputs = inputs_from_bundle(self.roles["test"])
        self.assertTrue(torch.all(model.initial_state(inputs)[:, 0] == 5.0))

    def test_g4_residual_only_changes_conservative_chemistry_flux(self):
        physics = MechanisticODE.identifiable(dynamics="physics_only")
        gray = MechanisticODE.identifiable(dynamics="graybox")
        gray.load_state_dict(physics.state_dict(), strict=True)
        final = [m for m in gray.residual.modules() if isinstance(m, torch.nn.Linear)][-1]
        with torch.no_grad():
            final.bias.copy_(torch.tensor([0.5, -0.25]))
        z = torch.tensor([5.0, 6.0, 10.0])
        p = physics(torch.tensor(20.0), z)
        g = gray(torch.tensor(20.0), z)
        self.assertEqual(float(p[0].detach()), float(g[0].detach()))
        self.assertFalse(torch.allclose(p[1:], g[1:]))
        # Both vector fields obey the same exact mass-balance identity.
        for core, dz in ((physics, p), (gray, g)):
            volume = core.volume_mL(z[0])
            dvolume = core.dvolume_dt_mL_s(z[0], dz[0])
            mass_rate = volume * dz[1] + z[1] * dvolume + (
                core.capture_area_m2 * P.MW_HCG_G_PER_MOL * dz[2]
            )
            self.assertAlmostEqual(float(mass_rate), 0.0, places=6)

    def test_g5_optics_separate_from_vector_field(self):
        core = MechanisticODE.identifiable(dynamics="graybox")
        z = torch.tensor([5.0, 6.0, 10.0])
        before = core(torch.tensor(5.0), z).detach().clone()
        optics = MeasurementModel()
        with torch.no_grad():
            optics.alpha.mul_(100.0)
            optics.beta.add_(100.0)
        after = core(torch.tensor(5.0), z).detach()
        self.assertTrue(torch.equal(before, after))

    def test_g6_frozen_priors_and_tamper_rejection(self):
        model = GB.LatentODEForecaster(GB.TrainConfig())
        model.validate_science()
        for name in ("log_k_on", "log_k_off", "log_B_max"):
            self.assertFalse(getattr(model.core, name).requires_grad)
        result = {
            "model": model,
            "cfg": model.cfg,
            "best_validation": 1.0,
            "best_epoch": 0,
            "epochs_run": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            GB.save_checkpoint(result, path)
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["state_dict"]["core.log_k_on"] += math.log(2.0)
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError):
                GB.load_checkpoint(path)

    def test_g7_roles_have_zero_day_and_lot_overlap(self):
        D.assert_no_leakage(*self.roles.values())
        self.assertEqual(sum(map(len, self.roles.values())), len(self.data["I_obs"]))
        alternate = D.grouped_role_split(self.data, seed=1)
        self.assertNotEqual(self.roles["test"].groups(), alternate["test"].groups())

    def test_mass_bounds_over_operating_extremes(self):
        core = MechanisticODE.identifiable(dynamics="graybox")
        core.set_covariate_tensors(torch.tensor([18.0, 42.0]), torch.tensor([25.0, 85.0]))
        z0 = torch.tensor([[5.0, 0.0, 0.0], [5.0, P.mIU_per_mL_to_ng_per_mL(125.0), 0.0]])
        time = torch.linspace(0.0, 900.0, 50)
        state, _ = integrate(core, MeasurementModel(), z0, time, use_adjoint=False)
        self.assertGreaterEqual(float(state[..., 1].min()), -1e-5)
        self.assertGreaterEqual(float(state[..., 2].min()), -1e-5)
        self.assertLessEqual(float(state[..., 2].max()), float(core.B_max) + 1e-4)
        mass = core.total_analyte_ng(state)
        self.assertLess(float((mass - mass[0]).abs().max()), 2e-4)

    def test_solver_tolerance_convergence(self):
        core = MechanisticODE.identifiable(dynamics="graybox")
        optics = MeasurementModel()
        z0 = torch.tensor([5.0, P.mIU_per_mL_to_ng_per_mL(125.0), 0.0])
        time = torch.tensor([0.0, 900.0])
        _, normal = integrate(core, optics, z0, time, use_adjoint=False, rtol=1e-5, atol=1e-7)
        _, tight = integrate(core, optics, z0, time, use_adjoint=False, rtol=1e-8, atol=1e-10)
        self.assertLess(float(torch.abs(normal[-1] - tight[-1])), 0.01)


class InformationAndConformalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = D.load()
        cls.roles = D.grouped_role_split(cls.data, seed=0)

    def test_model_api_excludes_answer_metadata(self):
        fields = set(ForecastInputs.__dataclass_fields__)
        self.assertEqual(fields, {"t", "I_early", "T_ambient", "RH"})
        self.assertNotIn("C_f0", inspect.signature(GB.LatentODEForecaster.predict).parameters)

    def test_trained_forecast_ignores_late_window_and_uses_early_trace(self):
        result = GB.train(
            self.roles["train"], self.roles["validation"],
            GB.TrainConfig(epochs=8, patience=8, dynamics="physics_only"),
        )
        proof = GB.information_invariance(result["model"], self.roles["test"])
        self.assertEqual(proof["late_window_delta"], 0.0)
        self.assertGreater(proof["early_window_delta"], 1e-6)

    def test_tcn_capacity_is_matched(self):
        gray = GB.LatentODEForecaster(GB.TrainConfig())
        tcn = TCN.TCNForecaster()
        difference = abs(GB.count_trainable(gray) - TCN.count_trainable(tcn))
        self.assertLessEqual(difference / GB.count_trainable(gray), 0.01)

    def test_finite_sample_quantile_edge(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertTrue(math.isinf(C.scp_quantile(np.arange(8.0), 0.1)))
        self.assertEqual(C.scp_quantile(np.arange(9.0), 0.1), 8.0)

    def test_conformal_fails_closed_on_bad_arrays(self):
        for values in ([], [1.0, np.nan], [[1.0, 2.0]]):
            with self.assertRaises(ValueError):
                C.scp_quantile(values)
        with self.assertRaises(ValueError):
            C.nonconformity_scores(np.ones(2), np.ones(3))

    def test_test_outcomes_cannot_change_intervals(self):
        calibration = C.fit_scp(np.zeros(20), np.arange(20.0))
        prediction = np.linspace(0.0, 1.0, 7)
        before = calibration.interval(prediction)
        arbitrary_hidden_targets = np.full(7, 1e9)
        del arbitrary_hidden_targets
        after = calibration.interval(prediction)
        self.assertTrue(np.array_equal(before[0], after[0]))
        self.assertTrue(np.array_equal(before[1], after[1]))
        self.assertNotIn("test_scores", inspect.signature(C.fit_iwcp).parameters)
        self.assertNotIn("true", inspect.signature(C.IWCPCalibrator.interval).parameters)

    def test_uniform_weight_iwcp_equals_scp(self):
        rng = np.random.default_rng(4)
        scores = np.abs(rng.normal(size=40))
        scp = C.scp_quantile(scores)
        weighted = C.weighted_quantiles(scores, np.ones(40), np.ones(10))
        self.assertTrue(np.allclose(weighted, scp))

    def test_synthetic_zero_mismatch_is_exact_null(self):
        common = dict(n_groups=2, reps_per_level=1, n_t=19, group_effect_scale=0.0, seed=77)
        none = synthetic.generate(synthetic.SyntheticConfig(
            mismatch_mechanism="none", mismatch_scale=1.0, **common
        ))
        zero = synthetic.generate(synthetic.SyntheticConfig(
            mismatch_mechanism="combined", mismatch_scale=0.0, **common
        ))
        self.assertTrue(torch.equal(none["I_clean"], zero["I_clean"]))

    def test_current_early_trace_has_recoverable_signal(self):
        between, within, ratio = synthetic.early_window_separation(self.data)
        self.assertGreater(ratio, 3.0, (between, within, ratio))

    def test_capture_raw_npy_frames_enter_analysis_loader(self):
        import data_pipeline as pipeline

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            first = np.full((20, 30), 1000, dtype=np.uint16)
            second = np.full((20, 30), 1000, dtype=np.uint16)
            first[8:12, 10:20] = 800
            second[8:12, 10:20] = 700
            first[2:6, 2:7] = 600
            second[2:6, 2:7] = 600
            np.save(folder / "frame_000000_0000000000ms.npy", first)
            np.save(folder / "frame_000001_0000001000ms.npy", second)
            samples = pipeline.load_frames(folder)
            self.assertEqual([sample.t_ms for sample in samples], [0.0, 1000.0])
            trace = pipeline.extract_intensity_timeseries(
                samples, (10, 8, 10, 4), grey_roi=(2, 2, 5, 4),
                smooth_window=3, smooth_polyorder=1,
            )
            self.assertEqual(len(trace), 2)
            self.assertGreater(trace.I_raw.iloc[0], trace.I_raw.iloc[1])


if __name__ == "__main__":
    unittest.main()
