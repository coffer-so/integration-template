//! Marginal price `d(user_output) / d(amount_in)` for the weighted curve.
//!
//! With `V_in`, `V_out` the virtual balances, `w_in / w_out` the weights,
//! `phi` the swap-fee fraction and `x' = (1 - phi) x`, the contract's curve is
//!
//! ```text
//! f(x) = V_out * (1 - (V_in / (V_in + x'))^r),   r = w_in / w_out
//! ```
//!
//! whose derivative is
//!
//! ```text
//! f'(x) = (1 - phi) * V_out * r * V_in^r * (V_in + x')^(-r-1)
//!       = (1 - phi) * V_out * r * (V_in / (V_in + x'))^r / (V_in + x')
//! ```
//!
//! This is positive and strictly decreasing in `x` (the curve is concave).
//!
//! When the input token's sell-off window is configured with a surge curve,
//! the user receives `f(x) - S(x)` where `S` charges the output produced above
//! the window threshold at the (piecewise-linear) surge rate. In the
//! continuous model `dS/dx = rate(fill(x)) * f'(x)`, where `fill(x)` is the
//! window position `(effective_before + x) / cap` — the window advances by the
//! GROSS input. So the marginal user rate is `f'(x) * (1 - rate(fill(x)))`,
//! which stays positive and non-increasing (both factors are non-increasing).
//!
//! What this does NOT capture is the contract's discretisation of `S`: it
//! splits the taxed span into `SURGE_FEE_SEGMENTS` pieces and charges each
//! piece's real output at that piece's average rate, ceiling per piece. That
//! over-collects relative to the exact integral (Chebyshev, see the contract),
//! by an amount that varies with `x`, so in the surge regime `price` is the
//! derivative of the continuous model, not of the exact integer function. The
//! fixture tests measure the residual against the shipped MVT tolerance.

use crate::coffer::constants::{PERCENT_SCALE, SWAP_FEE_PRECISION, WEIGHT_SCALE};
use crate::coffer::state::AssetConfig;

/// Everything the closed-form derivative needs, all in `f64`.
#[derive(Debug, Clone, Copy)]
pub struct CurveParams {
    pub virtual_balance_in: f64,
    pub virtual_balance_out: f64,
    /// `w_in / w_out`.
    pub weight_ratio: f64,
    /// `1 - swap_fee_rate / SWAP_FEE_PRECISION`.
    pub keep: f64,
}

impl CurveParams {
    pub fn new(
        virtual_balance_in: u64,
        weight_in: u64,
        virtual_balance_out: u64,
        weight_out: u64,
        swap_fee_rate: u32,
    ) -> Self {
        let _ = WEIGHT_SCALE; // weights enter only as a ratio
        Self {
            virtual_balance_in: virtual_balance_in as f64,
            virtual_balance_out: virtual_balance_out as f64,
            weight_ratio: weight_in as f64 / weight_out as f64,
            keep: 1.0 - swap_fee_rate as f64 / SWAP_FEE_PRECISION as f64,
        }
    }

    /// `f'(amount_in)` of the fee-then-curve output, output atoms per input atom.
    pub fn marginal_output(&self, amount_in: u64) -> f64 {
        let effective_in = self.keep * amount_in as f64;
        let denom = self.virtual_balance_in + effective_in;
        let base = self.virtual_balance_in / denom;
        self.keep * self.virtual_balance_out * self.weight_ratio * base.powf(self.weight_ratio)
            / denom
    }
}

/// Instantaneous surge rate (fraction of output) at window position
/// `effective_selloff` against `cap`, for the token's configured curve —
/// the continuous version of `surge_fee::rate_at` (no unit rounding).
/// Returns 0 when the surge is disabled or the position is at/below the
/// threshold; saturates at 100% fill.
pub fn surge_rate(cfg: &AssetConfig, effective_selloff: u64, cap: u64) -> f64 {
    if cap == 0 || cfg.variable_fee_slope_high_pct == 0 {
        return 0.0;
    }
    let scale = PERCENT_SCALE as f64;
    let thr = cfg.variable_fee_threshold_pct as f64;
    if thr >= scale {
        return 0.0;
    }
    let fill = (effective_selloff as f64 * scale / cap as f64).min(scale);
    if fill <= thr {
        return 0.0;
    }
    // Normalised position within the taxed span, t in (0, 1].
    let t = ((fill - thr) / (scale - thr)).min(1.0);
    let low = cfg.variable_fee_slope_low_pct as f64;
    let mid = cfg.variable_fee_slope_mid_pct as f64;
    let high = cfg.variable_fee_slope_high_pct as f64;

    // Mirror `kink_t`: a kink that would not split the span means one line.
    let kink_fill = cfg.variable_fee_kink_pct as f64 * (scale / 100.0);
    let k = if cfg.variable_fee_kink_pct == 0 || kink_fill <= thr || kink_fill >= scale {
        0.0
    } else {
        (kink_fill - thr) / (scale - thr)
    };

    let rate_pct = if k == 0.0 {
        low + (high - low).max(0.0) * t
    } else if t <= k {
        low + (mid - low).max(0.0) * (t / k)
    } else {
        mid + (high - mid).max(0.0) * ((t - k) / (1.0 - k))
    };
    (rate_pct / scale).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coffer::math::CofferMath;

    fn params() -> CurveParams {
        CurveParams::new(5_000_000_000_000, 8_000, 30_000_000_000_000, 2_000, 3_000)
    }

    /// The closed form must agree with a central difference of the exact
    /// integer curve (fee included).
    #[test]
    fn marginal_output_matches_finite_difference_of_the_ported_curve() {
        let p = params();
        for &x in &[
            10_000_000u64,
            1_000_000_000,
            50_000_000_000,
            2_000_000_000_000,
        ] {
            let analytic = p.marginal_output(x);
            let h = (x / 1000).max(100_000);
            let f = |amt: u64| {
                let fee = crate::coffer::swap::calculate_swap_fee(amt, 3_000).unwrap();
                CofferMath::calc_out_given_in(
                    5_000_000_000_000,
                    8_000,
                    30_000_000_000_000,
                    2_000,
                    amt - fee,
                    u64::MAX,
                    9,
                    9,
                )
                .unwrap() as f64
            };
            let fd = (f(x + h) - f(x - h)) / (2.0 * h as f64);
            let rel = (analytic - fd).abs() / fd;
            assert!(rel < 1e-3, "x={x}: analytic={analytic} fd={fd} rel={rel}");
        }
    }

    #[test]
    fn marginal_output_is_positive_and_decreasing() {
        let p = params();
        let mut prev = f64::INFINITY;
        for &x in &[0u64, 1, 1_000, 1_000_000, 1_000_000_000, 1_000_000_000_000] {
            let v = p.marginal_output(x);
            assert!(v > 0.0 && v <= prev, "x={x}: {v} vs {prev}");
            prev = v;
        }
    }

    #[test]
    fn surge_rate_follows_the_three_point_curve() {
        let cfg = AssetConfig {
            variable_fee_threshold_pct: 2_000,
            variable_fee_slope_low_pct: 100,
            variable_fee_slope_mid_pct: 500,
            variable_fee_slope_high_pct: 5_000,
            variable_fee_kink_pct: 60,
            ..Default::default()
        };
        let cap = 10_000;
        assert_eq!(surge_rate(&cfg, 2_000, cap), 0.0);
        assert!((surge_rate(&cfg, 2_001, cap) - 0.01).abs() < 1e-4);
        assert!((surge_rate(&cfg, 6_000, cap) - 0.05).abs() < 1e-9);
        assert!((surge_rate(&cfg, 10_000, cap) - 0.5).abs() < 1e-9);
        assert!((surge_rate(&cfg, 20_000, cap) - 0.5).abs() < 1e-9);
        // no kink: straight line low → high
        let flat = AssetConfig {
            variable_fee_kink_pct: 0,
            ..cfg
        };
        assert!((surge_rate(&flat, 6_000, cap) - 0.255).abs() < 1e-9);
        // disabled curve
        let off = AssetConfig {
            variable_fee_slope_high_pct: 0,
            ..cfg
        };
        assert_eq!(surge_rate(&off, 9_000, cap), 0.0);
    }
}
