//! Coffer / Cube DEX venue (program
//! `8iQtGj9mcUfFUGaiCpPy89swC3s8YTC8FhVZWfgeZhwu`).
//!
//! A coffer pool is a 2..=10-token weighted AMM with virtual liquidity (Balancer
//! V2 style), a ceil-rounded swap fee with a protocol cut, a per-token
//! sliding-window sell-off cap and a surge fee that rises as that window
//! fills. All of that math lives in [`crate::coffer`], a verbatim port of the
//! contract; this module is the Titan glue: account loading, the trait
//! implementation, the closed-form marginal price and the swap instruction.
//!
//! Quote semantics (see [`TradingVenue::quote`] for the contract):
//! - every guard the on-chain handler runs is run here in the same order;
//! - `amount == 0` returns zero output and the spot price;
//! - a swap the sell-off window would reject (`MaxSelloffExceeded`) is
//!   reported as a partial fill: `not_enough_liquidity = true`, `amount` = the
//!   largest gross input the window still admits, `expected_output` = the
//!   output at that size;
//! - a swap whose curve output exceeds the LP-owned balance
//!   (`AmountOutExceedsBalance`) is likewise a partial fill, sized by a binary
//!   search over the exact quote;
//! - `ZeroFeeAmount` (unreachable on `main`: the fee rounds UP) and config /
//!   arithmetic failures are errors.
//!
//! State freshness: nothing a quote reads survives `update_state`. The pool
//! account (virtual/actual balances, weights, fee rates, kill switches, the
//! sell-off caps and surge curves, the window accumulators and snapshot), the
//! mints and the Clock sysvar are re-read on every refresh and the per-direction
//! curve parameters are rebuilt from them; the only thing fixed at creation is
//! the token set, which the contract cannot change either. The pool's admin
//! and range-manager roles may move virtual balances, weights, caps and curves
//! at any time between two refreshes — see `directions_num` for why the
//! declared directions do not depend on any of that.

pub mod instruction;
pub mod price;

use async_trait::async_trait;
use solana_account::Account;
use solana_instruction::Instruction;
use solana_pubkey::Pubkey;
use solana_sysvar::clock::{self, Clock};
use spl_associated_token_account::get_associated_token_address_with_program_id;

use crate::{
    account_caching::AccountsCache,
    coffer::{
        constants::MAX_TOKENS,
        errors::ErrorCode,
        state::CofferPool,
        swap::{SwapOutcome, quote_exact_in, selloff_headroom},
    },
    coffer_venue::{
        instruction::{INITIALIZE_POOL_DISCRIMINATOR, SwapAccounts, swap_instruction},
        price::{CurveParams, surge_rate},
    },
    trading_venue::{
        AddressLookupTableTrait, FromAccount, QuoteRequest, QuoteResult, SwapType, TradingVenue,
        error::{ErrorInfo, TradingVenueError},
        protocol::PoolProtocol,
        token_info::TokenInfo,
        venue_creation::{ParsedInstruction, PoolCreation},
    },
};

/// Mainnet program id of the Coffer program.
pub const COFFER_PROGRAM_ID: Pubkey =
    Pubkey::from_str_const("8iQtGj9mcUfFUGaiCpPy89swC3s8YTC8FhVZWfgeZhwu");

/// Fixed accounts of `initialize_pool` before the per-token mints:
/// config, pool, bpt_mint, payer, token_program, associated_token_program,
/// system_program. The mints follow as `remaining_accounts`.
const INIT_POOL_FIXED_ACCOUNTS: usize = 7;
const INIT_POOL_ACCOUNT_INDEX: usize = 1;

/// Detect every coffer pool created by a confirmed transaction.
///
/// On `main`, `initialize_pool(normalized_weights, initial_virtual_balances,
/// swap_fee_rate, pool_id, banned_extensions_override)` takes the token set
/// purely from `remaining_accounts` (one mint per token, in slot order), so the
/// mints are `accounts[7..]` and the new pool PDA is `accounts[1]`. The
/// Borsh-encoded `normalized_weights` length is cross-checked against the
/// number of trailing accounts when the data parses.
pub fn parse_pool_creations(instructions: &[ParsedInstruction]) -> Vec<PoolCreation> {
    instructions
        .iter()
        .filter(|ix| ix.program_id == COFFER_PROGRAM_ID)
        .filter(|ix| ix.data.len() >= 8 && ix.data[..8] == INITIALIZE_POOL_DISCRIMINATOR)
        .filter_map(|ix| {
            let pool = *ix.accounts.get(INIT_POOL_ACCOUNT_INDEX)?;
            let mints = ix.accounts.get(INIT_POOL_FIXED_ACCOUNTS..)?;
            if mints.len() < 2 || mints.len() > MAX_TOKENS {
                return None;
            }
            // Defensive cross-check: `normalized_weights: Vec<u64>` is the
            // first argument, so its length is the token count.
            if ix.data.len() >= 12 {
                let n = u32::from_le_bytes(ix.data[8..12].try_into().ok()?) as usize;
                if n != mints.len() {
                    return None;
                }
            }
            Some(PoolCreation {
                protocol: PoolProtocol::Coffer,
                pool,
                mints: mints.to_vec(),
            })
        })
        .collect()
}

/// Per-direction constants precomputed in `update_state` so `quote` only
/// does arithmetic.
#[derive(Debug, Clone, Copy, Default)]
struct DirectionParams {
    curve: Option<CurveParams>,
    /// Upper end of the quotable domain when the input token's surge fee is
    /// live: the input at which the user's NET output peaks. Beyond it the
    /// contract's segmented surge charge grows faster than the curve output
    /// (a larger input pays fewer atoms), so requests past it are partial
    /// fills at the peak — Titan's `f` must be non-decreasing and `price`
    /// positive on the reported domain. `None` when the direction has no
    /// surge (net output = curve output, monotone by construction).
    fill_limit: Option<u64>,
}

/// Off-chain state of one coffer pool.
///
/// The refresh-sensitive fields (`pool`, `now`, the per-direction
/// parameters derived from them) are private: they are only ever written
/// together by `update_state`, so a quote can never see a pool from one
/// refresh with curve parameters from another.
#[derive(Clone)]
pub struct CofferVenue {
    /// The pool account address.
    pub pool_key: Pubkey,
    /// Decoded pool account (refreshed by `update_state`).
    pool: CofferPool,
    /// `Clock::unix_timestamp` the pool was last refreshed against; feeds the
    /// sell-off window arithmetic exactly as the program's `Clock::get()`.
    now: i64,
    /// Current epoch (for Token-2022 transfer-fee schedules).
    epoch: u64,
    /// One entry per active slot, in slot order (index == on-chain token index).
    token_info: Vec<TokenInfo>,
    /// Mint decimals per slot.
    decimals: [u8; MAX_TOKENS],
    /// `[in][out]` closed-form price parameters.
    directions: [[DirectionParams; MAX_TOKENS]; MAX_TOKENS],
    /// Accounts `update_state` needs: pool, every mint, the clock sysvar.
    required_state_pubkeys: Vec<Pubkey>,
    initialized: bool,
}

impl CofferVenue {
    /// The decoded pool, as of the last `update_state`.
    pub fn pool(&self) -> &CofferPool {
        &self.pool
    }

    /// `Clock::unix_timestamp` of the last `update_state` — the clock every
    /// quote's window arithmetic runs at.
    pub fn now(&self) -> i64 {
        self.now
    }

    /// Epoch of the last `update_state`.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    /// The quotable domain limit of a direction (see `DirectionParams`):
    /// `None` when the direction is not surge-limited.
    pub fn fill_limit(&self, in_idx: u8, out_idx: u8) -> Option<u64> {
        self.directions[in_idx as usize][out_idx as usize].fill_limit
    }

    /// Slot index of `mint` among the active tokens.
    fn index_of(&self, mint: &Pubkey) -> Option<u8> {
        self.pool
            .active_slots()
            .iter()
            .position(|slot| slot.config.mint == *mint)
            .and_then(|i| u8::try_from(i).ok())
    }

    /// Vault (ATA of the pool PDA) for slot `i`.
    pub fn vault(&self, i: usize) -> Pubkey {
        let cfg = &self.pool.tokens[i].config;
        CofferPool::derive_vault(&self.pool_key, &cfg.mint, &cfg.token_program)
    }

    fn decode_pool(pubkey: &Pubkey, account: &Account) -> Result<CofferPool, TradingVenueError> {
        if account.owner != COFFER_PROGRAM_ID {
            return Err(TradingVenueError::FromAccountError(
                "account is not owned by the Coffer program".into(),
            ));
        }
        CofferPool::from_account_data(&account.data)
            .map_err(|e| TradingVenueError::DeserializationFailed(format!("{pubkey}: {e}").into()))
    }

    /// Rebuild every per-direction parameter from the freshly decoded pool,
    /// clock and decimals (called last in `update_state`).
    fn rebuild_directions(&mut self) {
        let n = self.pool.token_count as usize;
        for i in 0..n.min(MAX_TOKENS) {
            for j in 0..n.min(MAX_TOKENS) {
                let (ci, cj) = (&self.pool.tokens[i], &self.pool.tokens[j]);
                let curve = (i != j && cj.config.normalized_weight > 0).then(|| {
                    CurveParams::new(
                        ci.dynamics.virtual_balance,
                        ci.config.normalized_weight,
                        cj.dynamics.virtual_balance,
                        cj.config.normalized_weight,
                        self.pool.swap_fee_rate,
                    )
                });
                self.directions[i][j] = DirectionParams {
                    curve,
                    fill_limit: None,
                };
            }
        }
        // The surge-limited directions: the input token has a live cap AND a
        // surge curve, and the pool can trade. The search runs the exact
        // quote ~150 times per such direction, once per refresh.
        if !self.pool.pool_enabled || !self.pool.swaps_enabled {
            return;
        }
        for i in 0..n.min(MAX_TOKENS) {
            let cfg = &self.pool.tokens[i].config;
            if cfg.max_selloff_pct == 0 || cfg.variable_fee_slope_high_pct == 0 || !cfg.is_active {
                continue;
            }
            let Ok(in_idx) = u8::try_from(i) else {
                continue;
            };
            let headroom = match selloff_headroom(&self.pool, in_idx, self.now) {
                Ok(Some(h)) => h,
                _ => continue,
            };
            for j in 0..n.min(MAX_TOKENS) {
                let Ok(out_idx) = u8::try_from(j) else {
                    continue;
                };
                if i == j || self.directions[i][j].curve.is_none() {
                    continue;
                }
                // Window / LP-balance limit, then where the rate hits 100%
                // (every further atom buys nothing: price would be 0), then
                // the net-output peak.
                let hi = self.largest_fillable(in_idx, out_idx, headroom);
                let hi = self.surge_exhaustion_point(in_idx, hi);
                self.directions[i][j].fill_limit = Some(self.net_output_peak(in_idx, out_idx, hi));
            }
        }
    }

    /// Largest input `x <= hi` whose post-swap window position keeps the surge
    /// rate below 100% (the rate is non-decreasing in the fill).
    fn surge_exhaustion_point(&self, in_idx: u8, hi: u64) -> u64 {
        let cfg = &self.pool.tokens[in_idx as usize].config;
        let saturated = |x: u64| match self.selloff_after(in_idx, x) {
            Ok(Some((effective, cap))) => surge_rate(cfg, effective, cap) >= 1.0,
            Ok(None) => false,
            Err(_) => true,
        };
        if hi == 0 || !saturated(hi) {
            return hi;
        }
        let (mut lo, mut hi) = (0u64, hi);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if saturated(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        lo
    }

    /// The input `x <= hi` at which the user's net output (curve output minus
    /// the surge fee) is largest.
    ///
    /// In the continuous model the net marginal rate `f'(x)·(1 - rate)` is
    /// never negative, but the contract charges the surge on 4 segments of
    /// the taxed span at each segment's average rate (quantised to 0.01%),
    /// re-partitioned for every request size: past some point the charge
    /// grows faster than the curve pays out, and a larger input receives
    /// fewer atoms. The net function is unimodal up to a sawtooth of at most
    /// ~1e-4 of the output (one rate quantum on a segment's output), so a
    /// ternary search lands on the plateau around the maximum, and the limit
    /// is the far end of that plateau (see below).
    fn net_output_peak(&self, in_idx: u8, out_idx: u8, hi: u64) -> u64 {
        let net = |x: u64| {
            self.exact(in_idx, out_idx, x)
                .map(|o| o.amount_out_user)
                .unwrap_or(0)
        };
        let hi_limit = hi;
        let (mut lo, mut hi) = (0u64, hi);
        while hi - lo > 2 {
            let third = (hi - lo) / 3;
            let (m1, m2) = (lo + third, hi - third);
            if net(m1) < net(m2) {
                lo = m1;
            } else {
                hi = m2;
            }
        }
        let mut best = (lo, net(lo));
        for x in lo + 1..=hi {
            let v = net(x);
            if v > best.1 {
                best = (x, v);
            }
        }
        // The sawtooth makes the top a plateau, not a point: extend the limit
        // to the largest input whose net output is within two rate quanta
        // (2e-4) of the maximum, so a curve too mild to bend the net output
        // keeps the whole window as its domain, while a real decline (0.3% at
        // 95/5 with a 0 → 25% curve) still ends it at the peak.
        let (peak, max_net) = best;
        let tolerance = max_net / 5_000;
        let within = |x: u64| net(x).saturating_add(tolerance) >= max_net;
        if within(hi_limit) {
            return hi_limit;
        }
        let (mut lo, mut hi) = (peak, hi_limit);
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if within(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// Marginal user output per input atom at `amount_in` (see `price.rs`).
    /// `effective_selloff` is the window position AFTER this swap, if capped.
    fn marginal_price(
        &self,
        in_idx: u8,
        out_idx: u8,
        amount_in: u64,
        selloff: Option<(u64, u64)>,
    ) -> Result<f64, TradingVenueError> {
        let curve = self.directions[in_idx as usize][out_idx as usize]
            .curve
            .ok_or(TradingVenueError::MathError(
                "direction has no curve".into(),
            ))?;
        let mut price = curve.marginal_output(amount_in);
        if let Some((effective_selloff, cap)) = selloff {
            let cfg = &self.pool.tokens[in_idx as usize].config;
            price *= 1.0 - surge_rate(cfg, effective_selloff, cap);
        }
        if !price.is_finite() || price < 0.0 {
            return Err(TradingVenueError::MathError(
                "non-finite marginal price".into(),
            ));
        }
        Ok(price)
    }

    /// The window position/cap a quote of `amount_in` would land on, if the
    /// input token is capped. `None` when uncapped.
    fn selloff_after(
        &self,
        in_idx: u8,
        amount_in: u64,
    ) -> Result<Option<(u64, u64)>, TradingVenueError> {
        let cfg = &self.pool.tokens[in_idx as usize].config;
        if cfg.max_selloff_pct == 0 {
            return Ok(None);
        }
        let mut dynamics = self.pool.tokens[in_idx as usize].dynamics;
        let vb = dynamics.virtual_balance;
        match crate::coffer::math::max_selloff::check_and_advance(
            &mut dynamics,
            cfg.max_selloff_pct as u64,
            cfg.max_selloff_period_length,
            amount_in,
            vb,
            self.now,
        ) {
            Ok(Some(r)) => Ok(Some((r.effective_selloff, r.max_selloff_cap))),
            Ok(None) => Ok(None),
            // Beyond the cap: the rate saturates at full fill.
            Err(ErrorCode::MaxSelloffExceeded) => Ok(Some((u64::MAX, 1))),
            Err(e) => Err(map_error(e)),
        }
    }

    fn exact(&self, in_idx: u8, out_idx: u8, amount_in: u64) -> Result<SwapOutcome, ErrorCode> {
        quote_exact_in(
            &self.pool,
            amount_in,
            in_idx,
            out_idx,
            self.decimals[in_idx as usize],
            self.decimals[out_idx as usize],
            self.now,
        )
    }

    /// Largest `x <= hi` for which the exact quote succeeds (the failure
    /// predicate — curve output over the LP balance, or the window cap — is
    /// monotone in `x`). Returns 0 when nothing fits.
    fn largest_fillable(&self, in_idx: u8, out_idx: u8, hi: u64) -> u64 {
        if hi == 0 || self.exact(in_idx, out_idx, hi).is_ok() {
            return hi;
        }
        let (mut lo, mut hi) = (0u64, hi); // lo always fillable (0 = nothing), hi not
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            if self.exact(in_idx, out_idx, mid).is_ok() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// The single place a fill is sized: the largest input `<= hi` the program
    /// accepts (window cap, LP balance, overflow), capped by the direction's
    /// net-output peak. Every partial-fill path — beyond the window, beyond
    /// the LP balance, beyond the peak — goes through here, so the reported
    /// amount is the same whichever limit the request tripped first.
    fn fillable(&self, in_idx: u8, out_idx: u8, hi: u64) -> u64 {
        let amount = self.largest_fillable(in_idx, out_idx, hi);
        match self.directions[in_idx as usize][out_idx as usize].fill_limit {
            Some(limit) => amount.min(limit),
            None => amount,
        }
    }

    /// A partial fill of at most `hi` gross input atoms (see `fillable`).
    fn partial_fill(
        &self,
        request: &QuoteRequest,
        in_idx: u8,
        out_idx: u8,
        hi: u64,
    ) -> Result<QuoteResult, TradingVenueError> {
        let amount = self.fillable(in_idx, out_idx, hi);
        let (expected_output, price) = if amount == 0 {
            (
                0,
                self.marginal_price(in_idx, out_idx, 0, self.selloff_after(in_idx, 0)?)?,
            )
        } else {
            let o = self.exact(in_idx, out_idx, amount).map_err(map_error)?;
            let selloff = o
                .max_selloff_result
                .map(|r| (r.effective_selloff, r.max_selloff_cap));
            (
                o.amount_out_user,
                self.marginal_price(in_idx, out_idx, amount, selloff)?,
            )
        };
        Ok(QuoteResult {
            input_mint: request.input_mint,
            output_mint: request.output_mint,
            amount,
            expected_output,
            not_enough_liquidity: true,
            price,
        })
    }
}

/// Map a contract error to the template's error model. Pool-level switches
/// are `InactivePoolError`; everything else keeps the contract's variant
/// name as static context (no allocation).
fn map_error(e: ErrorCode) -> TradingVenueError {
    match e {
        ErrorCode::MathOverflow | ErrorCode::MathUnderflow | ErrorCode::DivisionByZero => {
            TradingVenueError::MathError(ErrorInfo::StaticStr(e.name()))
        }
        other => TradingVenueError::AmmMethodError(ErrorInfo::StaticStr(other.name())),
    }
}

impl FromAccount for CofferVenue {
    fn from_account(pubkey: &Pubkey, account: &Account) -> Result<Self, TradingVenueError> {
        let pool = Self::decode_pool(pubkey, account)?;
        if pool.token_count < 2 || pool.token_count as usize > MAX_TOKENS {
            return Err(TradingVenueError::FromAccountError(
                "coffer pool has an invalid token_count".into(),
            ));
        }
        let mut required_state_pubkeys = Vec::with_capacity(pool.token_count as usize + 2);
        required_state_pubkeys.push(*pubkey);
        required_state_pubkeys.extend(pool.active_slots().iter().map(|s| s.config.mint));
        required_state_pubkeys.push(clock::ID);
        Ok(Self {
            pool_key: *pubkey,
            pool,
            now: 0,
            epoch: 0,
            token_info: Vec::new(),
            decimals: [0; MAX_TOKENS],
            directions: [[DirectionParams::default(); MAX_TOKENS]; MAX_TOKENS],
            required_state_pubkeys,
            initialized: false,
        })
    }
}

#[async_trait]
impl TradingVenue for CofferVenue {
    fn initialized(&self) -> bool {
        self.initialized
    }

    fn program_id(&self) -> Pubkey {
        COFFER_PROGRAM_ID
    }

    fn program_dependencies(&self) -> Vec<Pubkey> {
        // The swap CPIs into whichever token program each side uses.
        vec![COFFER_PROGRAM_ID, spl_token::ID, spl_token_2022::ID]
    }

    fn market_id(&self) -> Pubkey {
        self.pool_key
    }

    fn tradable_mints(&self) -> Result<Vec<Pubkey>, TradingVenueError> {
        Ok(self
            .pool
            .active_slots()
            .iter()
            .map(|s| s.config.mint)
            .collect())
    }

    fn get_token_info(&self) -> &[TokenInfo] {
        &self.token_info
    }

    fn protocol(&self) -> PoolProtocol {
        PoolProtocol::Coffer
    }

    /// Every ordered pair of the pool's token slots — a STRUCTURAL
    /// declaration, independent of the pool's mutable state.
    ///
    /// The token set is fixed at creation, so this list never changes for a
    /// given pool. Everything that decides whether a pair can trade RIGHT NOW
    /// is mutable and is answered by `quote` instead:
    /// - a deactivated input token (`set_token_active`, an admin kill switch
    ///   that can be flipped back) makes every sell of it error with
    ///   `TokenInactive`; buying it keeps working;
    /// - a capped input token whose sell-off window has no headroom quotes as
    ///   a partial fill of 0 atoms until the window rotates (minutes);
    /// - a sidelined output token (`actual_balance == 0`, refilled by the next
    ///   `add_liquidity`) makes every quote a partial fill of 0 atoms;
    /// - a disabled pool / disabled swaps error with `InactivePoolError`.
    ///
    /// Titan documents this method as the venue's *declared* directions and
    /// may evaluate it once when the venue is registered rather than after
    /// every `update_state`. Filtering on transient state here would then
    /// silently and permanently drop a direction that happened to be paused
    /// at registration (a full window, a token deactivated for an hour, an
    /// unseeded slot), while declaring a direction that cannot trade at the
    /// moment costs the router nothing: `bounds` reports no quotable range and
    /// `quote` reports zero fillable atoms or the contract's error.
    fn directions_num(&self) -> Vec<(u8, u8)> {
        let n = self.pool.active_slots().len();
        let mut out = Vec::with_capacity(n * n.saturating_sub(1));
        for i in 0..n {
            for j in 0..n {
                if i != j
                    && let (Ok(a), Ok(b)) = (u8::try_from(i), u8::try_from(j))
                {
                    out.push((a, b));
                }
            }
        }
        out
    }

    fn get_required_pubkeys_for_update(&self) -> Result<Vec<Pubkey>, TradingVenueError> {
        Ok(self.required_state_pubkeys.clone())
    }

    async fn update_state(&mut self, cache: &dyn AccountsCache) -> Result<(), TradingVenueError> {
        let keys = self.required_state_pubkeys.clone();
        let accounts = cache.get_accounts(&keys).await?;
        if accounts.len() != keys.len() {
            return Err(TradingVenueError::FailedToFetchMultipleAccountData);
        }

        let pool_account = accounts[0]
            .as_ref()
            .ok_or(TradingVenueError::NoAccountFound(self.pool_key.into()))?;
        let pool = Self::decode_pool(&self.pool_key, pool_account)?;
        if pool.token_count != self.pool.token_count
            || pool
                .active_slots()
                .iter()
                .zip(self.pool.active_slots())
                .any(|(a, b)| a.config.mint != b.config.mint)
        {
            // The token set is fixed at creation; a change means this is not
            // the pool we were built from.
            return Err(TradingVenueError::MissingState(
                "coffer pool token set changed under the venue".into(),
            ));
        }

        let clock_account = accounts[keys.len() - 1]
            .as_ref()
            .ok_or(TradingVenueError::NoAccountFound(clock::ID.into()))?;
        let clock: Clock = bincode::deserialize(&clock_account.data)
            .map_err(|_| TradingVenueError::DeserializationFailed(clock::ID.into()))?;

        let mut token_info = Vec::with_capacity(pool.token_count as usize);
        let mut decimals = [0u8; MAX_TOKENS];
        for (i, slot) in pool.active_slots().iter().enumerate() {
            let mint_key = slot.config.mint;
            let mint_account = accounts[1 + i]
                .as_ref()
                .ok_or(TradingVenueError::NoAccountFound(mint_key.into()))?;
            if mint_account.owner != slot.config.token_program {
                return Err(TradingVenueError::InvalidMint(mint_key.into()));
            }
            let info = TokenInfo::new(&mint_key, mint_account, clock.epoch)?;
            decimals[i] = u8::try_from(info.decimals)
                .map_err(|_| TradingVenueError::DataConversionError(mint_key.into()))?;
            token_info.push(info);
        }

        self.pool = pool;
        self.now = clock.unix_timestamp;
        self.epoch = clock.epoch;
        self.token_info = token_info;
        self.decimals = decimals;
        self.rebuild_directions();
        self.initialized = true;
        Ok(())
    }

    fn quote(&self, request: QuoteRequest) -> Result<QuoteResult, TradingVenueError> {
        if !self.initialized {
            return Err(TradingVenueError::NotInitialized(
                "call update_state before quoting".into(),
            ));
        }
        if request.swap_type != SwapType::ExactIn {
            return Err(TradingVenueError::ExactOutNotSupported);
        }
        let in_idx = self
            .index_of(&request.input_mint)
            .ok_or(TradingVenueError::InvalidMint(request.input_mint.into()))?;
        let out_idx = self
            .index_of(&request.output_mint)
            .ok_or(TradingVenueError::InvalidMint(request.output_mint.into()))?;
        if in_idx == out_idx {
            return Err(TradingVenueError::InvalidMint(request.output_mint.into()));
        }

        // The handler's pool-level guards, before anything else.
        if !self.pool.pool_enabled || !self.pool.swaps_enabled {
            return Err(TradingVenueError::InactivePoolError(
                self.pool_key,
                PoolProtocol::Coffer,
            ));
        }
        if !self.pool.tokens[in_idx as usize].config.is_active {
            return Err(TradingVenueError::AmmMethodError(ErrorInfo::StaticStr(
                ErrorCode::TokenInactive.name(),
            )));
        }

        if request.amount == 0 {
            // Zero output at the spot price f'(0), with the surge rate at the
            // window's current position folded in.
            let selloff = self.selloff_after(in_idx, 0)?;
            return Ok(QuoteResult {
                input_mint: request.input_mint,
                output_mint: request.output_mint,
                amount: 0,
                expected_output: 0,
                not_enough_liquidity: false,
                price: self.marginal_price(in_idx, out_idx, 0, selloff)?,
            });
        }

        // Past the direction's net-output peak: a partial fill at the peak.
        if self.directions[in_idx as usize][out_idx as usize]
            .fill_limit
            .is_some_and(|limit| request.amount > limit)
        {
            return self.partial_fill(&request, in_idx, out_idx, request.amount);
        }

        match self.exact(in_idx, out_idx, request.amount) {
            Ok(o) => {
                let selloff = o
                    .max_selloff_result
                    .map(|r| (r.effective_selloff, r.max_selloff_cap));
                Ok(QuoteResult {
                    input_mint: request.input_mint,
                    output_mint: request.output_mint,
                    amount: request.amount,
                    expected_output: o.amount_out_user,
                    not_enough_liquidity: false,
                    price: self.marginal_price(in_idx, out_idx, request.amount, selloff)?,
                })
            }
            Err(ErrorCode::MaxSelloffExceeded) => {
                let headroom = selloff_headroom(&self.pool, in_idx, self.now)
                    .map_err(map_error)?
                    .unwrap_or(0);
                self.partial_fill(&request, in_idx, out_idx, headroom.min(request.amount))
            }
            Err(ErrorCode::AmountOutExceedsBalance) | Err(ErrorCode::InsufficientLiquidity) => {
                self.partial_fill(&request, in_idx, out_idx, request.amount)
            }
            Err(ErrorCode::PoolDisabled) | Err(ErrorCode::SwapsDisabled) => Err(
                TradingVenueError::InactivePoolError(self.pool_key, PoolProtocol::Coffer),
            ),
            Err(e) => Err(map_error(e)),
        }
    }

    fn generate_swap_instruction(
        &self,
        request: QuoteRequest,
        user: Pubkey,
    ) -> Result<Instruction, TradingVenueError> {
        if request.swap_type != SwapType::ExactIn {
            return Err(TradingVenueError::ExactOutNotSupported);
        }
        let in_idx = self
            .index_of(&request.input_mint)
            .ok_or(TradingVenueError::InvalidMint(request.input_mint.into()))?;
        let out_idx = self
            .index_of(&request.output_mint)
            .ok_or(TradingVenueError::InvalidMint(request.output_mint.into()))?;
        let cfg_in = &self.pool.tokens[in_idx as usize].config;
        let cfg_out = &self.pool.tokens[out_idx as usize].config;

        let accounts = SwapAccounts {
            pool: self.pool_key,
            token_mint_in: cfg_in.mint,
            token_mint_out: cfg_out.mint,
            user_token_account_in: get_associated_token_address_with_program_id(
                &user,
                &cfg_in.mint,
                &cfg_in.token_program,
            ),
            user_token_account_out: get_associated_token_address_with_program_id(
                &user,
                &cfg_out.mint,
                &cfg_out.token_program,
            ),
            vault_in: self.vault(in_idx as usize),
            vault_out: self.vault(out_idx as usize),
            user,
            token_program_in: cfg_in.token_program,
            token_program_out: cfg_out.token_program,
        };
        // `minimum_amount_out = 0`, like the Raydium reference: Titan's router
        // enforces slippage on the whole route, not per leg.
        Ok(swap_instruction(
            COFFER_PROGRAM_ID,
            &accounts,
            request.amount,
            0,
            in_idx,
            out_idx,
        ))
    }
}

#[async_trait]
impl AddressLookupTableTrait for CofferVenue {
    /// Every static account a swap on this pool touches: the pool, its
    /// vaults, mints and token programs, the program itself, and the pool's
    /// own on-chain ALT (`pool.lookup_table`) when one has been provisioned.
    /// The ALT is created AFTER the pool by `initialize_pool_alt` (once, then
    /// frozen), so its address is read from the pool state of the latest
    /// `update_state` on every call rather than captured at construction.
    async fn get_lookup_table_keys(
        &self,
        _accounts_cache: Option<&dyn AccountsCache>,
    ) -> Result<Vec<Pubkey>, TradingVenueError> {
        let mut keys = vec![self.pool_key, COFFER_PROGRAM_ID];
        for (i, slot) in self.pool.active_slots().iter().enumerate() {
            keys.push(slot.config.mint);
            keys.push(self.vault(i));
            if !keys.contains(&slot.config.token_program) {
                keys.push(slot.config.token_program);
            }
        }
        if self.pool.lookup_table != Pubkey::default() {
            keys.push(self.pool.lookup_table);
        }
        Ok(keys)
    }
}
