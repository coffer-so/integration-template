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
}

/// Off-chain state of one coffer pool.
#[derive(Clone)]
pub struct CofferVenue {
    /// The pool account address.
    pub pool_key: Pubkey,
    /// Decoded pool account (refreshed by `update_state`).
    pub pool: CofferPool,
    /// `Clock::unix_timestamp` the pool was last refreshed against; feeds the
    /// sell-off window arithmetic exactly as the program's `Clock::get()`.
    pub now: i64,
    /// Current epoch (for Token-2022 transfer-fee schedules).
    pub epoch: u64,
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
    /// The decoded pool.
    pub fn pool(&self) -> &CofferPool {
        &self.pool
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
                self.directions[i][j] = DirectionParams { curve };
            }
        }
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
            Err(ErrorCode::MaxSelloffExceeded) => {
                // Beyond the cap: the rate saturates at full fill.
                let probe = selloff_headroom(&self.pool, in_idx, self.now).map_err(map_error)?;
                let _ = probe;
                Ok(Some((u64::MAX, 1)))
            }
            Err(e) => Err(map_error(e)),
        }
    }

    /// Largest input `x <= hi` whose post-swap window position keeps the surge
    /// rate below 100% (the rate is non-decreasing in the fill).
    fn surge_exhaustion_point(&self, in_idx: u8, hi: u64) -> u64 {
        let cfg = &self.pool.tokens[in_idx as usize].config;
        let saturated = |x: u64| match self.selloff_after(in_idx, x) {
            Ok(Some((effective, cap))) => surge_rate(cfg, effective, cap) >= 1.0,
            _ => true,
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

    fn partial_fill(
        &self,
        request: &QuoteRequest,
        in_idx: u8,
        out_idx: u8,
        hi: u64,
    ) -> Result<QuoteResult, TradingVenueError> {
        let amount = self.largest_fillable(in_idx, out_idx, hi);
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

    /// Every ordered pair the program would accept and that can produce
    /// output: the input token must be active (`is_active` is an input-side
    /// kill switch; inactive tokens can still be bought) and have virtual
    /// liquidity, the output token must have virtual AND actual (LP-owned)
    /// balance — a sidelined output token reverts with
    /// `AmountOutExceedsBalance` at any size.
    ///
    /// A capped input token whose sell-off window has no headroom at the
    /// venue's `now` is likewise unavailable: every sell of it reverts with
    /// `MaxSelloffExceeded` until the window rotates, so the direction is
    /// left out rather than reported with an empty quotable range. Each pair
    /// is finally probed once at its largest admissible size (headroom, or the
    /// LP-balance cap) so a range that rounds to zero output is dropped too.
    fn directions_num(&self) -> Vec<(u8, u8)> {
        let slots = self.pool.active_slots();
        let mut out = Vec::new();
        for (i, a) in slots.iter().enumerate() {
            if !a.config.is_active || a.dynamics.virtual_balance == 0 {
                continue;
            }
            let Ok(in_idx) = u8::try_from(i) else {
                continue;
            };
            let headroom = match selloff_headroom(&self.pool, in_idx, self.now) {
                Ok(Some(h)) => h,
                Ok(None) => u64::MAX,
                Err(_) => continue,
            };
            if headroom == 0 {
                continue;
            }
            for (j, b) in slots.iter().enumerate() {
                if i == j || b.dynamics.virtual_balance == 0 || b.dynamics.actual_balance == 0 {
                    continue;
                }
                let Ok(out_idx) = u8::try_from(j) else {
                    continue;
                };
                // Largest admissible size: the window headroom, else the
                // largest input the LP balance can pay for.
                let hi = self.largest_fillable(in_idx, out_idx, headroom);
                let quotable = hi > 0
                    && self
                        .exact(in_idx, out_idx, hi)
                        .map(|o| o.amount_out_user > 0)
                        .unwrap_or(false);
                if quotable {
                    out.push((in_idx, out_idx));
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

        match self.exact(in_idx, out_idx, request.amount) {
            Ok(o) => {
                let selloff = o
                    .max_selloff_result
                    .map(|r| (r.effective_selloff, r.max_selloff_cap));
                if let Some((effective, cap)) = selloff {
                    let cfg = &self.pool.tokens[in_idx as usize].config;
                    if surge_rate(cfg, effective, cap) >= 1.0 {
                        // The surge rate has reached 100% of the output: every
                        // further input atom is taken in full by the fee, so
                        // the pool's usable liquidity ends where the rate hits
                        // 100%. Report that point as the fillable amount.
                        let hi = self.surge_exhaustion_point(in_idx, request.amount);
                        return self.partial_fill(&request, in_idx, out_idx, hi);
                    }
                }
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
    /// own on-chain ALT (`pool.lookup_table`, frozen at creation and already
    /// holding the same set) when initialised.
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
