//! Local-validator stand support: everything needed to create, seed and
//! configure real Coffer pools on a `solana-test-validator`, and to
//! send REAL routed transactions through the Titan router program built from
//! `program-template`.
//!
//! Instruction shapes mirror the contract's `#[derive(Accounts)]` contexts on
//! branch `main` (`initialize_pool`, `add_liquidity`, `set_max_selloff`,
//! `set_token_active`, `set_swaps_enabled`, `set_pool_enabled`). The stand
//! manifest (`scripts/local-stand/stand.json`) written by the `local-stand`
//! binary is what `tests/local_stand_matrix.rs` iterates.

use std::path::{Path, PathBuf};

use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};
use solana_instruction::{AccountMeta, Instruction};
use solana_pubkey::Pubkey;
use spl_associated_token_account::get_associated_token_address_with_program_id;

use crate::coffer::state::CofferPool;
use crate::coffer_venue::COFFER_PROGRAM_ID;
use crate::coffer_venue::instruction::{
    ADD_LIQUIDITY_DISCRIMINATOR, INITIALIZE_POOL_DISCRIMINATOR, ROUTER_INITIALIZE_DISCRIMINATOR,
    SET_MAX_SELLOFF_DISCRIMINATOR, SET_POOL_ENABLED_DISCRIMINATOR, SET_SWAPS_ENABLED_DISCRIMINATOR,
    SET_TOKEN_ACTIVE_DISCRIMINATOR,
};
use crate::swap_route::{ROUTE_WEIGHT_ALL, build_swap_leg, encode_swap_route_v3_data};
use crate::trading_venue::{QuoteRequest, TradingVenue, error::TradingVenueError};

/// The Titan router program id declared by `program-template`.
pub const ROUTER_PROGRAM_ID: Pubkey =
    Pubkey::from_str_const("T1TANpTeScyeqVzzgNViGDNrkQ6qHz9KrSBS4aNXvGT");
/// `TitanPda::SEED` in the program template.
pub const TITAN_PDA_SEED: &[u8] = b"titan_pda";
/// The live `CofferPoolConfig` cloned (with `protocol_admin` rewritten to the
/// local wallet) into the validator by `scripts/local-stand/up.sh`.
pub const LOCAL_CONFIG: Pubkey =
    Pubkey::from_str_const("E9K7CxPXpAEp49Usgxcxxm9DR8qUFXyvpwLjKrqqtrbY");
/// Default local RPC.
pub const LOCAL_RPC: &str = "http://127.0.0.1:8899";

/// `set_max_selloff` per-token argument (`SelloffParams` in the contract).
#[derive(
    BorshSerialize,
    BorshDeserialize,
    Serialize,
    Deserialize,
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
)]
pub struct SelloffParams {
    pub max_selloff_pct: u16,
    pub period_length: u32,
    pub fee_threshold_pct: u16,
    pub fee_slope_low_pct: u16,
    pub fee_slope_high_pct: u16,
    pub fee_slope_mid_pct: u16,
    pub fee_kink_pct: u8,
}

impl SelloffParams {
    pub const OFF: Self = Self {
        max_selloff_pct: 0,
        period_length: 0,
        fee_threshold_pct: 0,
        fee_slope_low_pct: 0,
        fee_slope_high_pct: 0,
        fee_slope_mid_pct: 0,
        fee_kink_pct: 0,
    };

    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        max_selloff_pct: u16,
        period_length: u32,
        fee_threshold_pct: u16,
        low: u16,
        mid: u16,
        high: u16,
        kink: u8,
    ) -> Self {
        Self {
            max_selloff_pct,
            period_length,
            fee_threshold_pct,
            fee_slope_low_pct: low,
            fee_slope_high_pct: high,
            fee_slope_mid_pct: mid,
            fee_kink_pct: kink,
        }
    }
}

// ---------------------------------------------------------------------------
// PDAs
// ---------------------------------------------------------------------------

pub fn bpt_mint(pool: &Pubkey) -> Pubkey {
    Pubkey::find_program_address(&[b"bpt_mint", pool.as_ref()], &COFFER_PROGRAM_ID).0
}

pub fn titan_pda() -> Pubkey {
    Pubkey::find_program_address(&[TITAN_PDA_SEED], &ROUTER_PROGRAM_ID).0
}

pub fn ata(owner: &Pubkey, mint: &Pubkey, token_program: &Pubkey) -> Pubkey {
    get_associated_token_address_with_program_id(owner, mint, token_program)
}

// ---------------------------------------------------------------------------
// Coffer program instructions
// ---------------------------------------------------------------------------

/// `initialize_pool(normalized_weights, initial_virtual_balances,
/// swap_fee_rate, pool_id, banned_extensions_override = None)`.
pub fn initialize_pool_ix(
    config: Pubkey,
    payer: Pubkey,
    pool_id: u64,
    mints: &[(Pubkey, Pubkey)], // (mint, token_program)
    weights: &[u64],
    virtual_balances: &[u64],
    swap_fee_rate: u32,
) -> (Pubkey, Instruction) {
    let (pool, _) = CofferPool::derive_pool_address(&config, pool_id, &COFFER_PROGRAM_ID);
    let mut data = INITIALIZE_POOL_DISCRIMINATOR.to_vec();
    BorshSerialize::serialize(&weights.to_vec(), &mut data).unwrap();
    BorshSerialize::serialize(&virtual_balances.to_vec(), &mut data).unwrap();
    data.extend_from_slice(&swap_fee_rate.to_le_bytes());
    data.extend_from_slice(&pool_id.to_le_bytes());
    data.push(0); // Option::None
    let mut accounts = vec![
        AccountMeta::new_readonly(config, false),
        AccountMeta::new(pool, false),
        AccountMeta::new(bpt_mint(&pool), false),
        AccountMeta::new(payer, true),
        AccountMeta::new_readonly(spl_token::ID, false),
        AccountMeta::new_readonly(spl_associated_token_account::ID, false),
        AccountMeta::new_readonly(solana_sdk::system_program::ID, false),
    ];
    for (mint, _) in mints {
        accounts.push(AccountMeta::new_readonly(*mint, false));
    }
    (
        pool,
        Instruction {
            program_id: COFFER_PROGRAM_ID,
            accounts,
            data,
        },
    )
}

/// `add_liquidity(token_amounts, minimum_bpt_amount)` with the
/// `remaining_accounts` layout `[user_i, vault_i]*n, mint_i*n, token_program_i*n`.
pub fn add_liquidity_ix(
    pool: Pubkey,
    user: Pubkey,
    mints: &[(Pubkey, Pubkey)],
    amounts: &[u64],
    minimum_bpt_amount: u64,
) -> Instruction {
    let bpt = bpt_mint(&pool);
    let mut data = ADD_LIQUIDITY_DISCRIMINATOR.to_vec();
    BorshSerialize::serialize(&amounts.to_vec(), &mut data).unwrap();
    data.extend_from_slice(&minimum_bpt_amount.to_le_bytes());
    let mut accounts = vec![
        AccountMeta::new(pool, false),
        AccountMeta::new(bpt, false),
        AccountMeta::new(ata(&user, &bpt, &spl_token::ID), false),
        AccountMeta::new_readonly(user, true),
        AccountMeta::new_readonly(spl_token::ID, false),
    ];
    for (mint, tp) in mints {
        accounts.push(AccountMeta::new(ata(&user, mint, tp), false));
        accounts.push(AccountMeta::new(
            CofferPool::derive_vault(&pool, mint, tp),
            false,
        ));
    }
    for (mint, _) in mints {
        accounts.push(AccountMeta::new_readonly(*mint, false));
    }
    for (_, tp) in mints {
        accounts.push(AccountMeta::new_readonly(*tp, false));
    }
    Instruction {
        program_id: COFFER_PROGRAM_ID,
        accounts,
        data,
    }
}

/// `set_max_selloff(params)` — pool admin only.
pub fn set_max_selloff_ix(
    pool: Pubkey,
    authority: Pubkey,
    params: &[SelloffParams],
) -> Instruction {
    let mut data = SET_MAX_SELLOFF_DISCRIMINATOR.to_vec();
    BorshSerialize::serialize(&params.to_vec(), &mut data).unwrap();
    Instruction {
        program_id: COFFER_PROGRAM_ID,
        accounts: vec![
            AccountMeta::new(pool, false),
            AccountMeta::new_readonly(authority, true),
        ],
        data,
    }
}

fn config_pool_authority(
    discriminator: [u8; 8],
    config: Pubkey,
    pool: Pubkey,
    authority: Pubkey,
    args: &[u8],
) -> Instruction {
    let mut data = discriminator.to_vec();
    data.extend_from_slice(args);
    Instruction {
        program_id: COFFER_PROGRAM_ID,
        accounts: vec![
            AccountMeta::new_readonly(config, false),
            AccountMeta::new(pool, false),
            AccountMeta::new_readonly(authority, true),
        ],
        data,
    }
}

/// `set_token_active(token_index, is_active)` — pool admin or protocol admin.
pub fn set_token_active_ix(
    config: Pubkey,
    pool: Pubkey,
    authority: Pubkey,
    token_index: u8,
    is_active: bool,
) -> Instruction {
    config_pool_authority(
        SET_TOKEN_ACTIVE_DISCRIMINATOR,
        config,
        pool,
        authority,
        &[token_index, is_active as u8],
    )
}

/// `set_swaps_enabled(enabled)` — pool admin or protocol admin.
pub fn set_swaps_enabled_ix(
    config: Pubkey,
    pool: Pubkey,
    authority: Pubkey,
    enabled: bool,
) -> Instruction {
    config_pool_authority(
        SET_SWAPS_ENABLED_DISCRIMINATOR,
        config,
        pool,
        authority,
        &[enabled as u8],
    )
}

/// `set_pool_enabled(enabled)` — protocol admin (the stand rewrites the cloned
/// config's `protocol_admin` to the local wallet).
pub fn set_pool_enabled_ix(
    config: Pubkey,
    pool: Pubkey,
    authority: Pubkey,
    enabled: bool,
) -> Instruction {
    config_pool_authority(
        SET_POOL_ENABLED_DISCRIMINATOR,
        config,
        pool,
        authority,
        &[enabled as u8],
    )
}

// ---------------------------------------------------------------------------
// Titan router (program-template) instructions
// ---------------------------------------------------------------------------

/// `initialize` — creates the TitanPDA.
pub fn router_initialize_ix(payer: Pubkey) -> Instruction {
    Instruction {
        program_id: ROUTER_PROGRAM_ID,
        accounts: vec![
            AccountMeta::new(payer, true),
            AccountMeta::new(titan_pda(), false),
            AccountMeta::new_readonly(solana_sdk::system_program::ID, false),
        ],
        data: ROUTER_INITIALIZE_DISCRIMINATOR.to_vec(),
    }
}

/// A single-leg `swap_route_v3` (mint 0 = input, mint 1 = output) exactly as
/// the program-template route harness builds it, with `payer` as both payer
/// and user. Suitable for a REAL send: the user's ATAs must exist and the
/// input one must be funded; the TitanPDA ATAs are created by the router
/// when missing.
pub fn build_route_ix(
    venue: &dyn TradingVenue,
    request: &QuoteRequest,
    payer: Pubkey,
) -> Result<Instruction, TradingVenueError> {
    let tokens = venue.get_token_info();
    let token_for = |mint: &Pubkey| {
        tokens
            .iter()
            .find(|t| t.pubkey == *mint)
            .ok_or(TradingVenueError::InvalidMint(mint.into()))
    };
    let (input_token, output_token) = (
        token_for(&request.input_mint)?,
        token_for(&request.output_mint)?,
    );
    let (in_tp, out_tp) = (
        input_token.get_token_program(),
        output_token.get_token_program(),
    );
    let pda = titan_pda();

    let mut accounts = vec![
        AccountMeta::new(payer, true),
        AccountMeta::new(payer, true),
        AccountMeta::new_readonly(pda, false),
        AccountMeta::new(ata(&payer, &request.input_mint, &in_tp), false),
        AccountMeta::new(ata(&payer, &request.output_mint, &out_tp), false),
        AccountMeta::new_readonly(spl_token::ID, false),
        AccountMeta::new_readonly(spl_token_2022::ID, false),
        AccountMeta::new_readonly(solana_sdk::system_program::ID, false),
        AccountMeta::new_readonly(spl_associated_token_account::ID, false),
        AccountMeta::new_readonly(ROUTER_PROGRAM_ID, false),
        AccountMeta::new_readonly(ROUTER_PROGRAM_ID, false),
        AccountMeta::new_readonly(ROUTER_PROGRAM_ID, false),
        AccountMeta::new(ata(&pda, &request.input_mint, &in_tp), false),
        AccountMeta::new(ata(&pda, &request.output_mint, &out_tp), false),
        AccountMeta::new_readonly(request.input_mint, false),
        AccountMeta::new_readonly(request.output_mint, false),
    ];
    let (spec, leg) = build_swap_leg(venue, request, pda, 0, 1, ROUTE_WEIGHT_ALL)?;
    accounts.extend(leg);
    Ok(Instruction {
        program_id: ROUTER_PROGRAM_ID,
        accounts,
        data: encode_swap_route_v3_data(request.amount, 2, &[spec]),
    })
}

// ---------------------------------------------------------------------------
// Stand manifest
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StandMint {
    pub name: String,
    pub address: String,
    pub decimals: u8,
    pub token_2022: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StandPool {
    /// Matrix case id, e.g. `b_thr0_kink50`.
    pub case: String,
    /// Human description of the configuration.
    pub description: String,
    pub address: String,
    /// Token names in slot order.
    pub tokens: Vec<String>,
    pub weights: Vec<u64>,
    pub swap_fee_rate: u32,
    /// The `set_max_selloff` params applied per slot (`OFF` when none).
    pub selloff: Vec<SelloffParams>,
    /// Slots deactivated with `set_token_active(false)` at setup.
    pub inactive_tokens: Vec<u8>,
    pub swaps_enabled: bool,
    pub pool_enabled: bool,
    /// Reserved for the dynamic (state-mutating) tests; the shared suite skips these.
    pub dynamic: bool,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct StandManifest {
    pub rpc: String,
    pub wallet: String,
    pub config: String,
    pub router: String,
    pub titan_pda: String,
    pub created_at_unix: i64,
    pub mints: Vec<StandMint>,
    pub pools: Vec<StandPool>,
}

impl StandManifest {
    pub fn path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/local-stand/stand.json")
    }

    pub fn load() -> Option<Self> {
        let text = std::fs::read_to_string(Self::path()).ok()?;
        serde_json::from_str(&text).ok()
    }

    pub fn save(&self) {
        std::fs::write(Self::path(), serde_json::to_string_pretty(self).unwrap()).unwrap();
    }

    pub fn mint(&self, name: &str) -> &StandMint {
        self.mints
            .iter()
            .find(|m| m.name == name)
            .expect("mint in manifest")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selloff_params_borsh_layout() {
        let p = SelloffParams::new(1_000, 3_600, 8_000, 100, 1_000, 3_000, 90);
        let bytes = borsh::to_vec(&p).unwrap();
        // u16 + u32 + u16 + u16 + u16(high) + u16(mid) + u8 = 15 bytes, mid AFTER high
        assert_eq!(bytes.len(), 15);
        assert_eq!(&bytes[10..12], &3_000u16.to_le_bytes());
        assert_eq!(&bytes[12..14], &1_000u16.to_le_bytes());
        assert_eq!(bytes[14], 90);
    }
}
